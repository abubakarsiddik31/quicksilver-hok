"""The General: an LLM consulted in a background thread, woken by events, publishing a :class:`hok.plan.Plan`.

The fast loop (``HokAgent.choose_action``) never waits on it. Calls never overlap; triggers that arrive while one
is running are coalesced and fire as soon as it finishes, against fresh memory. A minimum gap, per-kind cooldowns,
an error backoff and a call cap bound the cost. Every failure leaves the previous plan in force, and with no model
the default plan runs for the whole match (the fight model still decides each hero's engage / retreat).
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from hok import trace
from hok.memory import MatchMemory
from hok.plan import DEFAULT_PLAN, Plan, parse_plan, plan_schema

MIN_GAP_S = 3.0
MODEL_TIMEOUT_S = 30.0
MAX_CALLS = 300
BACKOFF_FIRST_S, BACKOFF_MAX_S = 5.0, 60.0
ENDGAME_S = 90
EXPIRE_AFTER_REVIEWS = 3  # a plan whose author missed this many of its own reviews loses its authority
HISTORY = 3
EVENT_COOLDOWN_S = {
    "our_hero_died": 6.0,
    "our_kill": 8.0,
    "enemy_first_seen": 10.0,
    "fight_started": 6.0,
    "fight_ended": 6.0,
    "our_structure_hit": 15.0,
    "our_structure_destroyed": 3.0,
    "enemy_structure_destroyed": 3.0,
}

SYSTEM = """You are the General of a five-hero team in Honor of Kings 5v5 (a MOBA), played in real time through an API as \
one match of a tournament between autonomous agents. The enemy team is commanded by another agent that sees only what \
its own heroes see, as you do. There is no chat. A win scores 1, a draw 0.5, a loss 0.

You never move heroes yourself. You write a plan. A fast executor applies it to every hero several times a second, so \
your thinking time costs nothing; a fight model scores each situation from hp and level, from what is known of the \
enemy, and from the standing structures.

How the match is decided: destroy the enemy crystal. If neither crystal falls by the step or time limit, the team that \
dealt more damage to enemy towers and crystal wins, then the team with more hero kills, else a draw. Memory.projection \
tells you who is winning if it ended now. So structure damage is the currency and every death is a cost: a team that is \
ahead late should protect the lead and avoid coin-flip fights; a team that is behind must create real chances, not \
gamble early. Be patient. Do not push because the team looks strong, do not hide when it is ahead on the numbers, and \
keep re-checking while a fight is on.

What the executor does with your plan, hero by hero:
- engage: fight what is in reach, else walk toward the nearest enemy structure;
- retreat: walk home and heal;
- by default (auto) the fight model chooses. A hero fights while (local power ratio x its hp share) >= risk. When a hero \
has nothing in reach it is judged against the force expected at its destination (the "ahead" reading), counting only the \
allies who will be there, and it waits otherwise. So risk below 1 is bold, 1 is parity, above 1 is cautious.
- posture "defend" sends heroes home when no enemy is in sight; "pressure" and "balanced" leave it to the fight model.
- you may force engage or retreat for single heroes (use rarely, for a reason you can state).

Reading the numbers you are given (all derived from engine data only):
- fight_model_now: per hero, ratio (our local power over theirs, already corrected by calibration), allies near, \
visible_enemies, enemy_structures counted, hidden_power_counted (the believed share of enemies out of sight), hp_share.
- fight_model_calibration: below 1 means fights have gone worse than the model predicted; above 1, better. recent_fights \
lists expected_ratio against realised_ratio.
- earlier_plans: what each plan achieved while it was in force (structure damage dealt and taken, kills, deaths). Judge \
your own past plans by these and change course on evidence, not on mood. Do not change the plan without a reason.
- enemy_heroes_known carries last_seen_s_ago: an enemy out of sight is not gone, and the longer ago, the less you know.

What you do not know: hero kits, ranges and cooldowns, minion waves, the map. Do not invent them; reason from the \
numbers and the history only. Positions are game units.

Answer with one JSON object only:
{"posture": "pressure"|"balanced"|"defend",
 "risk": number 0.4-3.0,
 "heroes": {"<seat>": "engage"|"retreat"} (only heroes you want to override, otherwise {}),
 "review_in_s": number 4-40 (when to be consulted again if nothing happens),
 "assessment": "one or two sentences: what is going on, and why this plan"}"""


def build_user(
    memory: MatchMemory,
    *,
    triggers: list[str],
    plan: Plan,
    plan_age_s: float,
    history: list[str],
    readings: list[dict[str, Any]],
    latency: float | None,
) -> str:
    return json.dumps(
        {
            "consulted_because": triggers,
            "current_plan": plan.as_json(),
            "plan_age_s": round(plan_age_s, 1),
            "last_assessments": history,
            "last_consult_latency_s": latency,
            "memory": memory.briefing(),
            "fight_model_now": readings,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


class Strategist:
    def __init__(
        self,
        client: Any | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        min_gap: float = MIN_GAP_S,
        max_calls: int = MAX_CALLS,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.client = client
        self._clock = clock
        self._min_gap = min_gap
        self._max_calls = max_calls
        self._log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self._lock = threading.Lock()
        self._plan = DEFAULT_PLAN
        self._version = 0
        self._plan_at = clock()
        self._busy = False
        self._last_start = -1e9
        self._calls = 0
        self._errors = 0
        self._backoff_until = 0.0
        self._pending: list[str] = []
        self._started = False
        self._kind_at: dict[str, float] = {}
        self._endgame_sent = False
        self._history: list[str] = []
        self._latency: float | None = None
        self._notes: list[tuple] = []

    def current(self) -> Plan:
        """The plan in force. One that has gone unreviewed for several of its own review intervals relaxes to
        the default: a dead or silent General must never leave the team stuck on a stale, passive plan."""
        with self._lock:
            if self._plan is not DEFAULT_PLAN and self._clock() - self._plan_at > EXPIRE_AFTER_REVIEWS * self._plan.review_in_s:
                return DEFAULT_PLAN
            return self._plan

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @property
    def enabled(self) -> bool:
        return self.client is not None and not getattr(self.client, "disabled", None)

    def drain_notes(self, memory: MatchMemory) -> None:
        """Move plan notes written by the consult thread into memory, on the caller's (executor) thread."""
        with self._lock:
            notes, self._notes = self._notes, []
        for n in notes:
            memory.note_plan(*n)

    def notify(self, reason: str) -> None:
        with self._lock:
            self._add(reason)

    def _add(self, reason: str) -> None:
        if reason not in self._pending and len(self._pending) < 8:
            self._pending.append(reason)

    def _detect(self, events: list[str], seconds_left: float | None, now: float) -> None:
        if not self._started:
            self._started = True
            self._add("match_start")
        for ev in events:
            kind = ev.split(":", 1)[0]
            cooldown = EVENT_COOLDOWN_S.get(kind)
            if cooldown is not None and now - self._kind_at.get(kind, -1e9) >= cooldown:
                self._kind_at[kind] = now
                self._add(ev)
        if isinstance(seconds_left, (int, float)) and seconds_left <= ENDGAME_S and not self._endgame_sent:
            self._endgame_sent = True
            self._add(f"endgame:{int(seconds_left)}s_left")
        if not self._busy and now >= self._plan_at + self._plan.review_in_s:
            self._add("scheduled_review")

    def maybe_update(
        self,
        memory: MatchMemory,
        events: list[str],
        readings: list[dict[str, Any]],
        seats: list[int],
        *,
        sync: bool = False,
    ) -> None:
        """Note triggers and, if one is due, start a background consult. Never blocks."""
        if not self.enabled:
            return
        self.drain_notes(memory)
        now = self._clock()
        with self._lock:
            self._detect(events, memory.seconds_left, now)
            if (
                not self._pending
                or self._busy
                or self._calls >= self._max_calls
                or now < self._backoff_until
                or now - self._last_start < self._min_gap
            ):
                return
            self._busy = True
            self._last_start = now
            self._calls += 1
            triggers, self._pending = self._pending, []
            previous = self._plan
            history = list(self._history)
            age = now - self._plan_at
            latency = self._latency
        try:
            user = build_user(memory, triggers=triggers, plan=previous, plan_age_s=age,
                              history=history, readings=readings, latency=latency)
        except Exception as exc:  # noqa: BLE001 - never break the loop
            with self._lock:
                self._busy = False
            self._log(f"[general] could not build prompt: {exc!r}")
            return
        args = (user, previous, triggers, seats, memory)
        if sync:
            self._run(*args)
        else:
            threading.Thread(target=self._run, args=args, daemon=True).start()

    def _run(self, user: str, previous: Plan, triggers: list[str], seats: list[int], memory: MatchMemory) -> None:
        started = self._clock()
        parsed: Plan | None = None
        text = None
        try:
            hard = any(t.startswith(("match_start", "endgame")) for t in triggers)
            text = self.client.generate(  # type: ignore[union-attr]
                user, system=SYSTEM, schema=plan_schema(seats),
                level="high" if hard else "medium", timeout=MODEL_TIMEOUT_S,
            )
            parsed = parse_plan(text, previous, seats)
        except Exception as exc:  # noqa: BLE001 - the General must never break the loop
            self._log(f"[general] consult failed: {exc!r}")
        trace.write("consult", triggers=triggers, prompt=user, response=text)  # what it saw and what it said
        done = self._clock()
        with self._lock:
            self._latency = round(done - started, 1)
            if parsed is not None:
                self._plan, self._plan_at = parsed, done
                self._version += 1
                self._errors = 0
                self._pending = [r for r in self._pending if r != "scheduled_review"]
                self._history = [*self._history, parsed.assessment][-HISTORY:]
            else:
                self._errors += 1
                self._backoff_until = done + min(BACKOFF_MAX_S, BACKOFF_FIRST_S * 2 ** min(self._errors - 1, 4))
                for t in triggers[:4]:
                    self._add(t)
            self._busy = False
            version = self._version
        last = getattr(self.client, "last", None)
        trace.write("plan", ok=parsed is not None, triggers=triggers, latency_s=self._latency,
                    version=version, plan=parsed.as_json() if parsed else None,
                    outcome=getattr(last, "kind", None), detail=getattr(last, "detail", None),
                    shape=getattr(last, "shape", None), thought_tokens=getattr(last, "thought_tokens", None))
        if parsed is not None:
            with self._lock:
                self._notes.append((version, parsed.assessment, parsed.posture, parsed.risk))
            self._log(f"[general] plan v{version} ({self._latency}s, {','.join(triggers)}): {parsed.assessment}")
        else:
            self._log(f"[general] no usable plan ({self._errors} in a row); keeping the previous one")
