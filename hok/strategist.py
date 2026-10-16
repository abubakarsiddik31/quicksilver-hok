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

SYSTEM = """You are the General of a five-hero team in Honor of Kings 5v5 (a MOBA), played in real time through an API \
as one match of a tournament between autonomous agents (a university competition). The enemy team is commanded by \
another agent that sees only what its own heroes see, as you do. There is no chat. A win scores 1, a draw 0.5, a loss 0.

You never move a hero yourself. You write a PLAN. Below you, a fast executor applies it to every hero several times a \
second, a fight model scores each situation, and a second fast model (Jev) is asked about each hero about once a second. \
So you decide strategy; you are not racing the clock.

THE GAME, as measured in this server's matches
- One lane runs diagonally between the two bases. Each side has: a front tower (about 37% of the way from its base), a \
second tower (about 26%), a base turret, and a crystal. memory.lane gives positions as fractions of the lane: 0 is our \
base, 1 is theirs. Standing towers shoot enemy heroes, so a fight beside our own tower favours us and one beside \
theirs favours them.
- Win by destroying the enemy crystal. A match is 3000 steps (10 minutes). If no crystal falls, the team with more damage to \
enemy towers and crystal wins, then the team with more hero kills, else a draw. memory.projection says who wins now. \
Towers are worth 4000 and 6000 hp each, so one tower is most of what a 10-minute game decides. \
Kills only matter when structure damage is exactly equal, which almost never happens: THE GOAL IS STRUCTURE DAMAGE \
dealt minus taken. A kill is useful only as the time it buys to hit a structure, or to stop one of ours being hit.
- Hero speed is measured (memory.measured.hero_speed_units_per_s; a hero crosses the whole lane in roughly 25 s). A hero that dies is out for revive_s seconds (11 at level 2 up to about 35 \
at level 10); in that time the other side can take a tower freely: five heroes destroyed a 4000-hp tower in about 6 \
seconds. A wipe or a lopsided fight is therefore worth a tower; a lost fight is worth the same to them.
- Healing at our base is SLOW: memory.measured.base_heal_pct_max_hp_per_s (about 0.35% of max hp per second in our \
matches, so half hp takes over two minutes). "Go home and heal" takes a hero out for minutes, not seconds; in one match \
three hurt heroes sat at the spring while unseen attackers destroyed our crystal. A hurt hero is usually worth more \
standing at the structure that needs it than at the spring.
- Each enemy hero in reach costs about hp_lost_per_second_per_enemy_in_reach hp per second (measured). Retreat walks all \
the way to base, so a hero in contact cannot get away unless it outlasts the walk (survive_s vs flee_s in its reading).
- Structures are also hit by attackers the server does NOT list (no enemy hero in sight near the structure, sometimes \
with all five enemy heroes dead). They are killable: in one match the damage stopped the moment our heroes reached the \
crystal. In another, our 4000-hp front tower fell in 19 seconds at a steady ~225 hp/s while every enemy hero we could see \
was dead, because our heroes had walked away from it. memory.tempo.our_structure_hp_lost_with_no_enemy_hero_in_sight \
measures this in the current match. A structure being hit with no enemy visible is NOT safe: someone has to stand on it.
- Past matches: against a real opponent our team reached the centre spread out, retreated at the first bad number, and \
died while fleeing having dealt almost no damage; the enemy was at the centre by 18 seconds. Fighting at our own tower, \
together, is the plan we are trying instead of meeting them in the open.

YOUR TIMING
- Your plans arrive 10 to 25 seconds after the state you read (last_consult_latency_s). Plan for the next minute, not \
the next second: the executor already handles what cannot wait (answering a hit on our structures, punishing when \
enemies are dead and cannot be back in time, fighting when cornered, holding at the tower). Do not order a push \
because enemies were dead 20 seconds ago; set the posture that fits what is likely over the next minute.

WHAT YOUR PLAN DOES
- posture "defend": no hero walks past our front tower unless an enemy is in its reach; they gather at the tower and \
fight what comes. "balanced": heroes advance only while the force expected where they are going is one they beat \
(the fight model's `ahead` reading), otherwise they hold at the tower. "pressure": heroes advance regardless of the \
`ahead` reading (the `here` readings still decide fights and retreats).
- risk: a hero fights on while (its local power ratio x its hp share) >= risk. Below 1 is bold, 1 parity, above 1 cautious.
- jev_trust: Jev's `hold`/`retreat` answers stop a hero engaging only if Jev is at least this sure. Raise it if Jev has \
been wrong, lower it if its cautions have been right.
- heroes: force one hero to engage or retreat. Use rarely, with a stated reason.
- review_in_s: when you want to be consulted again if nothing happens. Events (a death, a fight starting or ending, a \
tower hit) wake you anyway.

HOW TO WIN (learned from our matches, all lost so far)
- Towers fall only when their defenders are away or dead. In our last match we won the fights (10 kills to 7) and still \
lost: we dealt 0 structure damage, sat at our base healing while 1 or 2 enemies took both our towers and most of the \
crystal, and then left the crystal at 3045 hp to chase a push and lost it. Kills are worth nothing unless they become \
tower damage, and every structure hit on us must be answered.
- 1. DEFEND what is being hit. memory.our_structures_under_attack lists it. The executor sends heroes to a structure \
under attack when the fight there is winnable (the structure counts on our side); keep `risk` low enough that it does.
- 2. FIGHT AT OUR FRONT TOWER. That is where the tower fights for us and where we can punish from.
- 3. PUNISH. memory.tempo.punish_window puts the numbers side by side: how many enemies are dead and when each is back \
(dead_back_in_s), how long each of our living heroes needs to walk to their nearest structure (our_living_heroes_walk_s), \
and how long that structure lasts at the fastest siege seen this match (target_falls_in_s_at_fastest_siege). A push is \
worth it when enough of our heroes arrive, and the structure falls (or loses a large share), before enough of theirs are \
back to make the fight there one we lose. Kills that are never followed by a push are worth almost nothing on the \
tie-break. Your plan reaches the heroes 10-25 s late: judge the window as it will be then. Return to `balanced` once \
they are back.
- 4. NEVER LEAVE OUR CRYSTAL OR A TOWER EXPOSED to go somewhere else. Read `side` on every structure: OURS or THEIRS. \
Do not confuse them.
- 5. A LEAD IS MEASURED IN SECONDS, NOT HP. memory.tempo.structure_margin_in_s_of_fastest_siege says how many \
seconds of the fastest siege known (fastest_siege_known_hp_per_s) our structure lead absorbs; compare it with game_seconds_left. In our last match \
we held a 151-hp lead with 12 more kills, called the game "secured" for ten plans while defending and dealing nothing, \
and lost: 151 hp was under one second of their siege, and they took 4000 hp in the last minute. A lead smaller than \
what they can deal in the time left is not a lead. Protect a real lead by standing on whatever of ours is being hit; \
grow a thin one with the punish window; if behind, structure damage is the only thing that counts and kills do not.

HOW TO REASON
- Read recent_fights: expected_ratio against realised_ratio says whether the fight model is optimistic. \
fight_model_calibration_by_place splits it by our_side / middle / their_side of the lane: fights beside our own tower \
usually go better than fights in the open. Below 1 there means fights there go worse than predicted.
- jev_track_record scores Jev's fight calls against what happened (Brier: 0 perfect, 0.25 a coin flip). Raise \
jev_trust when Jev is no better than a coin; lower it when its calls have been right.
- earlier_plans lists what each plan achieved while it was in force (structure damage, kills, deaths). Judge your own \
plans by it and change course on evidence, not on mood. Do not change the plan without a reason.
- `tactician` is Jev's latest view (stance per hero with its probability, team posture, the chance the enemy team is \
about to arrive, the chance we win the engagement). Treat it as a second opinion that is fast but unproven here.
- Enemy heroes out of sight are not gone: last_seen_s_ago says how old a sighting is. Heroes never seen are expected \
to mirror our own advance.

What you do not know: hero kits, skill ranges and cooldowns, what exactly the unlisted attackers are (how many, where \
they walk), and the opponent's style beyond what the memory \
shows. Do not invent them.

Answer with one JSON object only:
{"posture": "pressure"|"balanced"|"defend",
 "risk": number 0.4-3.0,
 "heroes": {"<seat>": "engage"|"retreat"} (only heroes you want to override, otherwise {}),
 "review_in_s": number 4-40,
 "jev_trust": number 0.5-0.95,
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
    extra: dict[str, Any] | None = None,
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
            **(extra or {}),
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
        extra: dict[str, Any] | None = None,
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
                              history=history, readings=readings, latency=latency, extra=extra)
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
