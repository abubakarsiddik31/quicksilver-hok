"""Honor of Kings 5v5 player: a fast executor under a slow General.

The server runs in real time (a step every 0.2 s whether we act or not), and every macro we send stays that hero's
standing order, so the team keeps playing while anything slower thinks. The layers:

* **Memory** (:mod:`hok.memory`) is updated on every new step and keeps the whole match.
* **Fight model** (:mod:`hok.battle`) reads, for each hero, the local force on each side including the believed
  share of the enemy out of sight, and says whether the hero should take or stay in the fight.
* **General** (:mod:`hok.strategist`, an LLM, off unless ``HOK_LLM=1``) runs in a background thread, woken by
  events, and publishes a plan: posture, the risk bar the fight model applies, per-hero overrides. With no model
  the default plan runs and the fight model alone chooses engage or retreat.
* **Executor** (here) maps each hero's stance onto the macros the server offers *right now*, sends only changes,
  one hero per call, and returns ``WAIT`` otherwise.
* **Floor** (``floor_macro``) is only the last resort when the model code cannot decide (a hero with no position,
  an exception). Every use is traced as a ``fallback``.

The only hard limits here are safety invariants: never send an order that is not on offer, never leave a pending
hero without one in lockstep, and never let tracing or a bad field cost a match.
"""

from __future__ import annotations

import atexit
import dataclasses
import hashlib
import os
import sys
import time
from typing import Any

from altruagent import WAIT, DecisionContext, GameState, LegalAction
from common.jev import JevClient
from common.llm import GeminiClient
from hok import trace
from hok.battle import Judge, Reading
from hok.gemini import GeneralModel
from hok.memory import FIGHT_R, MatchMemory, dist, pos
from hok.observation import (
    distance,
    is_realtime,
    offers_by_seat,
    pending_seats,
    raw,
    seat_summary,
    standing_orders,
    team_view,
)
from hok.strategist import Strategist
from hok.tactician import Tactician

STATE_EVERY_S = 1.0
HOLD_BAND_S = (
    1.0  # a held hero may drift this many seconds of walking either side of the rally point
)
RESEND_GUARD_S = 1.0  # roughly the time for a sent order to appear in the next state read
SNAPSHOT_EVERY_S = 30.0
HOLD = "hold"
SKILLS = ("macro_skill3", "macro_skill2", "macro_skill1")


def engage_macro(macros: dict[str, str], reach: float = FIGHT_R) -> str | None:
    """The macro that fights: a skill on a target in reach, else a basic attack, else walk on the enemy. The
    server offers a macro toward the nearest target at ANY distance (match 1: skills offered 60,000 units
    away), so an offered macro is not an in-reach one; its "N away" label is checked against the learned
    engagement radius."""
    for m in (*SKILLS, "macro_attack"):
        if m in macros and distance(macros[m]) <= reach:
            return m
    if "macro_advance" in macros:
        return "macro_advance"
    return next(iter(macros), None)


def floor_macro(summary: dict[str, Any], macros: dict[str, str]) -> str | None:
    """Last resort when the model code cannot decide: retreat when nearly dead, else fight or push."""
    if not macros:
        return None
    me = summary.get("my_hero") or {}
    hp, max_hp = me.get("hp"), me.get("max_hp")
    if (
        isinstance(hp, (int, float))
        and isinstance(max_hp, (int, float))
        and max_hp > 0
        and hp < 0.25 * max_hp
        and "macro_retreat" in macros
    ):
        return "macro_retreat"
    return engage_macro(macros)


def hold_order(raw_actions: list[LegalAction], seat: int) -> dict[str, Any] | None:
    """The do-nothing button for this hero, if the server lists it. A raw action plays once and clears the
    standing order, and a hero with no standing order stands still: that is how a hero waits."""
    for action in raw_actions:
        payload = dict((action.input or {}).get("action") or {})
        if payload.get("which_button") == 0:
            return {"which_button": 0, "seat": seat}
    return None


def complete_raw(raw_actions: list[LegalAction], seat: int) -> dict[str, Any]:
    """A playable action for a hero with no macro (lockstep only): the do-nothing button if listed, else
    the first button with the first legal value of each field."""
    for action in raw_actions:
        payload = dict((action.input or {}).get("action") or {})
        if payload.get("which_button") == 0:
            return {"which_button": 0, "seat": seat}
    if not raw_actions:
        return {"which_button": 0, "seat": seat}
    payload = dict((raw_actions[0].input or {}).get("action") or {})
    out: dict[str, Any] = {"seat": seat}
    for key, value in payload.items():
        if key.startswith("legal_"):
            if isinstance(value, list) and value:
                out[key[len("legal_") :]] = value[0]
        elif key != "seat":
            out[key] = value
    return out


class HokAgent:
    def __init__(self, client: Any | None = None, jev: Any | None = None, *, log=None) -> None:
        self._log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self.memory = MatchMemory()
        self.judge = Judge(self.memory)
        self.general = Strategist(client, log=self._log)
        self.tactician = Tactician(jev, log=self._log)
        self._readings: dict[int, Reading] = {}
        self._ahead: dict[int, Reading] = {}
        self._defend: dict[int, Reading] = {}
        self._sent: dict[int, tuple[str, float]] = {}
        self._finished = False
        atexit.register(self._write_final)
        self._meta_done = False
        self._baseline_camp = os.environ.get("HOK_BASELINE_CAMP", "").strip()
        self._step_seen = -1
        self._last_state_at = 0.0
        self._last_snapshot_at = 0.0
        self.fallbacks: dict[str, int] = {}
        self._log(
            f"[hok] General {'on' if client is not None else 'off (HOK_LLM=1 turns it on)'}; "
            f"Jev {'on' if jev is not None else 'off (needs TYPESAFE_API_KEY; HOK_JEV=0 disables)'}"
        )

    # -- the runner's hooks --------------------------------------------------------------

    def choose_action(self, state: GameState, context: DecisionContext):
        realtime = is_realtime(state)
        offers = offers_by_seat(getattr(state, "legal_actions", None))
        self._write_meta(state, context)
        self._observe(state, offers)
        self._record(state)
        if not offers:
            if realtime:
                return WAIT
            raise ValueError("Honor of Kings asked for a move but listed no legal actions")
        standing = standing_orders(state)
        pending = pending_seats(state, offers)
        if realtime and not pending:
            pending = sorted(offers)  # real time accepts an order for any hero at any step
        plan = self.general.current()
        now = time.monotonic()
        for seat in pending:
            macros = offers[seat]["macros"]
            want, source = self._decide(state, seat, macros, plan)
            if want == HOLD:
                held = hold_order(offers[seat]["raw"], seat)
                if held is not None:  # the server lists a do-nothing button (it did not in match 1)
                    if realtime and (seat not in standing or self._recently_sent(seat, HOLD, now)):
                        continue
                    self._sent[seat] = (HOLD, now)
                    trace.write("orders", order=held, source=source, plan_v=self.general.version)
                    return held
                want = self._hold_macro(seat, macros, standing)
                source += "/hold"
                if want is None:
                    continue  # inside the band and already going the right way: leave it
            if want is None:
                if realtime:
                    continue  # a dead hero: nothing to send
                return complete_raw(offers[seat]["raw"], seat)
            if realtime and standing.get(seat) == want:
                continue
            if realtime and self._recently_sent(seat, want, now):
                continue  # sent a moment ago; the state read may not show it yet
            self._sent[seat] = (want, now)
            order = {"macro": want, "seat": seat}
            r = self._readings.get(seat)
            trace.write(
                "orders",
                order=order,
                source=source,
                plan_v=self.general.version,
                reading=r.as_json() if r else None,
            )
            return order
        if realtime:
            return WAIT
        seat = pending[0] if pending else sorted(offers)[0]
        return complete_raw(offers[seat]["raw"], seat)

    def _recently_sent(self, seat: int, want: str, now: float) -> bool:
        """The same order to the same hero within one round trip of the server's state: an idempotency guard."""
        prev = self._sent.get(seat)
        return prev is not None and prev[0] == want and now - prev[1] < RESEND_GUARD_S

    def on_action_result(self, result: dict, context: DecisionContext) -> None:
        # Every answer carries the whole game state (30 of 33 MB in match 1); the ticks and snapshots hold it.
        slim = (
            {k: v for k, v in result.items() if k != "state"}
            if isinstance(result, dict)
            else result
        )
        trace.write("result", result=slim)
        if isinstance(result, dict) and result.get("error") == "GAME_ALREADY_COMPLETE":
            self._write_final("game_already_complete")

    def _write_final(self, why: str = "exit") -> None:
        """The match summary, once: on the server saying the game is over, or when this process exits."""
        if self._finished or self.memory.step == 0:
            return
        self._finished = True  # the runner's own end-of-game hook never reaches this process
        try:
            m = self.memory
            trace.write(
                "final",
                step=m.step,
                projection=m.projection(),
                dealt=round(m.dealt),
                taken=round(m.taken),
                kills=m.team_kills,
                deaths=m.team_deaths,
                calibration=round(m.calibration, 3),
                radius=round(m.R),
                fights=[f.as_json(m.step) for f in m.fights],
                fallbacks=self.fallbacks,
                general_calls=self.general.version,
                why=why,
                jev=m.jev_record(),
                calibration_by_place={k: len(v) for k, v in m._calib_by.items()},
            )
        except Exception:  # noqa: BLE001 - bookkeeping must never matter
            pass

    # -- observing -----------------------------------------------------------------------

    def _observe(self, state: GameState, offers: dict[int, dict[str, Any]]) -> None:
        """Fold a new step into memory, re-read the fight, and let the General be woken."""
        step = int(getattr(state, "state_version", 0) or 0)
        if step == self._step_seen:
            return
        self._step_seen = step
        seats = sorted(offers)
        self._tick(state, step)
        try:
            for entry in (
                offers.values()
            ):  # a macro is only offered with a target in reach: that is the reach
                for m in (*SKILLS, "macro_attack"):
                    if m in entry["macros"]:
                        self.memory.observe_reach(distance(entry["macros"][m]), m)
            events = self.memory.update(team_view(state))
            self._readings, self._ahead, self._defend = {}, {}, {}
            for seat in seats:
                r = self.judge.read(seat)
                if r is not None:
                    self._readings[seat] = r
                a = self.judge.read_ahead(seat)
                if a is not None:
                    self._ahead[seat] = a
            fight = self.memory.open_fight
            if fight is not None and fight.predicted is None:
                fight.predicted = self.judge.expected_ratio(list(self._readings.values()))
            advice = self.tactician.current()
            if (
                fight is not None
                and fight.jev_p is None
                and advice is not None
                and advice.win_fight is not None
            ):
                fight.jev_p = (
                    advice.win_fight
                )  # scored against the real exchange when the fight ends
            self.tactician.maybe_update(
                self.memory, self._readings, standing_orders(state), self.general.current()
            )
            self.general.maybe_update(
                self.memory,
                events,
                [r.as_json() for r in (*self._readings.values(), *self._ahead.values())],
                seats,
                extra={
                    "tactician": (
                        self.tactician.current().as_json() if self.tactician.current() else None
                    )
                },
            )
            for ev in events:
                trace.write("event", step=step, event=ev)
        except Exception as exc:  # noqa: BLE001 - the model layers must never cost a match
            self._note_fallback("observe_error", repr(exc))

    # -- deciding ------------------------------------------------------------------------

    def _decide(
        self, state: GameState, seat: int, macros: dict[str, str], plan
    ) -> tuple[str | None, str]:
        if not macros:
            return None, "none"
        if self._baseline_camp and str(self.memory.camp) == self._baseline_camp:
            # A/B harness: this camp plays the starter's plain rules, so a self-play match measures our agent
            # against a fixed baseline instead of against itself. Off unless HOK_BASELINE_CAMP is set.
            return floor_macro(seat_summary(state, seat), macros), "baseline"
        try:
            reading = self._readings.get(seat)
            stance = plan.stance(seat)
            source = "model"
            hold = False
            siege = self.memory.defend_point()
            if (
                stance == "auto"
                and siege is not None
                and reading is not None
                and reading.in_reach == 0
            ):
                # One of our structures is losing hp and nothing is on this hero, so the fight it is choosing is
                # the one at that structure, not the empty spot it stands on (where a lone hero only ever sees
                # itself against the minimum credible threat and walks home while the tower falls). Judge that
                # fight, with the structure on our side; if it is winnable, go to it.
                at = self.judge.read_toward(seat, siege, "defend")
                if at is not None:
                    self._defend[seat] = at
                    if self.judge.engage(at, plan.risk):
                        toward = self._toward(seat, siege, macros)
                        if toward is not None:
                            return toward, "model/defend"
            if stance == "auto":
                if reading is None:
                    raise LookupError("no reading for this hero")
                stance = "engage" if self.judge.engage(reading, plan.risk) else "retreat"
                if stance == "retreat" and not self.judge.can_escape(reading):
                    stance, source = (
                        "engage",
                        "model/cornered",
                    )  # cannot outrun them: fight, don't flee
                if stance == "engage":
                    # Jev may make a hero more careful (never less): a confident answer, fresh, from the plan's trust
                    caution = self.tactician.caution(seat, plan.jev_trust)
                    if caution == "retreat" and self.judge.can_escape(reading):
                        stance, source = "retreat", "jev"
                    elif caution == "hold":
                        hold, source = True, "jev"
            if (
                stance == "retreat"
                and reading is not None
                and reading.in_reach == 0
                and plan.stance(seat) == "auto"
            ):
                # Nobody is on this hero: "retreat" would only be a trip home to heal. Go only if healing beats
                # staying: a round trip of 2 x flee_s misses that share of the time between fights, so a hero
                # with more hp than 1 - trip/interval is worth more at the tower than on the road.
                trip = 2.0 * reading.flee_s
                if reading.hp_share >= 1.0 - trip / max(self.memory.fight_interval(), 1e-6):
                    stance, hold, source = "engage", True, source + "/stay"
            if stance == "retreat" and "macro_retreat" in macros:
                return "macro_retreat", source
            want = engage_macro(macros, self.memory.R)
            if want == "macro_advance" and plan.stance(seat) == "auto":
                # One of our structures is being hit: that is a fight to answer, whatever the posture. Go if the
                # force there (with the structure on our side) is one this hero's side can beat, else hold.
                if siege is not None:
                    at = self._defend.get(seat) or self.judge.read_toward(seat, siege, "defend")
                    if at is not None:
                        self._defend[seat] = at
                        if self.judge.engage(at, plan.risk):
                            toward = self._toward(seat, siege, macros)
                            return (toward or HOLD), "model/defend"
                        return HOLD, "model/defend"
                # Nothing in reach: this hero is about to walk into something. "defend" never walks in; "pressure"
                # always does; otherwise it goes only if the force expected where it is going is one it can beat.
                ahead = self._ahead.get(seat)
                if (
                    hold
                    or plan.posture == "defend"
                    or (
                        plan.posture != "pressure"
                        and ahead is not None
                        and not self.judge.engage(ahead, plan.risk)
                    )
                ):
                    return HOLD, source
            if want is not None:
                return want, source
        except Exception as exc:  # noqa: BLE001
            self._note_fallback("decide_error", repr(exc), seat)
        want = floor_macro(seat_summary(state, seat), macros)
        self._note_fallback("floor", "model could not decide", seat, want)
        return want, "floor"

    def _toward(self, seat: int, point: tuple[float, float], macros: dict[str, str]) -> str | None:
        """The lane direction that takes this hero to ``point``: advance and retreat are fixed joystick
        directions along the lane, so a hero already past a structure of ours walks back to it, not on."""
        m = self.memory
        if dist(pos(m.ours.get(seat)), point) <= m.R:
            return None  # already there: the hero's own reading (with the structure on our side) takes over
        here, there = m.progress(pos(m.ours.get(seat))), m.progress(point)
        if here is None or there is None:
            return "macro_advance" if "macro_advance" in macros else None
        want = "macro_advance" if here < there else "macro_retreat"
        return want if want in macros else None

    def _hold_macro(
        self, seat: int, macros: dict[str, str], standing: dict[int, str]
    ) -> str | None:
        """Hold position without a wait command: keep the hero within a second's walk of our front tower by
        stepping back when it is past the point and forward when it is short of it. ``None`` means it is
        inside the band and already heading the right way, so leave its order alone."""
        m = self.memory
        me = m.ours.get(seat)
        lane, rally = m.lane(), m.front_tower()
        here, there = m.progress(pos(me) if me else None), m.progress(rally)
        if lane is None or here is None or there is None:
            return None
        band = m.speed * HOLD_BAND_S / max(1.0, dist(*lane))
        if here > there + band and "macro_retreat" in macros:
            return "macro_retreat"
        if here < there - band and "macro_advance" in macros:
            return "macro_advance"
        if standing.get(seat) in ("macro_advance", "macro_retreat"):
            return None
        want = "macro_advance" if here < there else "macro_retreat"
        return want if want in macros else None

    def _note_fallback(
        self, kind: str, why: str, seat: int | None = None, chose: str | None = None
    ) -> None:
        n = self.fallbacks[kind] = self.fallbacks.get(kind, 0) + 1
        if n <= 3 or n % 50 == 0:
            self._log(f"[hok] fallback {kind} x{n}: {why}")
        trace.write("fallback", kind=kind, seat=seat, why=why, chose=chose)

    # -- recording -----------------------------------------------------------------------

    def _write_meta(self, state: GameState, context: DecisionContext) -> None:
        """Who played what, with which code and settings: a trace must identify its own match."""
        if self._meta_done:
            return
        self._meta_done = True
        try:
            ctx = {k: v for k, v in dataclasses.asdict(context).items() if k != "game_config"}
        except Exception:  # noqa: BLE001
            ctx = {"repr": repr(context)[:300]}
        r = raw(state)
        trace.write(
            "meta",
            context=ctx,
            camp=r.get("camp"),
            seats=r.get("seats"),
            pacing=r.get("pacing"),
            max_steps=r.get("max_steps"),
            time_limit_s=r.get("time_limit_s"),
            general=self.general.enabled,
            model=getattr(self.general.client, "model", None),
            baseline_camp=self._baseline_camp or None,
            code=_code_hash(),
            pid=os.getpid(),
        )

    def _tick(self, state: GameState, step: int) -> None:
        """A compact record of every step (5 per second): where everyone is, what they have, what stands."""
        try:
            v = team_view(state)
            trace.write(
                "tick",
                s=step,
                left=v["seconds_left"],
                h={
                    str(k): [
                        h.get("hp"),
                        h.get("x"),
                        h.get("z"),
                        h.get("level"),
                        h.get("kills"),
                        h.get("deaths"),
                        h.get("revive_time"),
                    ]
                    for k, h in v["ours"].items()
                },
                e={
                    k: [e.get("hp"), e.get("x"), e.get("z")]
                    for k, e in v["enemies"].items()
                    if e.get("visible")
                },
                st=[[s.get("camp"), s.get("config_id"), s.get("hp")] for s in v["structures"]],
                stand=standing_orders(state),
            )
        except Exception:  # noqa: BLE001 - recording must never cost a match
            pass

    def _record(self, state: GameState) -> None:
        try:
            now = time.monotonic()
            r = raw(state)
            if now - self._last_snapshot_at >= SNAPSHOT_EVERY_S:
                self._last_snapshot_at = now
                trace.write(
                    "snapshot", raw=r, legal=[a.action_id for a in state.legal_actions or []]
                )
            if now - self._last_state_at < STATE_EVERY_S:
                return
            self._last_state_at = now
            view = team_view(state)
            trace.write(
                "state",
                step=view["step"],
                camp=view["camp"],
                seconds_left=view["seconds_left"],
                phase=r.get("phase"),
                pacing=r.get("pacing"),
                standing=r.get("simultaneous"),
                heroes=view["ours"],
                enemies=view["enemies"],
                structures=view["structures"],
                readings=[x.as_json() for x in (*self._readings.values(), *self._ahead.values())],
                reach=round(self.memory.R),
                reach_stats=self.memory.reach_stats(),
                fights=[f.as_json(self.memory.step) for f in self.memory.fights[-3:]],
                n_fights=len(self.memory.fights),
                plan=self.general.current().as_json(),
                plan_v=self.general.version,
                dealt=round(self.memory.dealt),
                taken=round(self.memory.taken),
                calibration=round(self.memory.calibration, 3),
            )
        except Exception:  # noqa: BLE001 - recording must never cost a match
            pass


_CODE_HASH: str | None = None


def _code_hash() -> str:
    """A short fingerprint of the code that played, so traces from different versions are never confused."""
    global _CODE_HASH
    if _CODE_HASH is None:
        h = hashlib.sha1()
        try:
            root = os.path.dirname(os.path.abspath(__file__))
            for name in sorted(os.listdir(root)):
                if name.endswith(".py"):
                    h.update(open(os.path.join(root, name), "rb").read())
        except OSError:
            pass
        _CODE_HASH = h.hexdigest()[:10]
    return _CODE_HASH


def create_agent():
    on = os.environ.get("HOK_LLM", "0").strip().lower() in ("1", "true", "yes", "on")
    model = GeneralModel.from_gemini(GeminiClient.from_env()) if on else None
    if on and model is None:
        print(
            "[hok] HOK_LLM=1 but no GEMINI_API_KEY: playing without the General",
            file=sys.stderr,
            flush=True,
        )
    return HokAgent(model, JevClient.from_env(flag="HOK_JEV"))
