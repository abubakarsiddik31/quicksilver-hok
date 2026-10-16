"""Jev, the fast tactician: typed, calibrated answers about the fight in progress, about once a second.

The General (Gemini) writes the plan every few seconds to minutes; the fight model scores each hero every step.
Jev sits between them. It reads the whole tactical picture (every hero, every enemy known with how stale the
sighting is, the structures, the lane geometry, the fight model's own readings, the score) and answers bounded
questions with probabilities, never free text:

* for every living hero: ``engage`` / ``hold`` / ``retreat``;
* for the team: ``pressure`` / ``balanced`` / ``defend``, the chance the enemy team is about to arrive, and the
  chance we win the engagement in front of us.

How the answers are used is deliberately one-sided. They may make a hero MORE careful (a confident ``hold`` or
``retreat`` stops a hero walking into a fight, or sends a hurt one home) but never force a hero into danger, and
only when fresh and at least as confident as the plan's ``jev_trust``. The fight model keeps the last word on
aggression. Everything Jev says is traced so its value can be measured on recorded games before it is given more.

Total: any failure, an old answer or no key means no advice and the fight model plays alone.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from hok import trace
from hok.memory import STEP_S, MatchMemory, pos

ADVICE_TTL_S = 3.0
MIN_GAP_S = 0.8
MAX_CALLS = 900
BACKOFF_FIRST_S, BACKOFF_MAX_S = 3.0, 30.0

RULES = {
    "win": "Destroy the enemy crystal. At the time or step limit the team that dealt more damage to enemy towers "
    "and crystal wins, then the team with more hero kills.",
    "map": "A single lane. `f` is the position along it: 0 is our base, 1 is theirs, towers stand between. A "
    "standing tower fights for its side, so a fight near our own tower is worth more to us.",
    "heroes": "Heroes heal at their own base. A dead hero is out for a revive time (seconds) that grows with "
    "level, during which the other side can attack structures freely. Running home takes a long time "
    "(flee_s); a hero that cannot outlast the walk (survive_s below flee_s) while enemies are in reach is "
    "better off fighting.",
    "enemies": "Enemy heroes out of sight are not gone: `seen_s_ago` says how stale a sighting is. Enemies never "
    "seen are expected to be doing what we do. A dead enemy is back in `dead_back_in_s` seconds.",
    "strategy": "Towers fall only when their defenders are away or dead. Answer every hit on our structures; fight "
    "beside our front tower; when three or more enemies are dead and will not be back before we arrive, push their "
    "front tower (about 9 s from ours, 22 s from our base; five heroes take a 4000-hp tower in about 6 s); never "
    "leave our crystal or a tower under attack to go elsewhere.",
}

STANCES = {
    "engage": "Fight what is in reach, or walk on toward the enemy: the local force, the hero's health and the "
    "expected enemy reinforcements make this a fight we should take.",
    "hold": "Do not advance into the unknown or into a bad fight: wait near our own front tower for our allies or "
    "for the enemy to come to us.",
    "retreat": "Leave: this hero is too hurt, or the enemy force it faces is too strong, and it can still get away.",
}
POSTURES = {
    "pressure": "We are ahead or the enemy is weak: push for tower damage.",
    "balanced": "Take only the fights that are clearly favourable.",
    "defend": "Hold near our own tower, let the enemy come, and punish it there.",
}


@dataclass
class Advice:
    at: float = 0.0
    step: int = 0
    hero: dict[int, tuple[str, float]] = field(
        default_factory=dict
    )  # seat -> (stance, probability)
    posture: tuple[str, float] | None = None
    enemy_close: float | None = None
    win_fight: float | None = None
    latency_s: float = 0.0

    def as_json(self) -> dict[str, Any]:
        return {
            "hero": {str(s): [c, round(p, 2)] for s, (c, p) in sorted(self.hero.items())},
            "posture": self.posture and [self.posture[0], round(self.posture[1], 2)],
            "enemy_team_close_p": self.enemy_close,
            "win_fight_p": self.win_fight,
            "latency_s": self.latency_s,
        }


def _f(memory: MatchMemory, p: tuple[float, float] | None) -> float | None:
    v = memory.progress(p)
    return None if v is None else round(v, 2)


def build_request(
    memory: MatchMemory, readings: dict[int, Any], standing: dict[int, str], plan: Any
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """The state Jev reads and the questions it answers, from engine data only."""
    now = memory.step
    heroes, living = [], []
    for seat, h in sorted(memory.ours.items()):
        alive = (h.get("hp") or 0) > 0
        r = readings.get(seat)
        mx = h.get("max_hp") or 0
        heroes.append(
            {
                "seat": seat,
                "level": h.get("level"),
                "alive": alive,
                "hp_pct": round(100 * (h.get("hp") or 0) / mx) if mx else None,
                "f": _f(memory, pos(h)),
                "revive_s": h.get("revive_time"),
                "order": standing.get(seat),
                "fight": None
                if r is None
                else {
                    "power_ratio_here": round(r.ratio, 2),
                    "allies_near": r.allies,
                    "enemies_in_reach": r.in_reach,
                    "survive_s": None if r.survive_s == float("inf") else round(r.survive_s, 1),
                    "flee_s": round(r.flee_s, 1),
                },
            }
        )
        if alive:
            living.append(seat)
    enemies = [
        {
            "hero": e.hero_id,
            "visible_now": (now - e.step) * STEP_S <= 0.3,
            "hp_pct": round(100 * e.hp / e.max_hp) if e.max_hp else None,
            "level": e.level,
            "f": _f(memory, e.at()),
            "seen_s_ago": round((now - e.step) * STEP_S),
            "dead_back_in_s": None if memory.enemy_back_in(e) is None else round(memory.enemy_back_in(e)),
        }
        for e in sorted(memory.enemies.values(), key=lambda e: e.rid)
    ]
    structures = [
        {
            "side": "ours" if s["camp"] == memory.camp else "theirs",
            "type": str(s["type"]).replace("ACTOR_", "").lower(),
            "hp_pct": round(100 * s["hp"] / s["max_hp"]) if s["max_hp"] else None,
            "f": _f(memory, pos(s)),
        }
        for s in memory.structs.values()
        if s["type"] != "ACTOR_NONE"
    ]
    state = {
        "rules": RULES,
        "clock_s": round(now * STEP_S),
        "seconds_left": memory.seconds_left,
        "score": memory.projection(),
        "our_plan": {"posture": plan.posture, "risk": plan.risk},
        "enemies_never_seen": max(0, 5 - len(memory.enemies)),
        "our_structures_under_attack": [
            str(st["type"]).replace("ACTOR_", "").lower()
            for st in memory.structs.values()
            if st["camp"] == memory.camp and st["hp"] > 0 and st.get("hit_step") is not None
            and (now - st["hit_step"]) * STEP_S <= 8
        ],
        "heroes": heroes,
        "enemies": enemies,
        "structures": structures,
    }
    questions: dict[str, dict[str, Any]] = {}
    for seat in living:
        questions[f"hero_{seat}"] = {
            "type": "choice",
            "instructions": f"What should hero {seat} do right now? Judge it from its own `fight` numbers, its "
            "health, where its allies and the enemy team are, and the lane position `f`.",
            "criteria": dict(STANCES),
        }
    questions["team_posture"] = {
        "type": "choice",
        "instructions": "Which posture suits the whole team right now, given the score, the clock and the "
        "enemy's last known positions?",
        "criteria": dict(POSTURES),
    }
    questions["enemy_team_close"] = {
        "type": "noul",
        "instructions": "Is most of the enemy team within about ten seconds of reaching our front-most hero?",
        "criteria": {
            "true": "Most of the enemy team is close, seen or expected by its last sighting and the "
            "mirror of our own advance",
            "false": "The enemy is far, dead, or split up",
        },
    }
    questions["win_fight"] = {
        "type": "noul",
        "instructions": "If our team fought the enemy heroes that are close to it now, would we win the exchange?",
        "criteria": {
            "true": "We would kill more than we lose",
            "false": "We would lose more than we kill",
        },
    }
    return state, dict(list(questions.items())[:10])


class Tactician:
    def __init__(
        self,
        client: Any | None,
        *,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.client = client
        self._clock = clock
        self._log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self._lock = threading.Lock()
        self._advice: Advice | None = None
        self._busy = False
        self._last_start = -1e9
        self._calls = 0
        self._errors = 0
        self._backoff_until = 0.0
        self.disabled: str | None = None

    @property
    def enabled(self) -> bool:
        return self.client is not None and not self.disabled

    def maybe_update(
        self,
        memory: MatchMemory,
        readings: dict[int, Any],
        standing: dict[int, str],
        plan: Any,
        *,
        sync: bool = False,
    ) -> None:
        """Start a background request when one is due. Never blocks the caller."""
        if not self.enabled:
            return
        now = self._clock()
        with self._lock:
            if (
                self._busy
                or self._calls >= MAX_CALLS
                or now < self._backoff_until
                or now - self._last_start < MIN_GAP_S
            ):
                return
            self._busy = True
            self._last_start = now
            self._calls += 1
        try:
            state, questions = build_request(memory, readings, standing, plan)
        except Exception as exc:  # noqa: BLE001 - never break the loop
            with self._lock:
                self._busy = False
            self._log(f"[jev] could not build the request: {exc!r}")
            return
        step = memory.step
        if sync:
            self._run(state, questions, step)
        else:
            threading.Thread(target=self._run, args=(state, questions, step), daemon=True).start()

    def _run(self, state: dict[str, Any], questions: dict[str, dict[str, Any]], step: int) -> None:
        started = self._clock()
        answers = None
        try:
            answers = self.client.ask(state, questions)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            answers = None
        done = self._clock()
        advice = self._read(answers, step, done, round(done - started, 2)) if answers else None
        with self._lock:
            if advice is not None:
                self._advice, self._errors = advice, 0
            else:
                self._errors += 1
                self._backoff_until = done + min(
                    BACKOFF_MAX_S, BACKOFF_FIRST_S * 2 ** min(self._errors - 1, 4)
                )
            self._busy = False
        if advice is not None:
            trace.write("jev", step=step, advice=advice.as_json())
        else:
            err = getattr(self.client, "last_error", None)
            trace.write("jev", step=step, advice=None, error=err)
            if self._errors in (1, 5):
                self._log(f"[jev] no answer ({err}); the fight model plays alone meanwhile")

    @staticmethod
    def _read(answers: dict[str, Any], step: int, at: float, latency: float) -> Advice | None:
        adv = Advice(at=at, step=step, latency_s=latency)
        for name, a in answers.items():
            if (
                name.startswith("hero_")
                and name[5:].isdigit()
                and getattr(a, "choice", None) in STANCES
            ):
                p = (a.probabilities or {}).get(
                    a.choice, a.confidence if a.confidence is not None else 0.0
                )
                adv.hero[int(name[5:])] = (a.choice, float(p))
            elif name == "team_posture" and getattr(a, "choice", None) in POSTURES:
                p = (a.probabilities or {}).get(
                    a.choice, a.confidence if a.confidence is not None else 0.0
                )
                adv.posture = (a.choice, float(p))
            elif name == "enemy_team_close" and a.noul is not None:
                adv.enemy_close = round(float(a.noul), 2)
            elif name == "win_fight" and a.noul is not None:
                adv.win_fight = round(float(a.noul), 2)
        return adv if (adv.hero or adv.posture) else None

    # -- read side ---------------------------------------------------------------------

    def current(self) -> Advice | None:
        with self._lock:
            a = self._advice
        return a if a is not None and self._clock() - a.at <= ADVICE_TTL_S else None

    def caution(self, seat: int, trust: float) -> str | None:
        """``hold`` or ``retreat`` when Jev is fresh and at least ``trust`` sure the hero should not engage."""
        a = self.current()
        if a is None:
            return None
        stance = a.hero.get(seat)
        if stance and stance[0] in ("hold", "retreat") and stance[1] >= trust:
            return stance[0]
        return None
