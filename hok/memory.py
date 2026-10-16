"""Match memory: the whole game, kept.

The server shows a hero only what is in its sight this step. A commander remembers: where each enemy hero was
last seen and how healthy, how the towers have been trading, every fight so far and how it really went against
what was expected, and what each earlier plan achieved. This is that memory. It is updated every decision from
engine data only (ids, positions, hp, counters), never reset, and read by:

* the General (the strategist model), through :meth:`MatchMemory.briefing`;
* the fight model (:mod:`hok.battle`), through the sightings, the observed hero speed and the fight ledger's
  realised exchange against its predictions.

Game time is counted in steps (0.2 s each). Structure damage is measured from structure hp drops, because the
tie-break at a limit is damage to towers and the crystal first, then hero kills.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

STEP_S = 0.2
FIGHT_R = 9000.0  # game units: prior for the engagement radius until enough reach has been seen
DAMAGE_MIN_OBS = 30
FIGHT_INTERVAL_PRIOR_S = 60.0  # seconds between fights before the match shows its own rhythm
SPEED_PRIOR = 3600.0  # units per second: the median hero walking speed over our first recorded matches  # damage distances needed before the learned radius replaces the prior
FIGHT_QUIET_S = 10.0  # no contact for this long ends a fight
LEVEL_POWER = 0.12  # prior: power grows this much per level; calibrated online through the fight ledger
CALIBRATION_PRIOR = 2.0  # pseudo-fights that went exactly as predicted
VANISH_UPDATES = 10  # a structure missing this many updates in a row (2 s) is gone, not a flicker
CALIBRATION_BOUNDS = (0.25, 4.0)
PRIOR_HP_PER_ENEMY_S = 100.0  # hp lost per second per enemy in reach (first traces: mean ~106, bursts 450+)
PRIOR_EXPOSURE_S = 20.0  # weight of that prior, in seconds of contact
MAX_FIGHTS = 40
MAX_EVENTS = 40
MAX_PLANS = 12
SNAPSHOT_EVERY_S = 20.0


def power(hp: float | None, level: float | None) -> float:
    """Fighting weight of a hero: what it has left, scaled up by level (a prior the ledger corrects)."""
    if not isinstance(hp, (int, float)) or hp <= 0:
        return 0.0
    lvl = level if isinstance(level, (int, float)) and level > 0 else 1.0
    return float(hp) * (1.0 + LEVEL_POWER * (lvl - 1.0))


def dist(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    if a is None or b is None:
        return math.inf
    return math.hypot(a[0] - b[0], a[1] - b[1])


def pos(h: dict[str, Any] | None) -> tuple[float, float] | None:
    if not isinstance(h, dict):
        return None
    x, z = h.get("x"), h.get("z")
    if isinstance(x, (int, float)) and isinstance(z, (int, float)):
        return float(x), float(z)
    return None


@dataclass
class Sighting:
    rid: str
    hero_id: Any
    x: float | None
    z: float | None
    hp: float
    max_hp: float
    level: float
    step: int  # when it was last seen
    revive: float = 0.0  # its revive time when last seen dead, seconds

    def at(self) -> tuple[float, float] | None:
        return (self.x, self.z) if self.x is not None and self.z is not None else None


@dataclass
class Fight:
    start_step: int
    predicted: float | None = None  # exchange ratio the fight model expected (their loss per our loss)
    our_power0: float = 0.0
    their_power0: float = 0.0
    our_lost: float = 0.0
    their_lost: float = 0.0
    our_deaths: int = 0
    kills: int = 0
    last_contact: int = 0
    end_step: int | None = None
    plan: int = 0
    where: str = "middle"  # our_side / middle / their_side of the lane
    jev_p: float | None = None  # Jev's chance that we win it, asked when it started

    @property
    def realised(self) -> float:
        return (self.their_lost + 1.0) / (self.our_lost + 1.0)

    def as_json(self, now: int) -> dict[str, Any]:
        return {
            "start_s": round(self.start_step * STEP_S),
            "len_s": round(((self.end_step or now) - self.start_step) * STEP_S),
            "open": self.end_step is None,
            "expected_ratio": None if self.predicted is None else round(self.predicted, 2),
            "realised_ratio": round(self.realised, 2),
            "we_lost_hp": round(self.our_lost),
            "they_lost_hp": round(self.their_lost),
            "our_deaths": self.our_deaths,
            "where": self.where,
            "jev_win_p": self.jev_p,
            "our_kills": self.kills,
        }


class MatchMemory:
    def __init__(self) -> None:
        self.step = 0
        self.camp: Any = None
        self.ours: dict[int, dict[str, Any]] = {}
        self.enemies: dict[str, Sighting] = {}
        self.structs: dict[tuple, dict[str, Any]] = {}
        self.dealt = 0.0  # hp of enemy structures destroyed so far (our tie-break score)
        self.taken = 0.0  # hp of our structures destroyed (theirs)
        self.team_kills = 0
        self.team_deaths = 0
        self.fights: list[Fight] = []
        self.events: deque[tuple[int, str]] = deque(maxlen=MAX_EVENTS)
        self.plans: list[dict[str, Any]] = []
        self._speeds: deque[float] = deque(maxlen=60)
        self._calib: list[float] = []  # log(realised / expected) of finished fights
        self._calib_by: dict[str, list[float]] = {}  # the same, by where the fight was
        self.jev_scores: list[tuple[float, float]] = []  # (Jev's win chance, 1 if we won the exchange)
        self._prev_ours: dict[int, dict[str, Any]] = {}
        self._prev_enemy_hp: dict[str, float] = {}
        self._extent = FIGHT_R * 10.0
        self._reach: deque[float] = deque(maxlen=300)
        self._revive: dict[int, int] = {}  # level -> longest revive time seen, seconds
        self._lost_hp = 0.0  # hp our heroes lost while in reach of enemies ...
        self._exposure = 0.0  # ... over this many hero-seconds x enemies in reach
        self._damage: deque[float] = deque(maxlen=400)  # distances at which damage was actually dealt or taken
        self._reach_by: dict[str, deque[float]] = {}
        self.seconds_left: float | None = None
        self.max_steps: float | None = None

    # -- update ------------------------------------------------------------------------

    def update(self, view: dict[str, Any]) -> list[str]:
        """Fold one team view in. Returns this step's events (for the strategist's triggers)."""
        new: list[str] = []
        step = int(view.get("step") or 0)
        dt = max(0, step - self.step) * STEP_S if self.step else 0.0
        self.step = max(self.step, step)
        self.camp = view.get("camp") if view.get("camp") is not None else self.camp
        self.seconds_left = view.get("seconds_left")
        self.max_steps = view.get("max_steps") or self.max_steps
        ours: dict[int, dict[str, Any]] = view.get("ours") or {}
        kills = sum(int(h.get("kills") or 0) for h in ours.values())
        deaths = sum(int(h.get("deaths") or 0) for h in ours.values())
        our_lost = their_lost = 0.0
        visible_now = [e for e in (view.get("enemies") or {}).values() if e.get("visible")]
        for seat, h in ours.items():
            rt, lv = h.get("revive_time"), h.get("level")
            if isinstance(rt, (int, float)) and rt > 0 and isinstance(lv, (int, float)):
                self._revive[int(lv)] = max(self._revive.get(int(lv), 0), int(rt))
            prev = self._prev_ours.get(seat)
            if prev is not None:
                if (h.get("deaths") or 0) > (prev.get("deaths") or 0):
                    new.append(f"our_hero_died:{seat}")
                dhp = (prev.get("hp") or 0) - (h.get("hp") or 0)
                if dhp > 0 and (h.get("deaths") or 0) == (prev.get("deaths") or 0):
                    our_lost += dhp * (1.0 + LEVEL_POWER * ((h.get("level") or 1) - 1))
                    near = min((dist(pos(h), pos(e)) for e in visible_now), default=math.inf)
                    if near < math.inf:
                        self._damage.append(near)  # the nearest visible enemy is the likely source
                n_reach = sum(1 for e in visible_now if dist(pos(h), pos(e)) <= self.R)
                if n_reach and (h.get("hp") or 0) > 0 and dt > 0:
                    self._exposure += dt * n_reach
                    self._lost_hp += max(0.0, dhp)
                p0, p1 = pos(prev), pos(h)
                if dt > 0 and p0 and p1 and (h.get("hp") or 0) > 0 and (prev.get("hp") or 0) > 0:
                    sp = dist(p0, p1) / dt
                    if sp > 0:
                        self._speeds.append(sp)
        if kills > self.team_kills:
            new.append(f"our_kill:{kills - self.team_kills}")
        self.team_kills, self.team_deaths = kills, deaths
        for rid, e in (view.get("enemies") or {}).items():
            if not e.get("visible"):
                continue
            hp = float(e.get("hp") or 0)
            lvl = float(e.get("level") or 1)
            before = self._prev_enemy_hp.get(rid)
            if before is not None and hp < before:
                their_lost += (before - hp) * (1.0 + LEVEL_POWER * (lvl - 1))
                near = min((dist(pos(h), pos(e)) for h in ours.values() if (h.get("hp") or 0) > 0), default=math.inf)
                if near < math.inf:
                    self._damage.append(near)
            self._prev_enemy_hp[rid] = hp
            if rid not in self.enemies:
                new.append(f"enemy_first_seen:{e.get('hero_id')}")
            self.enemies[rid] = Sighting(
                rid, e.get("hero_id"), e.get("x"), e.get("z"), hp,
                float(e.get("max_hp") or hp or 1), lvl, step,
                revive=float(e.get("revive_time") or 0),
            )
        new += self._structures(view.get("structures") or [])
        self.ours = ours
        self._prev_ours = {s: dict(h) for s, h in ours.items()}
        new += self._fight(step, ours, view.get("enemies") or {}, our_lost, their_lost, kills, deaths)
        for ev in new:
            self.events.append((step, ev))
        return new

    def _structures(self, structures: list[dict[str, Any]]) -> list[str]:
        new: list[str] = []
        xs, zs = [], []
        for s in structures:
            key = (s.get("camp"), s.get("type"), round(float(s.get("x") or 0)), round(float(s.get("z") or 0)))
            hp = float(s.get("hp") or 0)
            xs.append(float(s.get("x") or 0))
            zs.append(float(s.get("z") or 0))
            prev = self.structs.get(key)
            if prev is not None and hp < prev["hp"]:
                lost = prev["hp"] - hp
                prev["hit_step"] = self.step
                if s.get("camp") == self.camp:
                    self.taken += lost
                    new.append(f"our_structure_hit:{s.get('type')}")
                else:
                    self.dealt += lost
                if hp <= 0 < prev["hp"]:
                    new.append(
                        f"{'our' if s.get('camp') == self.camp else 'enemy'}_structure_destroyed:{s.get('type')}"
                    )
            self.structs[key] = {"hp": hp, "max_hp": float(s.get("max_hp") or hp or 1), "type": s.get("type"),
                                 "camp": s.get("camp"), "x": s.get("x"), "z": s.get("z"),
                                 "hit_step": (prev or {}).get("hit_step"), "absent": (prev or {}).get("absent", 0)}
        seen = {(x.get("camp"), x.get("type"), round(float(x.get("x") or 0)), round(float(x.get("z") or 0)))
                for x in structures}
        if structures:  # a structure that drops off the list was destroyed: credit what it had left
            for key, prev in self.structs.items():
                if key in seen:
                    prev["absent"] = 0
                    continue
                prev["absent"] = prev.get("absent", 0) + 1
                if prev["hp"] > 0 and prev["absent"] >= VANISH_UPDATES:
                    if prev["camp"] == self.camp:
                        self.taken += prev["hp"]
                    else:
                        self.dealt += prev["hp"]
                    prev["hp"] = 0.0
                    new.append(f"{'our' if prev['camp'] == self.camp else 'enemy'}_structure_destroyed:{prev['type']}")
        if xs and zs:
            self._extent = max(self.R, math.hypot(max(xs) - min(xs), max(zs) - min(zs)))
        return new

    def _fight(self, step, ours, enemies, our_lost, their_lost, kills, deaths) -> list[str]:
        out: list[str] = []
        contact = False
        for h in ours.values():
            if (h.get("hp") or 0) <= 0:
                continue
            for e in enemies.values():
                if e.get("visible") and dist(pos(h), pos(e)) <= self.R:
                    contact = True
        open_fight = self.fights[-1] if self.fights and self.fights[-1].end_step is None else None
        if contact and open_fight is None:
            a = sum(power(h.get("hp"), h.get("level")) for h in ours.values())
            b = sum(power(e.get("hp"), e.get("level")) for e in enemies.values() if e.get("visible"))
            near = [pos(h) for h in ours.values() if (h.get("hp") or 0) > 0 and pos(h) and any(
                e.get("visible") and dist(pos(h), pos(e)) <= self.R for e in enemies.values())]
            centre = (sum(p[0] for p in near) / len(near), sum(p[1] for p in near) / len(near)) if near else None
            open_fight = Fight(start_step=step, our_power0=a, their_power0=b, last_contact=step,
                               plan=len(self.plans), where=self.place(centre))
            open_fight._kills0, open_fight._deaths0 = kills, deaths  # type: ignore[attr-defined]
            self.fights.append(open_fight)
            self.fights = self.fights[-MAX_FIGHTS:]
            out.append("fight_started")
        if open_fight is None:
            return out
        open_fight.our_lost += our_lost
        open_fight.their_lost += their_lost
        open_fight.kills = kills - getattr(open_fight, "_kills0", kills)
        open_fight.our_deaths = deaths - getattr(open_fight, "_deaths0", deaths)
        if contact:
            open_fight.last_contact = step
        elif (step - open_fight.last_contact) * STEP_S >= FIGHT_QUIET_S:
            open_fight.end_step = open_fight.last_contact
            if open_fight.predicted and open_fight.predicted > 0:
                err = math.log(open_fight.realised / open_fight.predicted)
                self._calib.append(err)
                self._calib_by.setdefault(open_fight.where, []).append(err)
            if open_fight.jev_p is not None:
                self.jev_scores.append((open_fight.jev_p, 1.0 if open_fight.realised > 1.0 else 0.0))
            out.append("fight_ended")
        return out

    # -- what the fight model reads -----------------------------------------------------

    @property
    def open_fight(self) -> Fight | None:
        return self.fights[-1] if self.fights and self.fights[-1].end_step is None else None

    def observe_reach(self, d: float, macro: str = "") -> None:
        """A macro was offered with a target ``d`` away, so its reach is at least that."""
        if d == d and d != math.inf and d > 0:
            self._reach.append(d)
            self._reach_by.setdefault(macro, deque(maxlen=300)).append(d)

    def reach_stats(self) -> dict[str, dict[str, float]]:
        """Per macro: how many offers, the median and the largest distance seen (to check 'offered = in reach')."""
        out = {}
        for macro, ds in self._reach_by.items():
            ordered = sorted(ds)
            out[macro] = {"n": len(ordered), "p50": round(ordered[len(ordered) // 2]), "max": round(ordered[-1])}
        return out

    @property
    def R(self) -> float:
        """The engagement radius: how far apart heroes really were when damage landed (median), else
        the prior. Macro offers say nothing about it: the server offers a macro toward the nearest target at any
        distance (the first match showed skills offered 60,000 units away)."""
        if len(self._damage) < DAMAGE_MIN_OBS:
            return FIGHT_R
        ordered = sorted(self._damage)
        return ordered[len(ordered) // 2]  # the median: the far tail is towers, minions and regeneration

    # -- lane geometry (all from the structures the server lists) --------------------------

    def _anchor(self, ours: bool) -> tuple[float, float] | None:
        """A side's base: its base turret, else its crystal."""
        crystal = None
        for s in self.structs.values():
            if (s["camp"] == self.camp) != ours or pos(s) is None:
                continue
            if s["type"] == "ACTOR_TOWER_SPRING":
                return pos(s)
            if s["type"] == "ACTOR_CRYSTAL":
                crystal = pos(s)
        return crystal

    def lane(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        own, foe = self._anchor(True), self._anchor(False)
        return (own, foe) if own and foe and dist(own, foe) > 0 else None

    def progress(self, p: tuple[float, float] | None) -> float | None:
        """How far along the lane from our base to theirs, as a fraction (0 = our base, 1 = theirs)."""
        lane = self.lane()
        if lane is None or p is None:
            return None
        (ox, oz), (fx, fz) = lane
        L2 = (fx - ox) ** 2 + (fz - oz) ** 2
        return ((p[0] - ox) * (fx - ox) + (p[1] - oz) * (fz - oz)) / L2

    def front_tower(self) -> tuple[float, float] | None:
        """Our standing tower nearest the enemy (the rally point); our base turret when none stands."""
        best, best_p = None, -1e9
        for s in self.structs.values():
            if s["camp"] != self.camp or s["type"] != "ACTOR_TOWER" or s["hp"] <= 0 or pos(s) is None:
                continue
            pr = self.progress(pos(s))
            if pr is not None and pr > best_p:
                best, best_p = pos(s), pr
        if best is not None:
            return best
        for s in self.structs.values():  # no tower left: the crystal, never the base turret behind it
            if s["camp"] == self.camp and s["type"] == "ACTOR_CRYSTAL" and s["hp"] > 0 and pos(s):
                return pos(s)
        return self._anchor(True)

    def place(self, p: tuple[float, float] | None) -> str:
        """Which part of the lane a point is in: behind the midpoint between our front tower and the centre is
        our side, beyond the midpoint between the centre and their front tower is theirs. Fights there differ:
        a tower fights for its own side."""
        f = self.progress(p)
        if f is None:
            return "middle"
        ours = [self.progress(pos(s)) for s in self.structs.values() if s["camp"] == self.camp and s["type"] == "ACTOR_TOWER"]
        theirs = [self.progress(pos(s)) for s in self.structs.values() if s["camp"] != self.camp and s["type"] == "ACTOR_TOWER"]
        lo = (max(ours) + 0.5) / 2 if ours else 0.4
        hi = (min(theirs) + 0.5) / 2 if theirs else 0.6
        return "our_side" if f <= lo else "their_side" if f >= hi else "middle"

    def calibration_at(self, p: tuple[float, float] | None) -> float:
        """The fight-model correction for fights at this part of the lane: that place's own record, pulled toward
        the record of all fights (so a place with no fights yet uses the overall one)."""
        xs = self._calib_by.get(self.place(p), [])
        overall = sum(self._calib) / (len(self._calib) + CALIBRATION_PRIOR)
        est = (sum(xs) + CALIBRATION_PRIOR * overall) / (len(xs) + CALIBRATION_PRIOR)
        lo, hi = CALIBRATION_BOUNDS
        return min(hi, max(lo, math.exp(est)))

    def fight_interval(self) -> float:
        """Seconds between fights in this match so far (prior FIGHT_INTERVAL_PRIOR_S): how long a hero that goes
        home to heal is likely to miss."""
        starts = [f.start_step for f in self.fights]
        gaps = [(b - a) * STEP_S for a, b in zip(starts, starts[1:])]
        return (sum(gaps) + 2 * FIGHT_INTERVAL_PRIOR_S) / (len(gaps) + 2)

    def jev_record(self) -> dict[str, Any] | None:
        """How Jev's fight calls have held up: Brier score (0 perfect, 0.25 a coin) over finished fights."""
        if not self.jev_scores:
            return None
        n = len(self.jev_scores)
        brier = sum((p - o) ** 2 for p, o in self.jev_scores) / n
        return {"fights": n, "brier": round(brier, 3), "coin_flip_brier": 0.25,
                "we_won": sum(o for _, o in self.jev_scores), "mean_predicted": round(sum(p for p, _ in self.jev_scores) / n, 2)}

    def revive_estimate(self, level: float | None) -> float:
        """How long a dead hero of this level stays dead: what we have seen for that level, else the nearest."""
        if not self._revive:
            return 20.0
        lv = int(level or 1)
        if lv in self._revive:
            return float(self._revive[lv])
        nearest = min(self._revive, key=lambda k: abs(k - lv))
        return float(self._revive[nearest])

    def enemy_back_in(self, e: Sighting) -> float | None:
        """Seconds until this enemy is expected back, if it was last seen dead (0 once the time has passed)."""
        if e.hp > 0:
            return None
        back = e.revive if e.revive > 0 else self.revive_estimate(e.level)
        return max(0.0, back - (self.step - e.step) * STEP_S)

    def defend_point(self, window_s: float = 8.0) -> tuple[float, float] | None:
        """Where one of our structures is being hit right now (the most recently hit one that still stands)."""
        best, best_step = None, -1
        for s in self.structs.values():
            hit = s.get("hit_step")
            if s["camp"] != self.camp or s["hp"] <= 0 or hit is None or pos(s) is None:
                continue
            if (self.step - hit) * STEP_S <= window_s and hit > best_step:
                best, best_step = pos(s), hit
        return best

    def mirror_of_team(self) -> tuple[float, float] | None:
        """Where an unseen enemy team is expected: our living heroes' centroid reflected through the middle of
        the lane. Both sides start symmetric and want the same thing, so until it is seen the enemy is most
        likely doing what we do."""
        lane = self.lane()
        pts = [pos(h) for h in self.ours.values() if (h.get("hp") or 0) > 0 and pos(h)]
        if lane is None or not pts:
            return None
        cx = sum(p[0] for p in pts) / len(pts)
        cz = sum(p[1] for p in pts) / len(pts)
        mx, mz = (lane[0][0] + lane[1][0]) / 2, (lane[0][1] + lane[1][1]) / 2
        return 2 * mx - cx, 2 * mz - cz

    @property
    def incoming(self) -> float:
        """hp a hero loses per second for each enemy in reach, learned from this match on top of a prior."""
        return (self._lost_hp + PRIOR_HP_PER_ENEMY_S * PRIOR_EXPOSURE_S) / (self._exposure + PRIOR_EXPOSURE_S)

    def home(self) -> tuple[float, float] | None:
        """Where retreat leads: our own base turret, else our crystal."""
        best = None
        for s in self.structs.values():
            if s["camp"] != self.camp or pos(s) is None:
                continue
            if s["type"] == "ACTOR_TOWER_SPRING":
                return pos(s)
            if s["type"] == "ACTOR_CRYSTAL":
                best = pos(s)
        return best

    @property
    def speed(self) -> float:
        """Observed hero walking speed, game units per second (median of recent moves)."""
        if len(self._speeds) < 5:
            return SPEED_PRIOR
        ordered = sorted(self._speeds)
        return ordered[len(ordered) // 2]

    @property
    def extent(self) -> float:
        return self._extent

    @property
    def calibration(self) -> float:
        """Multiplier on a predicted exchange ratio: how past fights compare with what was predicted."""
        total = sum(self._calib)
        n = len(self._calib) + CALIBRATION_PRIOR
        lo, hi = CALIBRATION_BOUNDS
        return min(hi, max(lo, math.exp(total / n)))

    # -- the General's view ---------------------------------------------------------------

    def note_plan(self, version: int, assessment: str, posture: str, risk: float) -> None:
        self.plans.append({"v": version, "at_s": round(self.step * STEP_S), "posture": posture,
                           "risk": risk, "assessment": assessment,
                           "dealt": round(self.dealt), "taken": round(self.taken),
                           "kills": self.team_kills, "deaths": self.team_deaths})
        self.plans = self.plans[-MAX_PLANS:]

    def _lane_json(self) -> dict[str, Any] | None:
        lane = self.lane()
        if lane is None:
            return None
        ft = self.front_tower()
        enemy_towers = [self.progress(pos(s)) for s in self.structs.values()
                        if s["camp"] != self.camp and s["type"] == "ACTOR_TOWER" and s["hp"] > 0]
        return {"length_units": round(dist(*lane)),
                "note": "0 = our base, 1 = their base; fractions are positions along the single lane",
                "our_rally_tower": None if ft is None else round(self.progress(ft) or 0, 2),
                "enemy_front_tower": round(min(enemy_towers), 2) if enemy_towers else None}

    def projection(self) -> dict[str, Any]:
        """Who wins if the match were decided now, by the tie-break rules: structure damage first, then kills."""
        structure = round(self.dealt - self.taken)
        kills = self.team_kills - self.team_deaths  # their kills are our deaths
        if structure != 0:
            result, by = ("ahead" if structure > 0 else "behind"), "structure damage"
        elif kills != 0:
            result, by = ("ahead" if kills > 0 else "behind"), "hero kills"
        else:
            result, by = "level", "nothing (a draw)"
        left = None
        if isinstance(self.max_steps, (int, float)):
            left = round(max(0, self.max_steps - self.step) * STEP_S)
        return {"if_decided_now": result, "decided_by": by, "structure_margin": structure, "kill_margin": kills,
                "game_seconds_left_by_step_limit": left, "seconds_left_by_clock": self.seconds_left}

    def _plan_outcomes(self) -> list[dict[str, Any]]:
        """Each recent plan with what happened while it was in force (changes up to the next plan, or now)."""
        out = []
        recent = self.plans[-4:]
        for i, p in enumerate(recent):
            nxt = recent[i + 1] if i + 1 < len(recent) else None
            end = nxt or {"at_s": round(self.step * STEP_S), "dealt": round(self.dealt), "taken": round(self.taken),
                          "kills": self.team_kills, "deaths": self.team_deaths}
            out.append({"v": p["v"], "posture": p["posture"], "risk": p["risk"], "assessment": p["assessment"],
                        "in_force_s": end["at_s"] - p["at_s"],
                        "structure_hp_dealt": end["dealt"] - p["dealt"], "structure_hp_taken": end["taken"] - p["taken"],
                        "kills": end["kills"] - p["kills"], "deaths": end["deaths"] - p["deaths"]})
        return out

    def briefing(self) -> dict[str, Any]:
        now = self.step
        return {
            "clock_s": round(now * STEP_S),
            "seconds_left": self.seconds_left,
            "projection": self.projection(),
            "lane": self._lane_json(),
            "measured": {"hp_lost_per_second_per_enemy_in_reach": round(self.incoming),
                         "hero_speed_units_per_s": round(self.speed), "engagement_radius": round(self.R),
                         "revive_s_by_level": dict(sorted(self._revive.items()))},
            "score": {"structure_hp_dealt": round(self.dealt), "structure_hp_taken": round(self.taken),
                      "our_kills": self.team_kills, "our_deaths": self.team_deaths},
            "our_heroes": {str(s): {k: h.get(k) for k in ("hero_id", "hp", "max_hp", "level", "x", "z",
                                                            "kills", "deaths", "money", "revive_time")}
                           for s, h in sorted(self.ours.items())},
            "enemy_heroes_known": [
                {"hero_id": e.hero_id, "hp": round(e.hp), "max_hp": round(e.max_hp), "level": e.level,
                 "f": None if self.progress(e.at()) is None else round(self.progress(e.at()), 2),
                 "last_seen_s_ago": round((now - e.step) * STEP_S),
                 "dead_back_in_s": None if self.enemy_back_in(e) is None else round(self.enemy_back_in(e))}
                for e in sorted(self.enemies.values(), key=lambda e: e.rid)
            ],
            "enemy_heroes_never_seen": max(0, 5 - len(self.enemies)),
            "structures": [
                {"side": "OURS" if s["camp"] == self.camp else "THEIRS",
                 "type": str(s["type"]).replace("ACTOR_", "").lower(), "hp": s["hp"], "max_hp": s["max_hp"],
                 "f": None if self.progress(pos(s)) is None else round(self.progress(pos(s)), 2),
                 "hit_s_ago": None if s.get("hit_step") is None else round((self.step - s["hit_step"]) * STEP_S)}
                for s in self.structs.values() if s["type"] != "ACTOR_NONE"
            ],
            "our_structures_under_attack": [
                {"type": str(s["type"]).replace("ACTOR_", "").lower(), "hp": s["hp"], "hit_s_ago": round((self.step - s["hit_step"]) * STEP_S)}
                for s in self.structs.values()
                if s["camp"] == self.camp and s["hp"] > 0 and s.get("hit_step") is not None
                and (self.step - s["hit_step"]) * STEP_S <= 12
            ],
            "recent_fights": [f.as_json(now) for f in self.fights[-5:]],
            "fight_model_calibration": round(self.calibration, 2),
            "fight_model_calibration_by_place": {k: round(math.exp(sum(v) / (len(v) + CALIBRATION_PRIOR)), 2)
                                                 for k, v in self._calib_by.items()},
            "jev_track_record": self.jev_record(),
            "seconds_between_fights": round(self.fight_interval()),
            "recent_events": [f"{round(s * STEP_S)}s {e}" for s, e in list(self.events)[-12:]],
            "earlier_plans": self._plan_outcomes(),
        }
