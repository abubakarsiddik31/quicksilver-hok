"""The fight model: is this engagement worth taking, for this hero, now?

A player does not retreat "below 25% HP". They ask how much force is here on each side, how much of the enemy
team is probably about to arrive from out of sight, what the towers add, and how fragile this particular hero is.

* **Local forces.** For one hero, its side is every living ally within one engagement radius plus our own
  structures there; the enemy side is the visible enemy heroes in that radius, the enemy structures there, and
  the *believed* share of the heroes out of sight (below).
* **Engagement radius** is learned: the memory records how far away the server offers each macro (a macro is
  only offered when its target is in reach), and the radius is the reach it has actually seen.
* **Belief, not sight.** An enemy last seen at distance d, t seconds ago, can be anywhere within
  ``d + speed * t`` of where it was. If it is uniformly somewhere in that disc, the chance it is inside this
  engagement radius is the ratio of the areas. Heroes never seen at all are spread over the whole map the same
  way. The enemy side therefore grows as sightings go stale instead of vanishing.
* **Minimum credible threat.** The enemy team has heroes, so the opposition is never treated as nothing: it is
  at least one average enemy hero.
* **Advance judgement.** A hero with nothing in reach is about to *walk into* something. It is judged against
  the force expected at its destination (the nearest enemy structure) with only the allies who will really be
  there, so a lone hero waits for the others and a group goes together, with no "advance with N" rule.
* **Fragility.** A hero fights on while ``ratio * hp_share >= risk``: a healthy hero needs parity, a hurt one a
  larger edge, smoothly. ``risk`` is the General's tolerance (above 1 cautious, below 1 bold).
* **Feedback.** The match memory compares each finished fight with the ratio expected at its start, and the
  correction (:attr:`MatchMemory.calibration`) scales every later prediction.

Power uses hp and level only, and a standing structure counts as ``STRUCT_HEROES`` average heroes: priors the
ledger corrects, not a table of stats we do not have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from hok.memory import MatchMemory, dist, pos, power

BIG = 99.0  # a ratio with nobody opposing
STRUCT_HEROES = 1.0  # prior: a standing structure fights like this many average heroes


@dataclass
class Reading:
    seat: int
    ratio: float  # our local power over theirs (calibrated)
    ours: float
    theirs: float
    allies: int
    visible_enemies: int
    hidden_counted: float  # power of unseen enemies counted into theirs
    hp_share: float
    structures: int = 0  # enemy structures counted on their side
    survive_s: float = math.inf  # how long it lasts at the damage rate seen so far, with enemies in reach
    flee_s: float = 0.0  # how long the walk home takes
    in_reach: int = 0  # enemies within reach right now
    kind: str = "here"  # "here" (where the hero is) or "ahead" (its destination)

    def as_json(self) -> dict[str, Any]:
        return {"seat": self.seat, "kind": self.kind, "ratio": round(self.ratio, 2), "allies": self.allies,
                "visible_enemies": self.visible_enemies, "enemy_structures": self.structures,
                "hidden_power_counted": round(self.hidden_counted), "hp_share": round(self.hp_share, 2),
                "in_reach": self.in_reach, "survive_s": None if self.survive_s == math.inf else round(self.survive_s, 1),
                "flee_s": round(self.flee_s, 1)}


class Judge:
    def __init__(self, memory: MatchMemory) -> None:
        self.m = memory

    # -- pieces ----------------------------------------------------------------------------

    def _arrival(self, last: tuple[float, float] | None, here: tuple[float, float], age_s: float) -> float:
        R = self.m.R
        reach = self.m.speed * age_s
        if last is None:
            far = self.m.extent
        else:
            d = dist(last, here)
            if d - reach > R:
                return 0.0  # cannot have got here
            far = d + reach
        return min(1.0, (R / max(far, R)) ** 2)

    def _hero_unit(self) -> float:
        """The power of one average hero: known enemies, else ours."""
        m = self.m
        known = [power(e.hp, e.level) for e in m.enemies.values() if power(e.hp, e.level) > 0]
        pool = known or [power(h.get("hp"), h.get("level")) for h in m.ours.values() if (h.get("hp") or 0) > 0]
        return sum(pool) / len(pool) if pool else 0.0

    def _structures_near(self, point: tuple[float, float], enemy: bool) -> int:
        m = self.m
        n = 0
        for s in m.structs.values():
            if s["hp"] <= 0 or (s["camp"] != m.camp) != enemy:
                continue
            if dist(pos(s), point) <= m.R:
                n += 1
        return n

    def _theirs(self, here: tuple[float, float]) -> tuple[float, float, int, int]:
        """(enemy force, of which unseen, visible heroes, enemy structures) around a point."""
        m = self.m
        theirs = hidden = 0.0
        visible = 0
        for e in m.enemies.values():
            p = power(e.hp, e.level)
            if p <= 0:
                continue
            age = (m.step - e.step) * 0.2
            if age <= 0.3:  # in sight right now
                if dist(e.at(), here) <= m.R:
                    theirs += p
                    visible += 1
                continue
            w = self._arrival(e.at(), here, age)
            theirs += w * p
            hidden += w * p
        unit = self._hero_unit()
        unseen = max(0, 5 - len(m.enemies))
        if unseen:
            w = self._arrival(None, here, 0.0)
            theirs += unseen * w * unit
            hidden += unseen * w * unit
        structs = self._structures_near(here, enemy=True)
        theirs += structs * STRUCT_HEROES * unit
        theirs = max(theirs, unit)  # the minimum credible threat
        return theirs, hidden, visible, structs

    def _ratio(self, ours: float, theirs: float) -> float:
        return BIG if theirs <= 0 else min(BIG, ours / theirs * self.m.calibration)

    @staticmethod
    def _share(me: dict[str, Any]) -> float:
        hp, mx = me.get("hp") or 0, me.get("max_hp") or 0
        return max(0.0, min(1.0, hp / mx)) if mx else 1.0

    # -- readings --------------------------------------------------------------------------

    def read(self, seat: int) -> Reading | None:
        """The fight where this hero stands."""
        m = self.m
        me = m.ours.get(seat)
        here = pos(me)
        if me is None or here is None or (me.get("hp") or 0) <= 0:
            return None
        ours = 0.0
        allies = 0
        for h in m.ours.values():
            if (h.get("hp") or 0) > 0 and dist(pos(h), here) <= m.R:
                ours += power(h.get("hp"), h.get("level"))
                allies += 1
        ours += self._structures_near(here, enemy=False) * STRUCT_HEROES * self._hero_unit()
        theirs, hidden, visible, structs = self._theirs(here)
        reach = sum(1 for e in m.enemies.values()
                    if (m.step - e.step) * 0.2 <= 0.3 and power(e.hp, e.level) > 0 and dist(e.at(), here) <= m.R)
        survive = (me.get("hp") or 0) / (m.incoming * reach) if reach else math.inf
        home = m.home()
        flee = dist(here, home) / m.speed if home is not None and m.speed > 0 else 0.0
        return Reading(seat, self._ratio(ours, theirs), ours, theirs, allies, visible, hidden,
                       self._share(me), structs, survive_s=survive, flee_s=flee, in_reach=reach)

    def read_ahead(self, seat: int) -> Reading | None:
        """The fight this hero would walk into: the nearest standing enemy structure, with only the allies who
        will be there about when it is (those no farther from it than this hero plus one engagement radius)."""
        m = self.m
        me = m.ours.get(seat)
        here = pos(me)
        if me is None or here is None or (me.get("hp") or 0) <= 0:
            return None
        targets = [pos(s) for s in m.structs.values() if s["hp"] > 0 and s["camp"] != m.camp and pos(s)]
        if not targets:
            return None
        dest = min(targets, key=lambda t: dist(t, here))
        mine = dist(here, dest)
        ours = 0.0
        allies = 0
        for h in m.ours.values():
            if (h.get("hp") or 0) > 0 and dist(pos(h), dest) <= mine + m.R:
                ours += power(h.get("hp"), h.get("level"))
                allies += 1
        theirs, hidden, visible, structs = self._theirs(dest)
        return Reading(seat, self._ratio(ours, theirs), ours, theirs, allies, visible, hidden,
                       self._share(me), structs, kind="ahead")

    @staticmethod
    def can_escape(reading: Reading) -> bool:
        """Does running home help? Only if nothing is on the hero, or it outlasts the walk. A hero that dies
        before reaching base deals no damage on the way and still dies; it should fight."""
        return reading.in_reach == 0 or reading.survive_s >= reading.flee_s

    def engage(self, reading: Reading, risk: float) -> bool:
        """Take (or stay in) the fight: the edge, discounted by how fragile this hero is, beats the risk bar."""
        return reading.ratio * max(reading.hp_share, 1e-3) >= risk

    def expected_ratio(self, readings: list[Reading]) -> float | None:
        """The team-level exchange the model expects, recorded on a fight when it starts."""
        rs = [r for r in readings if r.visible_enemies]
        if not rs:
            return None
        return max(0.05, min(BIG, math.exp(sum(math.log(max(r.ratio, 0.05)) for r in rs) / len(rs))))
