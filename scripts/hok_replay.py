"""Replay a recorded HoK trace through the current memory, fight model and executor decision (offline, free).

For every recorded `state` line (about one a second) the memory is updated as in the match, every hero is read by
the fight model and `_decide` is asked what it would order now, with the plan that was in force then. Printed:
how often the new code would send a hero to one of our structures being hit while the old order walked it the
other way, and the General's `tempo` briefing at a few fixed times.

    uv run python scripts/hok_replay.py traces/hok-1792182036-88694.jsonl
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

os.environ["HOK_TRACE"] = "0"  # a replay must never write a trace file of its own

from hok.agent import HokAgent
from hok.memory import STEP_S
from hok.plan import Plan


def main(path: str) -> int:
    agent = HokAgent(None, None, log=lambda m: None)
    m = agent.memory
    plan, max_steps = Plan(), 3000
    last_order: dict[int, str] = {}
    counts: Counter = Counter()
    pushes: Counter = Counter()  # hero-decisions walking on under "defend" because the destination is undefended
    away_while_hit = moved_to_defend = 0
    shown: set[int] = set()
    for line in open(path):
        r = json.loads(line)
        kind = r.get("kind")
        if kind == "meta":
            max_steps = r.get("max_steps") or max_steps
        elif kind == "plan" and r.get("ok") and r.get("plan"):
            p = r["plan"]
            plan = Plan(
                posture=p["posture"],
                risk=p["risk"],
                heroes={int(k): v for k, v in (p.get("heroes") or {}).items()},
                jev_trust=p.get("jev_trust", 0.8),
            )
        elif kind == "orders" and r.get("order", {}).get("macro"):
            last_order[r["order"]["seat"]] = r["order"]["macro"]
        elif kind == "state":
            ours = {int(k): v for k, v in (r.get("heroes") or {}).items()}
            view = {
                "camp": r.get("camp"),
                "step": r.get("step"),
                "seconds_left": r.get("seconds_left"),
                "max_steps": max_steps,
                "ours": ours,
                "enemies": r.get("enemies") or {},
                "structures": [
                    s for s in (r.get("structures") or []) if s.get("type") != "ACTOR_NONE"
                ],
            }
            m.update(view)
            agent._readings, agent._ahead, agent._defend = {}, {}, {}
            for seat in ours:
                rd = agent.judge.read(seat)
                if rd is not None:
                    agent._readings[seat] = rd
                a = agent.judge.read_ahead(seat)
                if a is not None:
                    agent._ahead[seat] = a
            siege = m.defend_point()
            macros = {
                "macro_advance": "Move toward the nearest enemy tower",
                "macro_retreat": "Move back toward your own base",
            }
            for seat, h in ours.items():
                if (h.get("hp") or 0) <= 0:
                    continue
                want, source = agent._decide(None, seat, macros, plan)  # type: ignore[arg-type]
                counts[source] += 1
                if plan.posture == "defend" and want == "macro_advance" and source == "model":
                    pushes[int(m.step * STEP_S) // 60] += 1
                if siege is not None:
                    toward = agent._toward(seat, siege, macros)
                    if (
                        toward is not None
                        and last_order.get(seat) in macros
                        and last_order[seat] != toward
                    ):
                        away_while_hit += 1
                        if want == toward:
                            moved_to_defend += 1
            t = int(m.step * STEP_S)
            for mark in (150, 300, 450, 500, 540):
                if t >= mark and mark not in shown:
                    shown.add(mark)
                    print(f"--- tempo at {t}s:", json.dumps(m.tempo()))
    print("decision sources (new code):", dict(counts))
    print(
        f"hero-seconds walking away from our structure under attack (old orders): {away_while_hit}; "
        f"new code sends them to it instead: {moved_to_defend}"
    )
    print("defend-posture walk-ons into an undefended destination, per minute:", dict(sorted(pushes.items())))
    print(f"final: dealt {round(m.dealt)} taken {round(m.taken)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
