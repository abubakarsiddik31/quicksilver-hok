"""One Honor of Kings match, measured the same way every time, from its trace.

    uv run python scripts/hok_report.py                  # the newest traces/hok-*.jsonl
    uv run python scripts/hok_report.py traces/hok-X.jsonl [more traces ...]

What it reports (CLAUDE.md rule 2: judge changes on many recorded games, not on one result):
- who, which code, which settings; how it ended (the agent's own final line, else the last state);
- structure damage dealt and taken over time: the tie-break currency;
- kills and deaths; where the heroes spent their time (at base, our half, their half, dead);
- retreats sent with no enemy in reach (healing trips) and the hp they left at;
- what decided the orders (fight model, hold, cornered, defend, Jev), the General's plans and latency, Jev's answers
  and its record on finished fights; fights with expected against realised exchange.
Reads only the trace; changes nothing.
"""

from __future__ import annotations

import collections
import glob
import json
import sys


def load(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass  # a line cut off by a crash
    return rows


def report(path: str) -> None:
    rows = load(path)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    print(f"=== {path}")
    meta = (by["meta"] or [{}])[0]
    camp = meta.get("camp")
    print(
        f"camp {camp} | code {meta.get('code')} | General {meta.get('general')} ({meta.get('model')}) | "
        f"baseline camp {meta.get('baseline_camp')} | session {(meta.get('context') or {}).get('session_id')}"
    )
    states = by["state"]
    if not states:
        print("no state lines")
        return
    last = states[-1]
    final = (by["final"] or [None])[-1]
    proj = (final or {}).get("projection") or {}
    print(
        f"ended at {last['step'] * 0.2:.0f}s | dealt {last['dealt']} taken {last['taken']} | "
        f"projection {proj.get('if_decided_now', '?')} by {proj.get('decided_by', '?')} | final line: {bool(final)}"
    )

    print("\nstructure damage over time (dealt / taken), and structures standing")
    for i, s in enumerate(states):
        if i % 60 and s is not last:
            continue
        ours = [
            x["hp"]
            for x in s["structures"]
            if x["camp"] == camp and x["type"] in ("ACTOR_TOWER", "ACTOR_CRYSTAL")
        ]
        theirs = [
            x["hp"]
            for x in s["structures"]
            if x["camp"] != camp and x["type"] in ("ACTOR_TOWER", "ACTOR_CRYSTAL")
        ]
        k = sum(h.get("kills") or 0 for h in s["heroes"].values())
        d = sum(h.get("deaths") or 0 for h in s["heroes"].values())
        print(
            f"  {s['step'] * 0.2:5.0f}s  dealt {s['dealt']:6}  taken {s['taken']:6}  kills/deaths {k}/{d}  "
            f"ours {ours}  theirs {theirs}"
        )

    # where the heroes spent their time, from the state samples
    springs = {
        x["camp"]: (x["x"], x["z"])
        for x in states[0]["structures"]
        if x["type"] == "ACTOR_TOWER_SPRING"
    }
    own = springs.get(camp)
    foe = next((v for k, v in springs.items() if k != camp), None)
    if own and foe:
        L2 = (foe[0] - own[0]) ** 2 + (foe[1] - own[1]) ** 2
        place = collections.Counter()
        for s in states:
            for h in s["heroes"].values():
                if not h.get("hp"):
                    place["dead"] += 1
                    continue
                f = (
                    (h["x"] - own[0]) * (foe[0] - own[0]) + (h["z"] - own[1]) * (foe[1] - own[1])
                ) / L2
                place["at base" if f < 0.1 else "our half" if f < 0.5 else "their half"] += 1
        n = sum(place.values())
        print("\nhero time: " + " | ".join(f"{k} {v / n:.0%}" for k, v in place.most_common()))

    orders = by["orders"]
    retreats = [o for o in orders if o["order"].get("macro") == "macro_retreat"]
    quiet = [o for o in retreats if (o.get("reading") or {}).get("in_reach") == 0]
    hps = sorted((o.get("reading") or {}).get("hp_share", 1) for o in quiet)
    print(
        f"\norders {len(orders)} | retreats {len(retreats)}, of which with no enemy in reach {len(quiet)}"
        + (f" (median hp {hps[len(hps) // 2]:.0%})" if hps else "")
    )
    print(
        "decided by: "
        + ", ".join(
            f"{k} {v}" for k, v in collections.Counter(o["source"] for o in orders).most_common()
        )
    )
    print(
        "macros: "
        + ", ".join(
            f"{k} {v}"
            for k, v in collections.Counter(
                o["order"].get("macro") or "raw" for o in orders
            ).most_common()
        )
    )

    plans = by["plan"]
    if plans:
        lat = sorted(p["latency_s"] for p in plans if p.get("latency_s") is not None)
        print(
            f"\nGeneral: {len(plans)} consults, ok {sum(1 for p in plans if p['ok'])}, "
            f"outcomes {dict(collections.Counter(p.get('outcome') for p in plans))}, median latency "
            f"{lat[len(lat) // 2] if lat else '?'}s"
        )
        postures = collections.Counter(
            (p.get("plan") or {}).get("posture") for p in plans if p.get("plan")
        )
        print("  postures chosen: " + ", ".join(f"{k} {v}" for k, v in postures.most_common()))
    jev = by["jev"]
    if jev:
        ok = [j for j in jev if j.get("advice")]
        stances = collections.Counter(v[0] for j in ok for v in j["advice"]["hero"].values())
        lat = sorted(j["advice"]["latency_s"] for j in ok)
        print(
            f"Jev: {len(jev)} calls, answered {len(ok)}, median latency {lat[len(lat) // 2] if lat else '?'}s, "
            f"hero stances {dict(stances)}; record {(final or {}).get('jev')}"
        )

    fights = (final or {}).get("fights") or (states[-1].get("fights") or [])
    if fights:
        print(f"\nfights ({len(fights)}): where, expected -> realised exchange, our deaths / kills")
        for f in fights:
            print(
                f"  {f.get('start_s'):>4}s {str(f.get('where')):10} {f.get('expected_ratio')} -> {f.get('realised_ratio')}"
                f"  deaths {f.get('our_deaths')} kills {f.get('our_kills')}  jev {f.get('jev_win_p')}"
            )
    fb = collections.Counter(r.get("kind") for r in by["fallback"])
    print(f"\nfallbacks: {dict(fb) or 'none'}")
    if final:
        print(f"calibration {final.get('calibration')} | radius {final.get('radius')}")


def main(argv: list[str]) -> int:
    paths = argv[1:] or sorted(glob.glob("traces/hok-*.jsonl"), key=lambda p: p)[-1:]
    if not paths:
        print("no traces/hok-*.jsonl found")
        return 1
    for p in paths:
        report(p)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
