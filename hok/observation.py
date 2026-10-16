"""Total readers for an Honor of Kings state: a missing or odd field gives an empty value, never an exception."""

from __future__ import annotations

import re
from typing import Any

MACROS = (
    "macro_advance",
    "macro_retreat",
    "macro_attack",
    "macro_skill1",
    "macro_skill2",
    "macro_skill3",
)
_SEAT_ID = re.compile(r"^s(\d+):(.+)$")
_DISTANCE = re.compile(r"(\d+) away\)")


def raw(state: Any) -> dict[str, Any]:
    r = getattr(state, "raw", None)
    return r if isinstance(r, dict) else {}


def is_realtime(state: Any) -> bool:
    pacing = raw(state).get("pacing")
    return isinstance(pacing, dict) and pacing.get("mode") == "realtime"


def offers_by_seat(legal_actions: Any) -> dict[int, dict[str, Any]]:
    """{seat: {"macros": {macro_id: label}, "raw": [LegalAction, ...]}} from ``s<seat>:<id>`` actions."""
    out: dict[int, dict[str, Any]] = {}
    for action in legal_actions or []:
        m = _SEAT_ID.match(str(getattr(action, "action_id", "")))
        if not m:
            continue
        seat, base = int(m.group(1)), m.group(2)
        entry = out.setdefault(seat, {"macros": {}, "raw": []})
        if base in MACROS:
            entry["macros"][base] = getattr(action, "label", None) or base
        else:
            entry["raw"].append(action)
    return out


def _simultaneous(state: Any) -> dict[str, Any]:
    s = raw(state).get("simultaneous")
    return s if isinstance(s, dict) else {}


def standing_orders(state: Any) -> dict[int, str]:
    orders = _simultaneous(state).get("standing_orders")
    if not isinstance(orders, dict):
        return {}
    return {int(k): v for k, v in orders.items() if str(k).isdigit() and isinstance(v, str)}


def pending_seats(state: Any, offers: dict[int, Any]) -> list[int]:
    pending = _simultaneous(state).get("your_pending_seats")
    if isinstance(pending, list):
        return [s for s in pending if s in offers]
    return sorted(offers)


def seat_summary(state: Any, seat: int) -> dict[str, Any]:
    obs = raw(state).get("observation")
    seats = obs.get("seats") if isinstance(obs, dict) else None
    entry = seats.get(str(seat)) if isinstance(seats, dict) else None
    summary = entry.get("summary") if isinstance(entry, dict) else None
    return summary if isinstance(summary, dict) else {}


def distance(label: str) -> float:
    """Macro labels end "(... 4123 away)"; inf without one."""
    m = _DISTANCE.search(label or "")
    return float(m.group(1)) if m else float("inf")


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def team_view(state: Any) -> dict[str, Any]:
    """The whole team's picture, merged over our five heroes' own views.

    ``ours`` {seat: my_hero}, ``enemies`` {runtime_id: hero dict, visible or not}, ``structures`` (one copy,
    every seat shows the same list), plus the match clock. Fields the server omits stay None; nothing raises.
    """
    r = raw(state)
    obs = r.get("observation")
    seats = obs.get("seats") if isinstance(obs, dict) else None
    ours: dict[int, dict[str, Any]] = {}
    enemies: dict[str, dict[str, Any]] = {}
    structures: list[dict[str, Any]] = []
    camp = r.get("camp")
    for key in sorted(seats or {}, key=lambda k: int(k) if str(k).isdigit() else 99):
        if not str(key).isdigit():
            continue
        summary = seat_summary(state, int(key))
        me = summary.get("my_hero")
        if isinstance(me, dict):
            ours[int(key)] = me
            camp = camp if camp is not None else me.get("camp")
        for other in summary.get("other_heroes") or []:
            if not isinstance(other, dict):
                continue
            if camp is not None and other.get("camp") == camp:
                continue  # an ally of ours, seen by another hero
            rid = str(other.get("runtime_id", other.get("hero_id")))
            old = enemies.get(rid)
            if old is None or (other.get("visible") and not old.get("visible")):
                enemies[rid] = other
        if not structures and isinstance(summary.get("structures"), list):
            # ACTOR_NONE is a placeholder object far off the map, not something that can be attacked or attack
            structures = [s for s in summary["structures"]
                          if isinstance(s, dict) and s.get("type") != "ACTOR_NONE"]
    return {
        "camp": camp,
        "step": int(getattr(state, "state_version", 0) or 0),
        "seconds_left": _num(r.get("seconds_left")),
        "max_steps": _num(r.get("max_steps")),
        "ours": ours,
        "enemies": enemies,
        "structures": structures,
    }
