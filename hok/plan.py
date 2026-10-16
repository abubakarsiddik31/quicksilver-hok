"""The plan the General writes and the executor carries out.

A plan is standing guidance, not a move: the team's posture, how much edge the fight model must see before a
hero takes a fight (``risk``), optional per-hero orders, and when the General wants to be consulted next. The
executor re-applies it to every fresh state (about five times a second), so the model's latency never leaves a
hero without an order.

The model's text is untrusted: every field is checked against a closed set or a range, and anything unusable
keeps the previous plan's value.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any

POSTURES = ("pressure", "balanced", "defend")
STANCES = ("auto", "engage", "retreat")
RISK_MIN, RISK_MAX = 0.4, 3.0
REVIEW_MIN_S, REVIEW_MAX_S = 4.0, 40.0
ASSESSMENT_MAX = 300


@dataclass(frozen=True)
class Plan:
    posture: str = "balanced"
    risk: float = 1.0
    heroes: dict[int, str] = field(default_factory=dict)  # seat -> stance override; absent means auto
    review_in_s: float = 15.0
    assessment: str = "default plan"

    def stance(self, seat: int) -> str:
        return self.heroes.get(seat, "auto")

    def as_json(self) -> dict[str, Any]:
        return {"posture": self.posture, "risk": self.risk,
                "heroes": {str(s): v for s, v in sorted(self.heroes.items())},
                "review_in_s": self.review_in_s, "assessment": self.assessment}


DEFAULT_PLAN = Plan()


def _clamp(v: Any, lo: float, hi: float, default: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return default
    return max(lo, min(hi, float(v)))


def parse_plan(text: str | None, previous: Plan, seats: list[int]) -> Plan | None:
    """The General's JSON as a Plan, or None when it is unusable."""
    if not isinstance(text, str):
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    posture = data.get("posture") if data.get("posture") in POSTURES else previous.posture
    heroes: dict[int, str] = {}
    raw = data.get("heroes")
    if isinstance(raw, dict):
        for k, v in raw.items():
            if str(k).isdigit() and int(k) in seats and v in STANCES and v != "auto":
                heroes[int(k)] = v
    elif "heroes" not in data:
        heroes = dict(previous.heroes)
    assessment = str(data.get("assessment") or "").strip()[:ASSESSMENT_MAX] or previous.assessment
    return replace(
        previous,
        posture=posture,
        risk=_clamp(data.get("risk"), RISK_MIN, RISK_MAX, previous.risk),
        heroes=heroes,
        review_in_s=_clamp(data.get("review_in_s"), REVIEW_MIN_S, REVIEW_MAX_S, previous.review_in_s),
        assessment=assessment,
    )


def plan_schema(seats: list[int]) -> dict[str, Any]:
    """The JSON Schema the model's answer must follow (the API enforces it; ``parse_plan`` still validates)."""
    return {
        "type": "object",
        "properties": {
            "posture": {"type": "string", "enum": list(POSTURES)},
            "risk": {"type": "number", "minimum": RISK_MIN, "maximum": RISK_MAX},
            "heroes": {
                "type": "object",
                "properties": {str(s): {"type": "string", "enum": ["engage", "retreat"]} for s in seats},
                "additionalProperties": False,
            },
            "review_in_s": {"type": "number", "minimum": REVIEW_MIN_S, "maximum": REVIEW_MAX_S},
            "assessment": {"type": "string"},
        },
        "required": ["posture", "risk", "heroes", "review_in_s", "assessment"],
        "additionalProperties": False,
    }
