"""The General's model client: typed outcomes, bounded retries, a circuit breaker, and a request shape that adapts.

``common.llm.GeminiClient`` turns every failure into ``None``, so a rejected request looks exactly like a silent
model. The General needs to know *why* a call failed, because the right reaction differs:

* ``ok``           a usable answer
* ``blocked``      the prompt or answer was blocked (``promptFeedback.blockReason`` / ``finishReason``): retry once
                   with a lean briefing is the caller's choice, repeated blocks are counted and given up on
* ``truncated``    ``MAX_TOKENS`` (thinking tokens count against the limit): the JSON is not trustworthy
* ``rate_limited`` 429: back off, honouring ``Retry-After``
* ``transient``    5xx / network / timeout: retry with exponential backoff and jitter inside the deadline
* ``fatal``        400/403/404 after the request-shape ladder is exhausted, or a bad key: never retried, and the
                   General is switched off for the match with a loud log
* ``malformed``    200 with no usable text

Request shapes differ across Gemini generations (``thinkingLevel`` vs ``thinkingBudget``, ``temperature`` being
dropped for newer models, a response schema). Rather than guess, a short ladder is tried on a 400, newest first;
the first shape the server accepts is pinned for the rest of the match. Authentication goes in a header, never in
the URL, so the key cannot reach a log line.

Every method is total: nothing here raises into the game loop.
"""

from __future__ import annotations

import random
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

DEADLINE_S = 25.0  # wall-clock budget for one consult, retries included (httpx timeouts are per phase)
MAX_OUTPUT_TOKENS = 4096  # thinking tokens count against this
RETRY_FIRST_S, RETRY_MAX_S = 1.0, 8.0
BREAKER_FAILS = 4
BREAKER_COOLDOWN_S, BREAKER_COOLDOWN_MAX_S = 20.0, 120.0
BLOCKED_GIVE_UP = 3

SAFETY = [
    {"category": c, "threshold": "BLOCK_NONE"}
    for c in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]


@dataclass
class Outcome:
    kind: str
    text: str | None = None
    detail: str = ""
    latency_s: float = 0.0
    shape: str = ""
    thought_tokens: int | None = None

    @property
    def ok(self) -> bool:
        return self.kind == "ok"


def _shapes(level: str, schema: dict[str, Any] | None) -> list[tuple[str, dict[str, Any], bool]]:
    """(name, generationConfig additions, send temperature) from newest to the shape proven to work."""
    json_cfg: dict[str, Any] = {"responseMimeType": "application/json"}
    ladder: list[tuple[str, dict[str, Any], bool]] = []
    if schema is not None:
        ladder.append(("level+schema", {**json_cfg, "responseJsonSchema": schema,
                                        "thinkingConfig": {"thinkingLevel": level}}, False))
    ladder.append(("level", {**json_cfg, "thinkingConfig": {"thinkingLevel": level}}, False))
    ladder.append(("budget", {**json_cfg, "thinkingConfig": {"thinkingBudget": 1024}}, True))
    ladder.append(("legacy", {**json_cfg, "thinkingConfig": {"thinkingBudget": 0}}, True))
    return ladder


class GeneralModel:
    """Gemini for the General. ``generate`` returns the text or None, like ``GeminiClient.generate``, and
    ``last`` / ``disabled`` say why."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        base_url: str,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self._key = api_key
        self.model = model
        self._base = base_url.rstrip("/")
        self._transport = transport
        self._clock = clock
        self._sleep = sleep
        self._log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self.last: Outcome | None = None
        self.disabled: str | None = None
        self.counts: dict[str, int] = {}
        self._pinned: int | None = None  # index into the shape ladder that the server accepted
        self._fails = 0
        self._open_until = 0.0
        self._cooldown = BREAKER_COOLDOWN_S
        self._blocked = 0

    @classmethod
    def from_gemini(cls, client: Any, **kw: Any) -> GeneralModel | None:
        if client is None or not getattr(client, "api_key", None):
            return None
        return cls(client.api_key, model=client.model, base_url=client.base_url, **kw)

    # -- public ----------------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        level: str = "medium",
        timeout: float = DEADLINE_S,
        **_: Any,
    ) -> str | None:
        out = self.call(prompt, system=system, schema=schema, level=level, deadline_s=timeout)
        return out.text if out.ok else None

    def call(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        level: str = "medium",
        deadline_s: float = DEADLINE_S,
    ) -> Outcome:
        started = self._clock()
        try:
            out = self._call(prompt, system, schema, level, started + deadline_s)
        except Exception as exc:  # noqa: BLE001 - a model must never break the game loop
            out = Outcome("transient", detail=f"unexpected {type(exc).__name__}")
        out.latency_s = round(self._clock() - started, 2)
        self.last = out
        self.counts[out.kind] = self.counts.get(out.kind, 0) + 1
        self._account(out)
        return out

    # -- internals -------------------------------------------------------------------------

    def _call(self, prompt: str, system: str | None, schema: dict | None, level: str, until: float) -> Outcome:
        if self.disabled:
            return Outcome("fatal", detail=self.disabled)
        if self._clock() < self._open_until:
            return Outcome("transient", detail="circuit open")
        ladder = _shapes(level, schema)
        idx = self._pinned if self._pinned is not None and self._pinned < len(ladder) else 0
        retry = RETRY_FIRST_S
        while True:
            name, cfg, with_temp = ladder[idx]
            try:
                resp = self._post(prompt, system, cfg, with_temp, max(1.0, until - self._clock()))
            except httpx.TimeoutException:
                kind, detail, wait = "transient", "timeout", retry
            except httpx.HTTPError as exc:
                kind, detail, wait = "transient", f"network {type(exc).__name__}", retry
            else:
                if resp.status_code == 200:
                    out = self._read(resp)
                    out.shape = name
                    if out.kind in ("ok", "truncated", "blocked"):
                        self._pinned = idx
                    return out
                kind, detail, wait = self._classify(resp, retry)
                if kind == "shape" and idx + 1 < len(ladder):
                    self._log(f"[general] request shape '{name}' refused ({detail}); trying '{ladder[idx + 1][0]}'")
                    idx += 1
                    continue
                if kind == "shape":
                    kind = "fatal"
                if kind == "fatal":
                    return Outcome("fatal", detail=detail, shape=name)
            if self._clock() + wait >= until:
                return Outcome(kind if kind != "shape" else "fatal", detail=detail + " (deadline)", shape=name)
            self._sleep(wait * (0.5 + random.random()))
            retry = min(RETRY_MAX_S, retry * 2)

    def _post(self, prompt: str, system: str | None, cfg: dict, with_temp: bool, timeout: float):
        gen: dict[str, Any] = {"maxOutputTokens": MAX_OUTPUT_TOKENS, **cfg}
        if with_temp:
            gen["temperature"] = 0.4
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": gen,
            "safetySettings": SAFETY,
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        with httpx.Client(timeout=timeout, transport=self._transport) as client:
            return client.post(
                f"{self._base}/models/{self.model}:generateContent",
                headers={"Content-Type": "application/json", "x-goog-api-key": self._key},
                json=body,
            )

    @staticmethod
    def _classify(resp: httpx.Response, retry: float) -> tuple[str, str, float]:
        code = resp.status_code
        try:
            msg = str((resp.json().get("error") or {}).get("message") or "")[:200]
        except Exception:  # noqa: BLE001
            msg = ""
        if code == 429:
            m = re.search(r"\d+(\.\d+)?", resp.headers.get("retry-after", ""))
            return "rate_limited", f"429 {msg}", float(m.group(0)) if m else retry
        if code in (408, 500, 502, 503, 504):
            return "transient", f"{code} {msg}", retry
        if code == 400:
            return "shape", f"400 {msg}", 0.0
        return "fatal", f"{code} {msg}", 0.0

    @staticmethod
    def _read(resp: httpx.Response) -> Outcome:
        try:
            body = resp.json()
        except ValueError:
            return Outcome("malformed", detail="not JSON")
        if not isinstance(body, dict):
            return Outcome("malformed", detail="not an object")
        usage = body.get("usageMetadata") if isinstance(body.get("usageMetadata"), dict) else {}
        thought = usage.get("thoughtsTokenCount")
        feedback = body.get("promptFeedback")
        if isinstance(feedback, dict) and feedback.get("blockReason"):
            return Outcome("blocked", detail=f"prompt {feedback['blockReason']}")
        cands = body.get("candidates")
        cand = cands[0] if isinstance(cands, list) and cands and isinstance(cands[0], dict) else None
        if cand is None:
            return Outcome("malformed", detail="no candidate")
        reason = str(cand.get("finishReason") or "")
        parts = (cand.get("content") or {}).get("parts") if isinstance(cand.get("content"), dict) else None
        text = "".join(p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)) \
            if isinstance(parts, list) else ""
        if reason == "MAX_TOKENS":
            return Outcome("truncated", detail="MAX_TOKENS", thought_tokens=thought)
        if reason in ("SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "OTHER"):
            return Outcome("blocked", detail=f"answer {reason}")
        if not text.strip():
            return Outcome("malformed", detail=f"empty ({reason})")
        return Outcome("ok", text=text.strip(), thought_tokens=thought)

    def _account(self, out: Outcome) -> None:
        if out.kind == "fatal":
            self.disabled = out.detail or "fatal"
            self._log(f"[general] model switched off for this match: {self.disabled}")
            return
        if out.kind == "blocked":
            self._blocked += 1
            if self._blocked >= BLOCKED_GIVE_UP:
                self.disabled = f"blocked {self._blocked} times ({out.detail})"
                self._log(f"[general] model switched off for this match: {self.disabled}")
            return
        if out.ok:
            self._fails, self._cooldown = 0, BREAKER_COOLDOWN_S
            return
        if out.detail == "circuit open":
            return
        self._fails += 1
        if self._fails >= BREAKER_FAILS:
            self._open_until = self._clock() + self._cooldown
            self._log(f"[general] circuit open for {self._cooldown:.0f}s after {self._fails} failures ({out.kind})")
            self._cooldown = min(BREAKER_COOLDOWN_MAX_S, self._cooldown * 2)
            self._fails = BREAKER_FAILS - 1  # one failed probe re-opens it
