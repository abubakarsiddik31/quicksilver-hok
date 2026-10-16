"""Game-type dispatch — the single ``create_agent()`` the runtime looks for.

The runtime calls ``create_agent()`` once per match, in that match's own
independent process, *before* it knows which game the match is. The game type
only arrives with the first decision, as ``context.game_type``. So this agent
defers: it builds the game-specific logic lazily on the first call and
forwards every later decision to it. One instance is therefore one game.

State in the delegate stays isolated to that game automatically, because the
runtime starts a fresh process — and a fresh ``create_agent()`` — per match.
"""

from __future__ import annotations

import os
import time
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from altruagent import TERMINATE_MESSAGING, DecisionContext, GameState
from common import trace

# Which package owns which game. Set ENABLE_RED_ALERT env to route red_alert
# to redalert.agent; otherwise falls back to common.fallback to satisfy starter test contract.
_FACTORY_MODULES = {
    "pokemon_vgc_doubles_draft": "pokemon.agent",
    "werewolf": "werewolf.agent",
    "honor_of_kings_5v5": "hok.agent",
}
_FALLBACK_MODULE = "common.fallback"


def _create_delegate(game_type: str | None):
    """Import and call the game's factory. Imported lazily so one broken
    package can't take down dispatch for the other games."""
    if game_type == "red_alert" and os.environ.get("ENABLE_RED_ALERT", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        module_name = "redalert.agent"
    else:
        module_name = _FACTORY_MODULES.get(game_type or "", _FALLBACK_MODULE)
    module = __import__(module_name, fromlist=["create_agent"])
    if hasattr(module, "create_agent"):
        return module.create_agent()
    return module


def _trace_game_end(final_state: Any, context: DecisionContext) -> None:
    """Record how a finished game ended, so traces carry the one thing they
    otherwise never show: the result. Total — a failure here must never
    matter to the match."""
    try:
        from altruagent.worker import _score, game_outcome

        agent_id = getattr(context, "agent_id", None)
        outcome = game_outcome(final_state, agent_id)
        trace.write(
            "final",
            session=getattr(context, "session_id", None),
            agent=agent_id,
            outcome=outcome,
            score=_score(final_state, agent_id),
            termination_reason=getattr(final_state, "termination_reason", None),
        )
        if getattr(context, "game_type", None) == "honor_of_kings_5v5":
            from hok import trace as hok_trace

            hok_trace.write(
                "final",
                outcome=outcome,
                score=_score(final_state, agent_id),
                termination_reason=getattr(final_state, "termination_reason", None),
                raw=getattr(final_state, "final_result", None),
            )
    except Exception:  # noqa: BLE001 - bookkeeping must never cost a match
        pass


def _trace_runner_game_ends() -> None:
    """Wrap ``altruagent.runner.run_game`` so every finished match leaves a
    ``final`` record in its trace. The terminal state reaches the runner as a
    *read*, which ``on_action_result`` never sees — this is the only hook that
    does see it. Applies per worker process (the runtime builds a fresh agent,
    and re-imports this module, for each match)."""
    try:
        from altruagent import runner as _runner

        original = _runner.run_game

        def run_game_traced(game, context, choose_action, **kwargs):
            final_state = original(game, context, choose_action, **kwargs)
            _trace_game_end(final_state, context)
            return final_state

        _runner.run_game = run_game_traced
    except Exception:  # noqa: BLE001 - tracing must never break the runtime
        pass


_trace_runner_game_ends()


def _call(delegate: Any, method_name: str, state: GameState, context: DecisionContext):
    """Mirror the runtime's own resolution rules (``altruagent.runner``): a
    plain callable is called directly, otherwise its named method is used."""
    if callable(delegate):
        return delegate(state, context)
    return getattr(delegate, method_name)(state, context)


class QuicksilverAgent:
    """One game's decision logic, chosen by that game's type."""

    def __init__(self) -> None:
        self.game_type: str | None = None
        self._delegate: Any = None

    def _delegate_for(self, context: DecisionContext):
        if self._delegate is None:
            self.game_type = getattr(context, "game_type", None)
            trace.set_game(self.game_type)
            self._delegate = _create_delegate(self.game_type)
        return self._delegate

    def _traced(self) -> bool:
        # Red Alert and Honor of Kings write their own, richer traces.
        return self.game_type not in ("red_alert", "honor_of_kings_5v5")

    def choose_action(self, state: GameState, context: DecisionContext):
        delegate = self._delegate_for(context)
        started = time.time()
        action = _call(delegate, "choose_action", state, context)
        if self._traced():
            trace.write(
                "decision",
                session=getattr(context, "session_id", None),
                agent=getattr(context, "agent_id", None),
                seat=getattr(context, "seat_position", None),
                action=action,
                latency_s=round(time.time() - started, 3),
                **trace.state_fields(state),
            )
        return action

    def on_action_result(self, result: dict, context: DecisionContext) -> None:
        """Forward the runner's per-move answer to a delegate that wants it (Red Alert)."""
        if self._traced():
            trace.write("result", result=result)
        hook = getattr(self._delegate, "on_action_result", None)
        if callable(hook):
            hook(result, context)

    def choose_message(self, state: GameState, context: DecisionContext):
        delegate = self._delegate_for(context)
        method = getattr(delegate, "choose_message", None)
        if callable(method):
            said = method(state, context)
            if self._traced():
                trace.write("message", said=said, **trace.state_fields(state))
            return said
        # The runtime's own default for an agent that doesn't talk: vote the
        # discussion window closed. Replicated here because this object *does*
        # define choose_message, so the runtime will always call it and the
        # default would otherwise never apply.
        return TERMINATE_MESSAGING


def create_agent():
    return QuicksilverAgent()
