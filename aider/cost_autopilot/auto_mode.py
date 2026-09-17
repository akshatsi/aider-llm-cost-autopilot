"""Wires the router into a live, ongoing Aider chat session: --model
auto (or /model auto mid-session) routes each message to whichever
ladder model the router picks for it, instead of one model being fixed
for the whole session.

Deliberately narrower than orchestrator.py: no test-based validation,
no escalation, no revert. The user reviews and tests changes the same
way they would in any other Aider session -- AUTO only answers "which
model handles this message," nothing more. Escalation inside a live,
multi-turn conversation is a materially bigger question (what does a
revert mean mid-conversation, with earlier turns already in history?)
and was deliberately left out of this first version.
"""

from __future__ import annotations

from typing import Callable

from aider.cost_autopilot.orchestrator import _model_for
from aider.cost_autopilot.router import pick_starting_model

AUTO_SENTINEL = "auto"

DEFAULT_AUTO_LADDER = ["ollama/llama3.2:1b", "ollama/qwen2.5:7b", "ollama/qwen2.5:14b"]


def is_auto_model_name(model_name: str) -> bool:
    return (model_name or "").strip().lower() == AUTO_SENTINEL


def enable_auto_routing(
    coder,
    ladder: list[str] | None = None,
    route_fn: Callable[[str, list[str]], str] | None = None,
    model_factory: Callable[[str], object] | None = None,
) -> None:
    """Instance-level wrap of coder.send, not coder.send_message --
    send already has a model=None -> self.main_model fallback built
    in, so this only needs to intervene in that fallback rather than
    duplicate any of send_message's message-formatting/cost/spinner
    logic.

    Reads the just-appended user turn straight off coder.cur_messages
    rather than re-parsing the fully formatted message list handed to
    send(), since that's the literal, unmodified prompt text the user
    typed -- exactly what "calls the right model with the prompt"
    means.

    route_fn and model_factory are injectable for the same reason
    route_fn is injectable on pick_starting_model itself: tests supply
    fakes instead of running real router inference or constructing a
    real litellm-backed Model."""
    resolved_ladder = list(ladder or DEFAULT_AUTO_LADDER)
    picker = route_fn or pick_starting_model
    build_model = model_factory or _model_for
    model_cache = {}
    original_send = coder.send

    def routed_send(messages, model=None, functions=None):
        if model is None:
            prompt = coder.cur_messages[-1]["content"] if coder.cur_messages else ""
            picked_name = picker(prompt, resolved_ladder)
            if picked_name not in model_cache:
                model_cache[picked_name] = build_model(picked_name)
            model = model_cache[picked_name]
            coder.io.tool_output(f"cost_autopilot: auto-routed to {picked_name}")
        return original_send(messages, model=model, functions=functions)

    coder.send = routed_send
    coder.cost_autopilot_auto_ladder = resolved_ladder
