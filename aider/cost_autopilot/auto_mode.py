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


def parse_ladder(raw: str) -> list[str]:
    """Parses --auto-ladder's and /model auto's comma-separated ladder
    string into an ordered list, cheapest first. Shared by both entry
    points so the format only needs to be defined once."""
    models = [m.strip() for m in raw.split(",") if m.strip()]
    if not models:
        raise ValueError(f"--auto-ladder got no usable model names: {raw!r}")
    return models


def enable_auto_routing(
    coder,
    ladder: list[str] | None = None,
    route_fn: Callable[[str, list[str]], str] | None = None,
    model_factory: Callable[[str], object] | None = None,
) -> None:
    """Instance-level wrap of coder.run_one -- the one place that
    receives the genuine top-level user_message directly, once per
    real turn, before Aider does anything else with it.

    This used to wrap coder.send instead (which also has a model=None
    -> self.main_model fallback, a seam Step 4 had already found). That
    was wrong in two ways a wrap this early fixes:

    1. send_message() builds its "Waiting for {model}" spinner from
       self.main_model *before* send() ever runs, so a wrap at send()
       always showed the stale placeholder model, never the one
       actually routed to. Setting main_model here, before run_one
       hands off to send_message, means the spinner (and cost/token
       accounting, which also reads main_model) reflect the real
       choice.
    2. send() gets called again for every reflection Aider runs on
       its own output (a parse/lint/test error fed back for the same
       model to retry) -- wrapping there meant re-routing reflection
       *error text* through a classifier trained on natural-language
       coding prompts, breaking the "same model gets a chance to fix
       its own mistake" assumption reflection depends on. run_one's
       user_message argument is untouched by its internal reflection
       loop, so routing here happens exactly once per real message.

    Skips routing entirely for a slash command or an empty message --
    neither is a prompt for the classifier, and send_message() (where
    the old wrap lived) was never reached for either case anyway, so
    this preserves that.

    route_fn and model_factory are injectable for the same reason
    route_fn is injectable on pick_starting_model itself: tests supply
    fakes instead of running real router inference or constructing a
    real litellm-backed Model."""
    resolved_ladder = list(ladder or DEFAULT_AUTO_LADDER)
    picker = route_fn or pick_starting_model
    build_model = model_factory or _model_for
    model_cache = {}
    original_run_one = coder.run_one

    def routed_run_one(user_message, preproc):
        is_slash_command = preproc and coder.commands.is_command(user_message)
        if user_message and not is_slash_command:
            picked_name = picker(user_message, resolved_ladder)
            if picked_name not in model_cache:
                model_cache[picked_name] = build_model(picked_name)
            coder.main_model = model_cache[picked_name]
            coder.io.tool_output(f"cost_autopilot: auto-routed to {picked_name}")
        return original_run_one(user_message, preproc)

    coder.run_one = routed_run_one
    coder.cost_autopilot_auto_ladder = resolved_ladder
