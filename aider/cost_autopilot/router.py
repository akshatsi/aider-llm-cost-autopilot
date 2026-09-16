"""Wraps RouteLLM's pretrained bert router: given a prompt and a ladder of
models (cheapest first), decide where to start.

Only ever returns ladder[0] or ladder[1] — RouteLLM makes a single binary
weak/strong call between exactly two models. Anything further up the
ladder is reached only through escalation (see escalation.py and
IMPLEMENTATION_PLAN.md), never a direct starting guess — same principle
carried over from the original Python backend's classifier design, where
the top tier was an escalation target only, never a direct prediction.

Only the bert router is used. mf and sw_ranking call OpenAI's embeddings
API per request; mf's pretrained weights are also dimensionally tied to
OpenAI's 1536-dim embedding space and cannot be redirected to a local
embedding model. bert runs a standard HuggingFace sequence-classification
checkpoint entirely on-device — downloaded once, cached locally by
huggingface_hub afterward, no per-call network access.
"""

from __future__ import annotations

import os
from typing import Protocol

# RouteLLM's import chain constructs an OpenAI() client at module load
# time, for a different router type this module never uses. The
# constructor just needs *a* value present to not raise — the bert
# router path never sends it a real request.
os.environ.setdefault("OPENAI_API_KEY", "unused-placeholder-not-a-real-key")

BERT_ROUTER_CHECKPOINT = "routellm/bert_gpt4_augmented"
DEFAULT_THRESHOLD = 0.5

_controller_cache: dict[tuple[str, str], object] = {}


def _get_controller(weak_model: str, strong_model: str):
    """One Controller per (weak, strong) pair, cached — constructing it
    loads real model weights, not something to repeat per call."""
    key = (weak_model, strong_model)
    if key not in _controller_cache:
        from routellm.controller import Controller

        _controller_cache[key] = Controller(
            routers=["bert"],
            strong_model=strong_model,
            weak_model=weak_model,
            config={"bert": {"checkpoint_path": BERT_ROUTER_CHECKPOINT}},
        )
    return _controller_cache[key]


def route_with_bert(
    prompt: str, weak_model: str, strong_model: str, threshold: float = DEFAULT_THRESHOLD
) -> str:
    """The real router call — local PyTorch inference against the
    pretrained bert checkpoint. Returns weak_model or strong_model."""
    controller = _get_controller(weak_model, strong_model)
    return controller.route(prompt=prompt, router="bert", threshold=threshold)


class RouteFn(Protocol):
    def __call__(
        self, prompt: str, weak_model: str, strong_model: str, threshold: float
    ) -> str: ...


def pick_starting_model(
    prompt: str,
    ladder: list[str],
    route_fn: RouteFn = route_with_bert,
    threshold: float = DEFAULT_THRESHOLD,
) -> str:
    """Decide which ladder position to start at for this prompt.

    route_fn is injectable specifically so callers (tests, mainly) can
    supply a fake decision without downloading a real checkpoint or
    running real inference — the same pattern classify_fn/execute_fn
    used in the original Python backend.

    A single-model ladder is returned outright, no router call — nothing
    to decide between.
    """
    if not ladder:
        raise ValueError("ladder must have at least one model")
    if len(ladder) == 1:
        return ladder[0]

    return route_fn(prompt, ladder[0], ladder[1], threshold)
