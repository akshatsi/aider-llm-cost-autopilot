"""Builds the AUTO ladder from whatever the user can actually use right
now, instead of a hardcoded default. Two sources, combined cheapest
first:

1. Ollama -- whatever is currently pulled in a running local instance,
   read live from its own API. Genuinely dynamic: no model name is
   ever hardcoded here, so it reflects reality even after models are
   added or removed.
2. Cloud providers -- there's no equivalent "list what I have access
   to" call for a paid API; the only real signal is whether that
   provider's API key env var is set. For each one that is, a small
   curated set of that provider's current mainstream chat models is
   added. This needs occasional upkeep as model lineups change --
   traded off deliberately against enumerating LiteLLM's full model
   table, which mixes in deprecated, niche, and non-chat entries with
   no clean way to filter them back out.

Ollama models sort first as a block (all genuinely $0, cheapest by
definition, ordered smallest-parameter-count first as a capability
proxy); cloud models follow, ordered by real LiteLLM pricing so a
cheap model from one provider doesn't lose to a pricier one from
another just because of dict order.
"""

from __future__ import annotations

import os

OLLAMA_API_BASE_DEFAULT = "http://localhost:11434"

# Verified against the installed LiteLLM's own pricing metadata before
# being added here -- see IMPLEMENTATION_PLAN.md for how and when.
CLOUD_PROVIDER_CANDIDATES: dict[str, list[str]] = {
    "ANTHROPIC_API_KEY": ["claude-haiku-4-5", "claude-sonnet-5"],
    "OPENAI_API_KEY": ["gpt-4o-mini", "gpt-4o"],
    "GEMINI_API_KEY": ["gemini/gemini-2.5-flash", "gemini/gemini-2.5-pro"],
    "GROQ_API_KEY": ["groq/llama-3.3-70b-versatile"],
}


def _discover_ollama_models(api_base: str | None = None) -> list[tuple[str, float]]:
    """[(model_name, parameter_size_in_billions), ...] for every
    chat-capable model currently pulled, smallest first. Empty list,
    never an error, if Ollama isn't reachable -- "not running" is a
    normal, expected state here, not a failure."""
    import requests

    base = api_base or os.environ.get("OLLAMA_API_BASE") or OLLAMA_API_BASE_DEFAULT
    try:
        resp = requests.get(f"{base}/api/tags", timeout=2)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return []

    found = []
    for entry in payload.get("models", []):
        if "completion" not in entry.get("capabilities", []):
            continue  # skip pure embedding models -- not usable for coding edits
        name = entry.get("model")
        if not name:
            continue
        param_size_str = entry.get("details", {}).get("parameter_size", "")
        try:
            size = float(param_size_str.rstrip("Bb"))
        except ValueError:
            size = float("inf")  # unknown size sorts last, not first
        found.append((f"ollama/{name}", size))

    found.sort(key=lambda pair: pair[1])
    return found


def _discover_cloud_models() -> list[str]:
    """Every candidate model for a provider whose API key env var is
    set, ordered by real price ascending (not dict-declaration order,
    so a cheap model from one provider doesn't lose to a pricier one
    from another just because of iteration order)."""
    from aider.models import Model

    candidates = []
    for env_var, model_names in CLOUD_PROVIDER_CANDIDATES.items():
        if os.environ.get(env_var):
            candidates.extend(model_names)

    priced = []
    for name in candidates:
        try:
            cost = Model(name).info.get("input_cost_per_token") or 0
        except Exception:
            continue  # a candidate that fails to resolve is dropped, not fatal
        priced.append((name, cost))

    priced.sort(key=lambda pair: pair[1])
    return [name for name, _cost in priced]


def discover_ladder() -> list[str]:
    """The ladder AUTO uses when none is given explicitly: every
    Ollama model actually pulled right now, then every cloud model
    from a provider whose key is set. Raises ValueError, with a
    concrete next step, when nothing is found at all -- silently
    routing nowhere would be worse than a clear error."""
    ollama_models = [name for name, _size in _discover_ollama_models()]
    cloud_models = _discover_cloud_models()
    ladder = ollama_models + cloud_models

    if not ladder:
        known_keys = ", ".join(CLOUD_PROVIDER_CANDIDATES)
        raise ValueError(
            "AUTO couldn't find any usable models: no local Ollama models are "
            "pulled and none of these API keys are set: "
            f"{known_keys}. Pull an Ollama model (ollama pull <name>), set one "
            "of those keys, or pass --auto-ladder explicitly."
        )
    return ladder
