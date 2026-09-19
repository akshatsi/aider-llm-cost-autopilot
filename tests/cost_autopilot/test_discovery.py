"""Unit tests for discovery.py -- fakes throughout: requests.get and
aider.models.Model are both monkeypatched so these never touch a real
Ollama server, a real network, or real environment variables. Real
discovery against this machine's actual Ollama/API keys is the live
check, not a unit test.
"""

import pytest

from aider.cost_autopilot.discovery import (
    CLOUD_PROVIDER_CANDIDATES,
    _discover_cloud_models,
    _discover_ollama_models,
    discover_ladder,
)


class FakeResponse:
    def __init__(self, payload, status_ok=True):
        self._payload = payload
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("simulated HTTP error")

    def json(self):
        return self._payload


def _ollama_entry(name, parameter_size, capabilities=("completion",)):
    return {
        "model": name,
        "details": {"parameter_size": parameter_size},
        "capabilities": list(capabilities),
    }


# --- _discover_ollama_models ---------------------------------------------


def test_ollama_models_are_sorted_smallest_parameter_count_first(monkeypatch):
    payload = {
        "models": [
            _ollama_entry("qwen2.5:14b", "14.0B"),
            _ollama_entry("llama3.2:1b", "1.2B"),
            _ollama_entry("qwen2.5:7b", "7.6B"),
        ]
    }
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResponse(payload))

    found = _discover_ollama_models()

    assert [name for name, _size in found] == [
        "ollama/llama3.2:1b",
        "ollama/qwen2.5:7b",
        "ollama/qwen2.5:14b",
    ]


def test_ollama_discovery_skips_pure_embedding_models(monkeypatch):
    payload = {
        "models": [
            _ollama_entry("nomic-embed-text:latest", "137M", capabilities=("embedding",)),
            _ollama_entry("llama3.2:1b", "1.2B"),
        ]
    }
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResponse(payload))

    found = _discover_ollama_models()

    assert [name for name, _size in found] == ["ollama/llama3.2:1b"]


def test_ollama_unreachable_returns_empty_list_not_an_exception(monkeypatch):
    def raise_connection_error(*args, **kwargs):
        raise ConnectionError("simulated: nothing listening")

    monkeypatch.setattr("requests.get", raise_connection_error)

    assert _discover_ollama_models() == []


def test_ollama_http_error_returns_empty_list_not_an_exception(monkeypatch):
    monkeypatch.setattr(
        "requests.get", lambda *a, **k: FakeResponse({}, status_ok=False)
    )

    assert _discover_ollama_models() == []


def test_ollama_unparseable_parameter_size_sorts_last_not_crashes(monkeypatch):
    payload = {
        "models": [
            _ollama_entry("weird-model:latest", "unknown"),
            _ollama_entry("llama3.2:1b", "1.2B"),
        ]
    }
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResponse(payload))

    found = [name for name, _size in _discover_ollama_models()]

    assert found == ["ollama/llama3.2:1b", "ollama/weird-model:latest"]


def test_ollama_no_models_pulled_returns_empty_list(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResponse({"models": []}))

    assert _discover_ollama_models() == []


def test_ollama_discovery_uses_a_custom_api_base_argument(monkeypatch):
    seen_urls = []

    def fake_get(url, timeout=None):
        seen_urls.append(url)
        return FakeResponse({"models": []})

    monkeypatch.setattr("requests.get", fake_get)
    _discover_ollama_models(api_base="http://example.test:1234")

    assert seen_urls == ["http://example.test:1234/api/tags"]


# --- _discover_cloud_models ------------------------------------------------


class FakeModel:
    _PRICES = {
        "claude-haiku-4-5": 1e-6,
        "claude-sonnet-5": 2e-6,
        "gpt-4o-mini": 1.5e-7,
        "gpt-4o": 2.5e-6,
    }

    def __init__(self, name):
        self.name = name
        self.info = {"input_cost_per_token": self._PRICES.get(name)}


def test_cloud_discovery_includes_only_providers_with_a_set_api_key(monkeypatch):
    monkeypatch.setattr("aider.models.Model", FakeModel)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")

    found = _discover_cloud_models()

    assert found == CLOUD_PROVIDER_CANDIDATES["ANTHROPIC_API_KEY"]


def test_cloud_discovery_orders_by_real_price_not_dict_declaration_order(monkeypatch):
    """ANTHROPIC_API_KEY is declared before OPENAI_API_KEY in the dict,
    but gpt-4o-mini is cheaper than claude-haiku-4-5 -- the cheaper one
    must come first regardless of provider iteration order."""
    monkeypatch.setattr("aider.models.Model", FakeModel)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    found = _discover_cloud_models()

    assert found.index("gpt-4o-mini") < found.index("claude-haiku-4-5")


def test_cloud_discovery_drops_a_candidate_that_fails_to_resolve(monkeypatch):
    class FlakyModel:
        def __init__(self, name):
            if name == "gpt-4o-mini":
                raise RuntimeError("simulated: unknown model")
            self.info = {"input_cost_per_token": 1e-6}

    monkeypatch.setattr("aider.models.Model", FlakyModel)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    found = _discover_cloud_models()

    assert found == ["gpt-4o"]  # gpt-4o-mini dropped, not fatal


def test_cloud_discovery_returns_empty_list_when_no_keys_are_set(monkeypatch):
    monkeypatch.setattr("aider.models.Model", FakeModel)
    for env_var in CLOUD_PROVIDER_CANDIDATES:
        monkeypatch.delenv(env_var, raising=False)

    assert _discover_cloud_models() == []


# --- discover_ladder: the combined, top-level entry point -----------------


def test_discover_ladder_puts_ollama_models_before_cloud_models(monkeypatch):
    monkeypatch.setattr(
        "aider.cost_autopilot.discovery._discover_ollama_models",
        lambda: [("ollama/llama3.2:1b", 1.2)],
    )
    monkeypatch.setattr(
        "aider.cost_autopilot.discovery._discover_cloud_models",
        lambda: ["claude-haiku-4-5"],
    )

    assert discover_ladder() == ["ollama/llama3.2:1b", "claude-haiku-4-5"]


def test_discover_ladder_raises_a_clear_error_when_nothing_is_found(monkeypatch):
    monkeypatch.setattr(
        "aider.cost_autopilot.discovery._discover_ollama_models", lambda: []
    )
    monkeypatch.setattr(
        "aider.cost_autopilot.discovery._discover_cloud_models", lambda: []
    )

    with pytest.raises(ValueError, match="couldn't find any usable models"):
        discover_ladder()
