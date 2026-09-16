"""Unit tests for pick_starting_model's decision logic. route_fn is
always faked here — no network access, no real checkpoint download, no
torch inference. The real route_with_bert function is exercised
separately by a manual/integration check, not by this suite (mirrors how
execute_fn/classify_fn were tested in the original Python backend: unit
tests cover control flow with fakes, real provider calls are verified
live, once, outside the automated suite).
"""

import pytest

from aider.cost_autopilot.router import DEFAULT_THRESHOLD, pick_starting_model

LADDER = ["model-weak", "model-strong", "model-biggest"]


def make_fake_route_fn(decision):
    calls = []

    def route_fn(prompt, weak_model, strong_model, threshold):
        calls.append(
            {
                "prompt": prompt,
                "weak_model": weak_model,
                "strong_model": strong_model,
                "threshold": threshold,
            }
        )
        return decision

    route_fn.calls = calls
    return route_fn


def test_single_model_ladder_returns_it_without_calling_router():
    def route_fn(*args, **kwargs):
        raise AssertionError("route_fn should never be called for a single-model ladder")

    result = pick_starting_model("any prompt", ["only-model"], route_fn=route_fn)

    assert result == "only-model"


def test_empty_ladder_raises():
    with pytest.raises(ValueError):
        pick_starting_model("any prompt", [], route_fn=make_fake_route_fn("x"))


def test_easy_decision_returns_the_weak_model():
    route_fn = make_fake_route_fn(LADDER[0])

    result = pick_starting_model("write a function that adds two numbers", LADDER, route_fn=route_fn)

    assert result == LADDER[0]


def test_hard_decision_returns_the_strong_model():
    route_fn = make_fake_route_fn(LADDER[1])

    result = pick_starting_model(
        "implement regex matching with . and * using dynamic programming", LADDER, route_fn=route_fn
    )

    assert result == LADDER[1]


def test_router_is_never_asked_to_choose_the_third_position_directly():
    """Locks in the design invariant: anything past position 1 is an
    escalation target only, never a direct starting guess. route_fn only
    ever sees ladder[0] and ladder[1] as the weak/strong pair, regardless
    of how many models the ladder actually has."""
    route_fn = make_fake_route_fn(LADDER[1])

    pick_starting_model("some prompt", LADDER, route_fn=route_fn)

    assert len(route_fn.calls) == 1
    call = route_fn.calls[0]
    assert call["weak_model"] == LADDER[0]
    assert call["strong_model"] == LADDER[1]
    assert LADDER[2] not in (call["weak_model"], call["strong_model"])


def test_route_fn_receives_the_prompt_and_default_threshold():
    route_fn = make_fake_route_fn(LADDER[0])

    pick_starting_model("a specific prompt text", LADDER, route_fn=route_fn)

    call = route_fn.calls[0]
    assert call["prompt"] == "a specific prompt text"
    assert call["threshold"] == DEFAULT_THRESHOLD


def test_custom_threshold_is_passed_through():
    route_fn = make_fake_route_fn(LADDER[0])

    pick_starting_model("a prompt", LADDER, route_fn=route_fn, threshold=0.8)

    assert route_fn.calls[0]["threshold"] == 0.8
