"""Unit tests for run_with_escalation's control flow. attempt_fn and
validate_fn are always faked — this module has no idea what a "model
call" or a "test run" actually is, and the tests reflect that.
"""

import pytest

from aider.cost_autopilot.escalation import (
    DEFAULT_MAX_ATTEMPTS,
    ValidationResult,
    run_with_escalation,
)

LADDER = ["weak", "mid", "strong"]


def make_attempt_fn():
    calls = []

    def attempt_fn(model):
        calls.append(model)
        return f"result-from-{model}"

    attempt_fn.calls = calls
    return attempt_fn


def make_validate_fn(passing_results):
    """passing_results: a set of result strings that should validate."""
    calls = []

    def validate_fn(result):
        calls.append(result)
        return ValidationResult(passed=result in passing_results, detail={"result": result})

    validate_fn.calls = calls
    return validate_fn


def test_first_attempt_succeeds_no_escalation():
    attempt_fn = make_attempt_fn()
    validate_fn = make_validate_fn(passing_results={"result-from-weak"})

    outcome = run_with_escalation(LADDER, "weak", attempt_fn, validate_fn)

    assert outcome.succeeded is True
    assert len(outcome.attempts) == 1
    assert outcome.attempts[0].model == "weak"
    assert outcome.attempts[0].position == 0
    assert attempt_fn.calls == ["weak"]


def test_escalates_one_step_on_failure():
    attempt_fn = make_attempt_fn()
    validate_fn = make_validate_fn(passing_results={"result-from-mid"})

    outcome = run_with_escalation(LADDER, "weak", attempt_fn, validate_fn)

    assert outcome.succeeded is True
    assert [a.model for a in outcome.attempts] == ["weak", "mid"]
    assert outcome.attempts[0].validation.passed is False
    assert outcome.attempts[1].validation.passed is True
    assert outcome.final_attempt.model == "mid"


def test_starts_from_a_middle_position_when_told_to():
    """The router can hand back ladder[1] directly — escalation must
    never fall back to ladder[0] in that case."""
    attempt_fn = make_attempt_fn()
    validate_fn = make_validate_fn(passing_results={"result-from-mid"})

    outcome = run_with_escalation(LADDER, "mid", attempt_fn, validate_fn)

    assert outcome.succeeded is True
    assert attempt_fn.calls == ["mid"]
    assert "weak" not in attempt_fn.calls


def test_exhausts_ladder_and_reports_failure():
    attempt_fn = make_attempt_fn()
    validate_fn = make_validate_fn(passing_results=set())  # nothing ever passes

    outcome = run_with_escalation(LADDER, "weak", attempt_fn, validate_fn, max_attempts=10)

    assert outcome.succeeded is False
    assert [a.model for a in outcome.attempts] == ["weak", "mid", "strong"]
    assert all(not a.validation.passed for a in outcome.attempts)
    assert outcome.final_attempt.model == "strong"


def test_max_attempts_caps_the_walk_even_if_ladder_is_longer():
    long_ladder = ["a", "b", "c", "d", "e"]
    attempt_fn = make_attempt_fn()
    validate_fn = make_validate_fn(passing_results=set())

    outcome = run_with_escalation(long_ladder, "a", attempt_fn, validate_fn, max_attempts=2)

    assert outcome.succeeded is False
    assert attempt_fn.calls == ["a", "b"]
    assert "c" not in attempt_fn.calls


def test_default_max_attempts_is_three():
    assert DEFAULT_MAX_ATTEMPTS == 3


def test_start_model_not_in_ladder_raises():
    with pytest.raises(ValueError):
        run_with_escalation(LADDER, "nonexistent", make_attempt_fn(), make_validate_fn(set()))


def test_max_attempts_below_one_raises():
    with pytest.raises(ValueError):
        run_with_escalation(
            LADDER, "weak", make_attempt_fn(), make_validate_fn(set()), max_attempts=0
        )


def test_validate_fn_receives_the_attempt_result():
    attempt_fn = make_attempt_fn()
    validate_fn = make_validate_fn(passing_results={"result-from-weak"})

    run_with_escalation(LADDER, "weak", attempt_fn, validate_fn)

    assert validate_fn.calls == ["result-from-weak"]


def test_attempts_recorded_in_order_with_correct_positions():
    attempt_fn = make_attempt_fn()
    validate_fn = make_validate_fn(passing_results={"result-from-strong"})

    outcome = run_with_escalation(LADDER, "weak", attempt_fn, validate_fn, max_attempts=5)

    assert [a.position for a in outcome.attempts] == [0, 1, 2]
    assert [a.model for a in outcome.attempts] == ["weak", "mid", "strong"]
