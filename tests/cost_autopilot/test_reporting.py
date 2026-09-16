"""Unit tests for format_trace. Built directly on escalation.py's real
types (Attempt, EscalationOutcome, ValidationResult) with a lightweight
fake standing in for orchestrator's AttemptOutcome — format_trace only
duck-types .latency_ms/.cost_usd off attempt.result, so a fake with just
those two fields is enough; no real Coder involved.
"""

from dataclasses import dataclass

from aider.cost_autopilot.escalation import Attempt, EscalationOutcome, ValidationResult
from aider.cost_autopilot.reporting import format_trace


@dataclass
class FakeResult:
    latency_ms: float
    cost_usd: float = 0.0


def make_attempt(model, position, passed, latency_ms=1000.0, cost_usd=0.0):
    return Attempt(
        model=model,
        position=position,
        result=FakeResult(latency_ms=latency_ms, cost_usd=cost_usd),
        validation=ValidationResult(passed=passed),
    )


def test_single_successful_attempt_reports_used_not_escalated():
    outcome = EscalationOutcome(
        attempts=[make_attempt("ollama/qwen2.5:7b", 0, passed=True, latency_ms=1928.0)],
        succeeded=True,
    )

    trace = format_trace(outcome, committed=True)

    assert "ollama/qwen2.5:7b" in trace
    assert "used" in trace
    assert "escalated" not in trace
    assert "committed" in trace
    assert "1.9s" in trace


def test_two_attempts_reports_escalated_to_the_final_model():
    outcome = EscalationOutcome(
        attempts=[
            make_attempt("ollama/llama3.2:1b", 0, passed=False, latency_ms=500.0),
            make_attempt("ollama/qwen2.5:7b", 1, passed=True, latency_ms=2000.0),
        ],
        succeeded=True,
    )

    trace = format_trace(outcome, committed=True)

    assert "escalated to ollama/qwen2.5:7b" in trace
    # both attempts appear in the per-line trace
    assert "ollama/llama3.2:1b" in trace
    assert "ollama/qwen2.5:7b" in trace
    # the failed attempt's line and the success line are distinguishable
    assert "✗" in trace
    assert "✓" in trace


def test_exhausted_ladder_reports_could_not_validate():
    outcome = EscalationOutcome(
        attempts=[
            make_attempt("a", 0, passed=False),
            make_attempt("b", 1, passed=False),
            make_attempt("c", 2, passed=False),
        ],
        succeeded=False,
    )

    trace = format_trace(outcome, committed=False)

    assert "could not validate" in trace
    assert "3 attempt" in trace
    assert "committed" not in trace or "not committed" in trace


def test_zero_cost_reads_as_unpriced_not_as_a_dollar_amount():
    """A literal $0.000000 could be misread as "somehow free API access."
    unpriced/local is honest about why it's zero."""
    outcome = EscalationOutcome(
        attempts=[make_attempt("ollama/llama3.2:1b", 0, passed=True, cost_usd=0.0)],
        succeeded=True,
    )

    trace = format_trace(outcome, committed=True)

    assert "unpriced/local" in trace
    assert "$0.000000" not in trace


def test_known_cost_is_shown_with_real_precision():
    outcome = EscalationOutcome(
        attempts=[make_attempt("anthropic/claude-haiku", 0, passed=True, cost_usd=0.001234)],
        succeeded=True,
    )

    trace = format_trace(outcome, committed=True)

    assert "$0.001234" in trace


def test_total_time_sums_across_all_attempts_not_just_the_final_one():
    outcome = EscalationOutcome(
        attempts=[
            make_attempt("a", 0, passed=False, latency_ms=500.0),
            make_attempt("b", 1, passed=True, latency_ms=2500.0),
        ],
        succeeded=True,
    )

    trace = format_trace(outcome, committed=True)

    # 500 + 2500 = 3000ms = 3.0s total, distinct from either individual attempt
    assert "total 3.0s" in trace
