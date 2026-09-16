"""Formats an escalation outcome into a terminal-friendly trace: which
model ran, whether it escalated, how long each attempt took, and dollar
cost where it's actually known. No dashboard — this prints inline, in
the same terminal session the task was run from, consistent with living
inside a terminal-first tool (see IMPLEMENTATION_PLAN.md's Step 5).

Depends only on escalation.py's generic EscalationOutcome, not on
orchestrator.py — orchestrator calls into this module, not the other
way around, so there's no import cycle. Takes attempt.result duck-typed
(expects .latency_ms and .cost_usd) rather than importing orchestrator's
AttemptOutcome type specifically, for the same reason.
"""

from __future__ import annotations

from aider.cost_autopilot.escalation import EscalationOutcome


def _format_cost(cost_usd: float) -> str:
    if not cost_usd:
        return "$0 (unpriced/local)"
    return f"${cost_usd:.6f}"


def _format_latency(latency_ms: float) -> str:
    if latency_ms >= 1000:
        return f"{latency_ms / 1000:.1f}s"
    return f"{latency_ms:.0f}ms"


def format_trace(outcome: EscalationOutcome, committed: bool) -> str:
    lines = ["cost_autopilot:"]

    for i, attempt in enumerate(outcome.attempts, start=1):
        icon = "✓" if attempt.validation.passed else "✗"
        cost = _format_cost(getattr(attempt.result, "cost_usd", 0.0))
        latency = _format_latency(getattr(attempt.result, "latency_ms", 0.0))
        lines.append(f"  {i}. {attempt.model:<28} {icon}  {latency:>7}  {cost}")

    escalated = len(outcome.attempts) > 1
    total_time_ms = sum(getattr(a.result, "latency_ms", 0.0) for a in outcome.attempts)
    total_cost = sum(getattr(a.result, "cost_usd", 0.0) for a in outcome.attempts)

    if outcome.succeeded:
        verb = "escalated to" if escalated else "used"
        status = "committed" if committed else "validated but not committed"
        lines.append(
            f"  -> {verb} {outcome.final_attempt.model} ({status}), "
            f"total {_format_latency(total_time_ms)}, {_format_cost(total_cost)}"
        )
    else:
        lines.append(
            f"  -> could not validate after {len(outcome.attempts)} attempt(s), "
            f"total {_format_latency(total_time_ms)}, {_format_cost(total_cost)}"
        )

    return "\n".join(lines)


def print_trace(outcome: EscalationOutcome, committed: bool, io) -> None:
    io.tool_output(format_trace(outcome, committed))
