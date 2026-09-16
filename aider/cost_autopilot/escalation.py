"""Ladder-walking escalation: try a model, validate the result, and if it
fails, move to the next model up the ladder — one step at a time, same
principle as the original Python backend's cascade (see
IMPLEMENTATION_PLAN.md).

Deliberately a plain loop, not a graph library. The original backend used
LangGraph for this same logic, which made sense there — it was one node
among several in a larger orchestration graph. Here it's the only control
flow this module owns; a for-loop is simpler, adds zero dependencies, and
is exactly as testable.

Pure and injectable: no Aider imports, no network, no opinion about what
"model" or "validate" actually mean beyond the types below. Step 4 wires
this to Aider's real send()/apply_edits()/test_cmd; the tests here wire
it to fakes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_MAX_ATTEMPTS = 3


@dataclass
class ValidationResult:
    passed: bool
    detail: Any = None


@dataclass
class Attempt:
    model: str
    position: int
    result: Any
    validation: ValidationResult


@dataclass
class EscalationOutcome:
    attempts: list[Attempt] = field(default_factory=list)
    succeeded: bool = False

    @property
    def final_attempt(self) -> Attempt:
        if not self.attempts:
            raise ValueError("no attempts were made")
        return self.attempts[-1]


class AttemptFn(Protocol):
    def __call__(self, model: str) -> Any: ...


class ValidateFn(Protocol):
    def __call__(self, result: Any) -> ValidationResult: ...


def run_with_escalation(
    ladder: list[str],
    start_model: str,
    attempt_fn: AttemptFn,
    validate_fn: ValidateFn,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> EscalationOutcome:
    """Try start_model; if validation fails, move one position up the
    ladder and try again — up to max_attempts models total, or the top
    of the ladder, whichever comes first. Stops at the first model whose
    result validates.

    start_model is a model string, not an index, because that's what
    router.pick_starting_model() returns — callers shouldn't need to
    translate between the two.
    """
    if start_model not in ladder:
        raise ValueError(f"{start_model!r} is not in the ladder {ladder!r}")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    position = ladder.index(start_model)
    attempts: list[Attempt] = []

    for _ in range(max_attempts):
        if position >= len(ladder):
            break

        model = ladder[position]
        result = attempt_fn(model)
        validation = validate_fn(result)
        attempts.append(
            Attempt(model=model, position=position, result=result, validation=validation)
        )

        if validation.passed:
            return EscalationOutcome(attempts=attempts, succeeded=True)

        position += 1

    return EscalationOutcome(attempts=attempts, succeeded=False)
