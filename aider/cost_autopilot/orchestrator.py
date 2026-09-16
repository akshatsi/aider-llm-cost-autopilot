"""Wires router.py + escalation.py into real Aider Coder instances.

One Coder is constructed fresh per attempt — not chained via
Coder.create(from_coder=...) — so each model gets an independent shot at
the task rather than inheriting a previous attempt's failed conversation
as context.

Every Coder built here has auto_commits=False and auto_test=False: Aider's
own commit-on-edit and test-then-confirm-retry behavior are both turned
off. This module owns the test run and the commit/revert decision
instead — see IMPLEMENTATION_PLAN.md's Step 4 and the test-oracle-
corruption finding from Step 1 for why.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any

from aider.coders import Coder

from aider.cost_autopilot.escalation import (
    DEFAULT_MAX_ATTEMPTS,
    EscalationOutcome,
    ValidationResult,
    run_with_escalation,
)
from aider.cost_autopilot.router import pick_starting_model


def _apply_protected_path_guard(coder, protected_paths: set[str]) -> None:
    """Instance-level override, not a subclass — works regardless of
    which concrete Coder subclass Coder.create() picked for this
    edit_format. Hard-rejects a protected path before Aider's own
    allowed_to_edit() ever runs, so there's no confirm prompt for --yes
    to auto-approve (see IMPLEMENTATION_PLAN.md, Step 1: --read alone
    does not prevent this)."""
    original_allowed_to_edit = coder.allowed_to_edit

    def guarded(path):
        if path in protected_paths:
            coder.io.tool_error(f"cost_autopilot: refusing to edit protected file: {path}")
            return False
        return original_allowed_to_edit(path)

    coder.allowed_to_edit = guarded


def _make_coder(model: str, io, fnames: list[str], protected_paths: set[str]):
    coder = Coder.create(
        main_model=_model_for(model),
        io=io,
        fnames=fnames,
        auto_commits=False,
        auto_test=False,
        suggest_shell_commands=False,
    )
    _apply_protected_path_guard(coder, protected_paths)
    return coder


def _model_for(model_name: str):
    from aider.models import Model

    return Model(model_name)


@dataclass
class AttemptOutcome:
    """What one Coder attempt produced — passed to validate_fn."""

    coder: Any
    edited_files: set

    @property
    def wrote_anything(self) -> bool:
        return bool(self.edited_files)


def _run_test_cmd(test_cmd: str, cwd: str) -> tuple[bool, str]:
    result = subprocess.run(
        test_cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    passed = result.returncode == 0
    detail = (result.stdout or "") + (result.stderr or "")
    return passed, detail[-4000:]


def _revert(edited_files: set, cwd: str) -> None:
    """Undo a failed attempt's edits, completely.

    git checkout -- only restores files git already knows about. A model
    can also hallucinate entirely new files (seen for real in testing:
    llama3.2:1b invented path/to/filename.js and a bogus gitignore file
    on a failed attempt) — those are untracked, so checkout silently does
    nothing for them and they leak onto disk. Each edited path is
    classified first: tracked -> checkout restores it; untracked -> it
    didn't exist before this attempt, so delete it outright, along with
    any now-empty parent directories it created.
    """
    if not edited_files:
        return

    tracked = []
    for f in edited_files:
        rel = os.path.relpath(f, cwd) if os.path.isabs(f) else f
        abs_path = f if os.path.isabs(f) else os.path.join(cwd, f)

        is_tracked = (
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", rel],
                cwd=cwd,
                capture_output=True,
            ).returncode
            == 0
        )

        if is_tracked:
            tracked.append(rel)
        else:
            try:
                os.remove(abs_path)
            except FileNotFoundError:
                pass
            parent = os.path.dirname(abs_path)
            while parent and parent != cwd and os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
                parent = os.path.dirname(parent)

    if tracked:
        subprocess.run(
            ["git", "checkout", "--"] + tracked,
            cwd=cwd,
            capture_output=True,
            text=True,
        )


@dataclass
class AutopilotResult:
    outcome: EscalationOutcome
    committed: bool


def run_autopilot_task(
    user_message: str,
    fnames: list[str],
    ladder: list[str],
    test_cmd: str,
    repo_root: str,
    io,
    protected_paths: set[str] | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> AutopilotResult:
    """Run one task through the router + escalation ladder against real
    Aider Coder instances. Nothing is committed until a validated attempt
    exists; a failed attempt's edits are reverted before the next one
    starts, so every attempt sees a clean working tree."""
    protected_paths = protected_paths or set()

    def attempt_fn(model: str) -> AttemptOutcome:
        coder = _make_coder(model, io, fnames, protected_paths)
        coder.run_one(user_message, preproc=True)
        return AttemptOutcome(coder=coder, edited_files=set(coder.aider_edited_files))

    def validate_fn(attempt: AttemptOutcome) -> ValidationResult:
        if not attempt.wrote_anything:
            return ValidationResult(passed=False, detail={"reason": "no files were edited"})

        passed, test_output = _run_test_cmd(test_cmd, repo_root)
        if not passed:
            _revert(attempt.edited_files, repo_root)
        return ValidationResult(
            passed=passed, detail={"edited_files": sorted(attempt.edited_files), "test_output": test_output}
        )

    start_model = pick_starting_model(user_message, ladder)
    outcome = run_with_escalation(
        ladder, start_model, attempt_fn, validate_fn, max_attempts=max_attempts
    )

    committed = False
    if outcome.succeeded:
        final = outcome.final_attempt
        coder = final.result.coder
        coder.auto_commits = True
        coder.auto_commit(final.result.edited_files, context=f"cost_autopilot: {final.model}")
        committed = True

    return AutopilotResult(outcome=outcome, committed=committed)
