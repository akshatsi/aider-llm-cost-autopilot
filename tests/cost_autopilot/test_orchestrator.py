"""Tests for orchestrator.py's own logic — the protected-path guard and
the revert-classification behavior. Deliberately not driving a real
Coder or a real model here (that's the live testbed check, not a unit
test); these test the two things orchestrator.py itself is responsible
for, in isolation.
"""

import os
import subprocess

import pytest

from aider.cost_autopilot.orchestrator import (
    LOCAL_MODEL_MAX_TOKENS,
    LOCAL_MODEL_TIMEOUT_S,
    _apply_protected_path_guard,
    _disable_reflections,
    _model_for,
    _revert,
)


class FakeIO:
    def __init__(self):
        self.errors = []

    def tool_error(self, msg):
        self.errors.append(msg)


class FakeCoder:
    """Minimal stand-in for a real Coder — just enough surface for
    _apply_protected_path_guard to patch and call."""

    def __init__(self):
        self.io = FakeIO()
        self.allow_calls = []

    def allowed_to_edit(self, path):
        self.allow_calls.append(path)
        return True  # the "real" Aider behavior this guard wraps


def test_protected_path_is_rejected_without_reaching_the_real_check():
    coder = FakeCoder()
    _apply_protected_path_guard(coder, protected_paths={"/repo/test_thing.py"})

    result = coder.allowed_to_edit("/repo/test_thing.py")

    assert result is False
    assert coder.allow_calls == []  # never delegated to the real allowed_to_edit
    assert len(coder.io.errors) == 1
    assert "test_thing.py" in coder.io.errors[0]


def test_unprotected_path_delegates_to_the_real_check():
    coder = FakeCoder()
    _apply_protected_path_guard(coder, protected_paths={"/repo/test_thing.py"})

    result = coder.allowed_to_edit("/repo/main.py")

    assert result is True
    assert coder.allow_calls == ["/repo/main.py"]
    assert coder.io.errors == []


def test_empty_protected_set_never_blocks_anything():
    coder = FakeCoder()
    _apply_protected_path_guard(coder, protected_paths=set())

    assert coder.allowed_to_edit("/repo/anything.py") is True
    assert coder.allow_calls == ["/repo/anything.py"]


# --- _revert: tracked vs. untracked/hallucinated files -----------------


@pytest.fixture
def scratch_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)

    tracked = repo / "tracked.py"
    tracked.write_text("original content\n")
    subprocess.run(["git", "add", "tracked.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo, check=True)

    return repo


def test_revert_restores_a_tracked_modified_file(scratch_repo):
    tracked = scratch_repo / "tracked.py"
    tracked.write_text("a bad model's edit\n")

    _revert({str(tracked)}, cwd=str(scratch_repo))

    assert tracked.read_text() == "original content\n"


def test_revert_deletes_a_hallucinated_untracked_file(scratch_repo):
    """The real finding from testing: a failed attempt can invent files
    that never existed. git checkout -- silently does nothing for these
    (they're untracked) — revert has to delete them outright."""
    hallucinated = scratch_repo / "path" / "to" / "filename.js"
    hallucinated.parent.mkdir(parents=True)
    hallucinated.write_text("nonsense\n")

    _revert({str(hallucinated)}, cwd=str(scratch_repo))

    assert not hallucinated.exists()


def test_revert_removes_empty_parent_dirs_left_behind_by_a_deleted_hallucination(scratch_repo):
    hallucinated = scratch_repo / "path" / "to" / "filename.js"
    hallucinated.parent.mkdir(parents=True)
    hallucinated.write_text("nonsense\n")

    _revert({str(hallucinated)}, cwd=str(scratch_repo))

    assert not (scratch_repo / "path").exists()


def test_revert_handles_a_mix_of_tracked_and_hallucinated_files_together(scratch_repo):
    tracked = scratch_repo / "tracked.py"
    tracked.write_text("a bad model's edit\n")

    hallucinated = scratch_repo / "gitignore"  # not .gitignore -- a real repro from testing
    hallucinated.write_text("bogus\n")

    _revert({str(tracked), str(hallucinated)}, cwd=str(scratch_repo))

    assert tracked.read_text() == "original content\n"
    assert not hallucinated.exists()


def test_revert_with_no_edited_files_is_a_safe_no_op(scratch_repo):
    _revert(set(), cwd=str(scratch_repo))  # must not raise


# --- _model_for: bounded timeout/tokens for local Ollama models --------


def test_ollama_model_gets_bounded_timeout_and_retries_and_max_tokens():
    """Real finding from the Step 6 batch run: with Aider's 600s default
    and litellm's own retries on top, one stuck local generation burned
    over 80 minutes. These bounds make it fail fast instead."""
    model = _model_for("ollama/llama3.2:1b")

    assert model.extra_params["timeout"] == LOCAL_MODEL_TIMEOUT_S
    assert model.extra_params["num_retries"] == 0
    assert model.extra_params["max_tokens"] == LOCAL_MODEL_MAX_TOKENS


def test_non_ollama_model_is_left_alone():
    model = _model_for("gpt-4o-mini")

    assert not model.extra_params or "timeout" not in model.extra_params


# --- _disable_reflections: no same-model retry storms -------------------


def test_disable_reflections_zeroes_out_aiders_own_retry_loop():
    """Real finding from re-running the Step 6 batch: even with the
    per-call timeout/max_tokens bounds, Aider's own reflection loop
    (default max_reflections=3) re-prompted a stuck model up to 4 times
    per attempt, each a fresh max-token generation of the same
    degenerate output. This is what actually made one attempt take
    several minutes instead of one call's worth of time."""
    coder = FakeCoder()
    coder.max_reflections = 3  # Aider's real class default

    _disable_reflections(coder)

    assert coder.max_reflections == 0
