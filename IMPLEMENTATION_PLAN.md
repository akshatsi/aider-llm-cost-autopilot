# Implementation Plan — Cost Autopilot for Aider

## What this is

A fork of [Aider](https://github.com/Aider-AI/aider) that adds a routing layer between the prompt and the model: pick the cheapest model likely to succeed, verify the result by running the project's own tests, and escalate to a stronger model only when that verification fails.

Two pieces do the work:
- **[RouteLLM](https://github.com/lm-sys/RouteLLM)'s pretrained `bert` router** makes the opening guess — which model to try first.
- **Deterministic escalation** corrects that guess when it's wrong, using real test results rather than a model's opinion.

The router guesses; the tests decide.

## What Aider already gives us

Worth stating plainly, because it shrinks the work considerably. Aider already has:

| Capability | Where |
|---|---|
| Per-call model override | `Coder.send(messages, model=...)` — `base_coder.py:1783` |
| Runs the project's own test command | `auto_test` + `self.test_cmd` — `base_coder.py:1616` |
| Retry loop with error feedback | `reflected_message` → `run_one`'s while loop — `base_coder.py:932` |
| Dry-run edit validation | `apply_edits_dry_run` — `base_coder.py:2431` |
| Per-call cost tracking | `litellm.completion_cost` — `base_coder.py:2037` |
| Multi-provider model calls | LiteLLM, already a dependency |

What's missing is narrow: Aider's retry re-prompts **the same model** with the error text. We want it to **escalate to a stronger model**, and we want the *first* model chosen by the router rather than always being `main_model`.

## One real constraint, stated up front

The intent was "validate before anything touches the user's files." That's only half achievable, and it's worth being honest about why: **the project's test suite can only test files that are on disk.** A change that hasn't been written can't be tested by `pytest`.

So the design splits validation into two stages:

1. **Dry run** — catch malformed or unappliable edits with zero writes. A cheap model producing garbage gets caught here and escalates without ever touching a file.
2. **Write, then test** — required for real validation. But the **git commit is deferred** until tests pass. A failed attempt is reverted (`git checkout -- <files>`) before the next attempt.

Net effect: nothing is ever *committed* unvalidated, and failed attempts leave no trace in git history — which was the actual concern behind wanting pre-validation.

## The test oracle must be protected — and Aider does not protect it

Found by running Step 1, not by reading code. Worth stating before anything else, because the whole design rests on it.

Given a failing test loop, `llama3.2:1b` edited the **test file** instead of the implementation, replacing every assertion with a `print()`:

```python
def test_reverse_words_basic():
    print("Reversing words...")     # was: assert reverse_words(...) == ...
```

Tests passed. Nothing was implemented. **If the model can edit the tests, test-based validation is worthless** — deleting the assertion is always the cheapest way to make a test pass.

The obvious fix is Aider's `--read` (read-only context). It does not work for this. `allowed_to_edit()` (`base_coder.py:2191`) never consults `abs_read_only_fnames`. Its actual logic:

1. In the editable set → allow
2. Gitignored → skip
3. File doesn't exist → prompt `"Create new file?"`
4. Otherwise → prompt `"Allow edits to file that has not been added to the chat?"` → **on confirm, adds it to the editable set and proceeds**

So `--read` is a soft convention. Under `--yes`, step 4 auto-approves and silently promotes a "read-only" file to editable — exactly what happened.

**Requirement:** our layer owns a protected-path set and hard-rejects edits to it, rather than relying on `--read`. Test files are protected by default. This is an invariant, not a config option — a configurable oracle isn't an oracle.

## Model ladder

Configured as an ordered list, cheapest first. Local defaults (all already pulled):

| Position | Model | Role |
|---|---|---|
| 0 | `ollama/llama3.2:1b` | Opening guess for easy tasks — **see caveat below** |
| 1 | `ollama/qwen2.5:7b` | Opening guess for harder tasks; first escalation target |
| 2 | `ollama/qwen2.5:14b` | Final escalation target |

Escalation walks up this list. The list is configuration — swap in any LiteLLM-supported model, local or hosted.

**Caveat on position 0, measured not assumed:** `llama3.2:1b` failed the Step 1 baseline outright — never implemented the function, and hallucinated a literal `path/to/` directory into existence. The earlier `Cheap_Worth` benchmark had this model passing 14/16 tasks, but those asked for a **standalone function**; Aider requires a valid search/replace edit block against existing file content, which is a materially harder output constraint. That success rate does not transfer. `qwen2.5:7b` handled the same task correctly on the first attempt.

Open question for Step 6: whether position 0 earns its place in an Aider ladder at all, or whether a 1B model is simply below the floor for in-repo editing and the ladder should start at 7B. Don't resolve this by argument — measure escalation rate from position 0 across the sample library.

## Steps

### Step 1 — Baseline: Aider unmodified, against Ollama — ✅ DONE

Confirmed stock Aider drives Ollama correctly, with `--test-cmd` wired to a real suite. No changes to Aider. This isolated "Aider + Ollama works" from "our layer works" — and immediately earned its keep by surfacing three findings before a line of our code existed.

Testbed: `~/Desktop/autopilot-testbed` — a git repo with `stringutils.py` (has `shout()`) and `test_stringutils.py` (tests a `reverse_words()` that doesn't exist yet). Tests fail on a clean checkout; a correct implementation makes 4 tests pass. Reusable for every later step.

**Result:** `qwen2.5:7b` implemented `reverse_words()` correctly on the first attempt, all 4 tests passed, and only `stringutils.py` was modified.

**What Step 1 surfaced** (see the two sections above for detail):
1. The test-oracle corruption failure mode, and that `--read` doesn't prevent it
2. `llama3.2:1b` is likely below the floor for Aider's edit format
3. `--yes` is actively dangerous here — it auto-approves the prompt that promotes a protected file to editable

### Step 2 — Router module (`aider/cost_autopilot/router.py`)
Wrap RouteLLM's `Controller` behind one function: `pick_starting_model(prompt, ladder) -> str`.

- `bert` router only (`routellm/bert_gpt4_augmented`) — the `mf` and `sw_ranking` routers call OpenAI's embeddings API, which breaks the local-only requirement. `mf`'s pretrained weights are dimensionally tied to OpenAI's 1536-dim embeddings and cannot be pointed at a local embedding model.
- RouteLLM's import chain constructs an `OpenAI()` client at module load even for `bert`. Needs a placeholder `OPENAI_API_KEY` env var set — never used for a real call, just satisfies the constructor.
- RouteLLM returns a binary weak/strong decision. Map that onto ladder positions.

Done — `aider/cost_autopilot/router.py`, `pick_starting_model()`. `route_fn` is injectable (mirrors `execute_fn`/`classify_fn` in the original backend): 7 unit tests run against a fake, 0.02s, no network, no torch. Locks in the real invariant too — `ladder[2]` and beyond are never passed to the router at all, confirmed by asserting on the fake's captured call args, not just the ladder's declared shape.

**Caveat, no longer hypothetical — reproduced directly against the real checkpoint:**

```
"Write a function that reverses a string"   → weak model  (llama3.2:1b)
"Write a function that adds two integers"   → strong model (qwen2.5:7b)
```

Both trivial, one-liner tasks. No principled reason one should route to the bigger model and the other shouldn't — this is the calibration mismatch from cloud-trained preference data applied to a ladder it's never seen, now with a concrete repro instead of an abstract warning. The code is verified correct (same prompt through `pick_starting_model()` and through a raw `Controller` call agree, deterministic across repeated runs) — this is the router's actual judgment, not a bug in the wiring around it.

Doesn't block moving forward — it's exactly why escalation exists. But Step 6 should specifically check the *rate* of decisions like this, not just "does the happy path work."

### Step 3 — Escalation module (`aider/cost_autopilot/escalation.py`)
Done — `aider/cost_autopilot/escalation.py`, `run_with_escalation()`. Takes a start *model* (not an index — matches what `pick_starting_model()` returns directly, no translation needed in the caller), walks up the ladder one position at a time on validation failure, stops at the first pass or after `max_attempts` (default 3) models tried, whichever comes first. 10 unit tests, fakes throughout, 0.02s.

One deliberate departure from the original backend worth noting: this is a plain `for` loop, not LangGraph. LangGraph made sense there because escalation was one node among several in a larger orchestration graph; here it's the only control flow this module owns, so a loop is simpler and adds zero dependencies.

Composition-checked against the real router (not just fakes): `pick_starting_model()`'s real output feeds directly into `run_with_escalation()`, a failing first attempt correctly escalates to the next ladder position, succeeds there. The two modules fit together as designed.

### Step 4 — Wire into the Coder
Subclass or wrap `Coder` so that a turn:
1. Calls `pick_starting_model()` on the user's message
2. Passes the result to `send(messages, model=...)`
3. Dry-runs the edits — malformed? escalate, nothing written
4. Applies edits, **holds the commit**
5. Runs `test_cmd` — pass? commit and finish. Fail? revert, escalate, retry
6. Ladder exhausted? surface the last attempt with an explicit "could not validate" note

**Status: done** — `aider/cost_autopilot/orchestrator.py`, `run_autopilot_task()`. A fresh `Coder` per attempt (not chained via `from_coder=` — each model gets an independent shot, no inherited failed-attempt context), `auto_commits=False` and `auto_test=False` on every one of them so Aider's own commit/retry behavior never fires; this module owns the test run and the commit/revert decision entirely, per the design above.

The protected-path guard is instance-level monkeypatching (`_apply_protected_path_guard`), not a subclass — it works regardless of which concrete `Coder` subclass `Coder.create()` picked for the edit format in use, by saving the real bound `allowed_to_edit` and replacing the instance attribute with a closure that checks the protected set first. 3 unit tests against a fake coder confirm: a protected path is rejected without ever reaching the real check; an unprotected path delegates through untouched; an empty protected set blocks nothing.

**Two things found by actually running this against the live testbed that no amount of reading code would have caught:**

1. **`llama3.2:1b` doesn't just fail — it hallucinates files.** Forced to start there (the router itself picks `qwen2.5:7b` for this task, so the failure path needed forcing to actually exercise it), it invented `path/to/filename.js` and a bogus `gitignore` file (not `.gitignore` — a new, separate file) and wrote them to disk. Both fully unrelated to the task.

2. **The original `_revert()` didn't clean them up — a real bug, not a hypothetical.** `git checkout -- <files>` only restores files git already knows about; these were untracked, so checkout silently no-opped and they leaked onto disk. Confirmed by inspection after the first live run: `gitignore` and `path/` were still sitting in the testbed after "revert."

   Fixed: `_revert()` now classifies each edited path — tracked (`git ls-files --error-unmatch` succeeds) gets `git checkout --`'d back to its committed state; untracked gets deleted outright, along with any now-empty parent directories it created. Re-ran the identical forced-escalation scenario after the fix: same result (`llama3.2:1b` fails → `qwen2.5:7b` succeeds → commits), but this time `git status` is clean and no stray files remain. 5 more unit tests lock this in against a real scratch git repo (tracked-file revert, untracked-file deletion, empty-dir cleanup, a mix of both in one call, and the no-op case) — not fakes, since the tracked/untracked distinction is exactly the kind of thing a mock would paper over.

Live-verified end to end, twice: once on the happy path (router picks `qwen2.5:7b` directly, succeeds first try, commits, `4 passed` in the testbed), once on the forced-escalation path (`llama3.2:1b` fails and is fully cleaned up → `qwen2.5:7b` succeeds → commits). 25 tests total across the package, all passing.

Prefer wrapping over editing `base_coder.py` in place — a smaller diff against upstream is easier to keep in sync. The instance-level `allowed_to_edit` patch turned out to be the right call over a subclass: no need to know or care which concrete `Coder` subclass got constructed.

### Step 5 — Reporting — done

`aider/cost_autopilot/reporting.py`, `format_trace()` + `print_trace()`. Prints inline after every turn via the same `io.tool_output()` Aider itself uses for status messages — no dashboard, no second place to look.

Cost comes for free, not from anything we built: Aider's own `calculate_and_show_tokens_and_cost()` already skips cost computation entirely when LiteLLM has no known price for a model (`if not self.main_model.info.get("input_cost_per_token"): return`) — true for every local Ollama model. So `coder.total_cost` reads a genuine `0` for local models, not a placeholder standing in for "unknown." Read directly off a fresh per-attempt `Coder` instance in `orchestrator.py`'s `attempt_fn`, alongside wall-clock latency measured around `run_one()`.

One deliberate wording choice: zero cost renders as `$0 (unpriced/local)`, not a bare `$0.000000`. A literal dollar amount of zero reads as "somehow got free API access"; this says why it's zero instead. Locked in with a test specifically asserting the bare form never appears.

`format_trace()` depends only on `escalation.py`'s generic `EscalationOutcome` type, not on `orchestrator.py` — kept the dependency one-directional (orchestrator → reporting → escalation) to avoid a circular import, since orchestrator needs to call `print_trace()` at the end of a run.

6 unit tests against fake attempt data (single success, escalation, exhausted ladder, the unpriced-wording assertion, a real-cost case for when the ladder eventually holds a priced model, and that total time sums across every attempt, not just the last one).

Live-verified against the same forced-escalation scenario as Step 4, output exactly as designed:

```
cost_autopilot:
  1. ollama/llama3.2:1b           ✗    31.8s  $0 (unpriced/local)
  2. ollama/qwen2.5:7b            ✓    10.9s  $0 (unpriced/local)
  -> escalated to ollama/qwen2.5:7b (committed), total 42.7s, $0 (unpriced/local)
```

31 tests total across the package, all passing.

### Step 6 — End-to-end testing
Reuse the 16-task sample library from the previous build (`tests/fixtures/sample_tasks/` in the `Cheap_Worth` repo) — already validated, already proven against Ollama. Then one real-project scenario: a scratch git repo with an actual test suite, exercising real-repo-mode validation, which no test has covered yet.

Watch for: does routing actually vary across tasks, does escalation fire when it should, is git history clean after failures, do the reported numbers match reality.

## Cost

Zero. Ollama for generation, the `bert` router runs locally on CPU via PyTorch, and no step requires a funded API key. The placeholder `OPENAI_API_KEY` is never used for a request.

## Prior art

Not novel, and worth knowing what we're differentiating against:
- **FrugalGPT** (Chen, Zaharia, Zou — Stanford) is the foundational cascade paper; reports 50–98% cost reduction.
- **[opencode-model-router](https://github.com/marco-jardim/opencode-model-router)** does tier delegation for OpenCode via prompt-based complexity self-assessment, reporting ~36% cost reduction.

The differentiator here is the verification signal: escalation triggers on **tests actually failing**, not on a model's self-assessment of difficulty. Whether that produces better results than prompt-based delegation is an empirical question Step 6 should start answering, not a claim to assume.
