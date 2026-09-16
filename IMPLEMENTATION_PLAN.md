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

## Model ladder

Configured as an ordered list, cheapest first. Local defaults (all already pulled):

| Position | Model | Role |
|---|---|---|
| 0 | `ollama/llama3.2:1b` | Opening guess for tasks the router judges easy |
| 1 | `ollama/qwen2.5:7b` | Opening guess for harder tasks; first escalation target |
| 2 | `ollama/qwen2.5:14b` | Final escalation target |

Escalation walks up this list. The list is configuration — swap in any LiteLLM-supported model, local or hosted.

## Steps

### Step 1 — Baseline: Aider unmodified, against Ollama
Confirm stock Aider edits a file correctly using `ollama/llama3.2:1b`, with `--test-cmd` wired to a real test suite. **No changes to Aider yet.** This isolates "Aider + Ollama works" from "our layer works" — without it, the first bug is ambiguous.

Done when: a trivial edit lands, tests run, Aider's own retry loop fires on a deliberate failure.

### Step 2 — Router module (`aider/cost_autopilot/router.py`)
Wrap RouteLLM's `Controller` behind one function: `pick_starting_model(prompt, ladder) -> str`.

- `bert` router only (`routellm/bert_gpt4_augmented`) — the `mf` and `sw_ranking` routers call OpenAI's embeddings API, which breaks the local-only requirement. `mf`'s pretrained weights are dimensionally tied to OpenAI's 1536-dim embeddings and cannot be pointed at a local embedding model.
- RouteLLM's import chain constructs an `OpenAI()` client at module load even for `bert`. Needs a placeholder `OPENAI_API_KEY` env var set — never used for a real call, just satisfies the constructor.
- RouteLLM returns a binary weak/strong decision. Map that onto ladder positions.

Done when: unit tests confirm easy prompts route low and hard prompts route high, with no network calls.

**Known caveat to watch, not solve yet:** the pretrained router was calibrated on cloud-model preference data (GPT-4-class pairs), not small local models. Its decisions were directionally sensible in manual testing, but its confidence scores shouldn't be assumed well-calibrated for this ladder. Escalation is what makes this safe to ship despite that.

### Step 3 — Escalation module (`aider/cost_autopilot/escalation.py`)
The ladder-walking logic, independent of Aider: given a starting position, a way to attempt, and a way to validate, return the first attempt that validates or exhaust the ladder.

Pure and injectable — no Aider imports, no network. Unit tested with fakes, same approach that worked on the previous build.

Config: `max_attempts` (default 3) caps the walk so one stubborn task can't burn the entire ladder.

### Step 4 — Wire into the Coder
Subclass or wrap `Coder` so that a turn:
1. Calls `pick_starting_model()` on the user's message
2. Passes the result to `send(messages, model=...)`
3. Dry-runs the edits — malformed? escalate, nothing written
4. Applies edits, **holds the commit**
5. Runs `test_cmd` — pass? commit and finish. Fail? revert, escalate, retry
6. Ladder exhausted? surface the last attempt with an explicit "could not validate" note

Prefer wrapping over editing `base_coder.py` in place — a smaller diff against upstream is easier to keep in sync.

### Step 5 — Reporting
After each turn, print the trace inline: which model ran, whether it escalated, time taken, and dollar cost where LiteLLM knows the price (`$0` for local models is real, not a placeholder). No dashboard — terminal-native, consistent with living inside a terminal tool.

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
