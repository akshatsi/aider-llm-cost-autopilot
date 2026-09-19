"""Unit tests for auto_mode.py -- fakes throughout, no real router
inference or real Model construction (that's the live check, not a
unit test). route_fn and model_factory are the injection points, same
pattern as pick_starting_model's own route_fn.
"""

import pytest

from aider.cost_autopilot.auto_mode import (
    enable_auto_routing,
    is_auto_model_name,
    parse_ladder,
)


class FakeIO:
    def __init__(self):
        self.output = []

    def tool_output(self, msg):
        self.output.append(msg)


class FakeCommands:
    """Stand-in for Coder.commands -- only is_command matters here."""

    def __init__(self, command_prefixes=("/",)):
        self.command_prefixes = command_prefixes

    def is_command(self, inp):
        return bool(inp) and inp.startswith(self.command_prefixes)


class FakeCoder:
    """Minimal stand-in for a real Coder -- just enough surface for
    enable_auto_routing to wrap: .run_one, .commands, .io, .main_model."""

    def __init__(self):
        self.io = FakeIO()
        self.commands = FakeCommands()
        self.main_model = "placeholder-model"
        self.run_one_calls = []

        def run_one(user_message, preproc):
            self.run_one_calls.append(
                {"user_message": user_message, "preproc": preproc, "main_model": self.main_model}
            )
            return "original run_one result"

        self.run_one = run_one


def test_is_auto_model_name_matches_case_and_whitespace_insensitively():
    assert is_auto_model_name("auto")
    assert is_auto_model_name("AUTO")
    assert is_auto_model_name("  Auto  ")
    assert not is_auto_model_name("ollama/qwen2.5:7b")
    assert not is_auto_model_name("")
    assert not is_auto_model_name(None)


def test_routes_using_the_real_user_message_run_one_receives():
    coder = FakeCoder()
    an_explicit_ladder = ["ollama/a", "ollama/b"]
    seen = {}

    def fake_route_fn(prompt, ladder):
        seen["prompt"] = prompt
        seen["ladder"] = ladder
        return "ollama/qwen2.5:7b"

    enable_auto_routing(
        coder,
        ladder=an_explicit_ladder,
        route_fn=fake_route_fn,
        model_factory=lambda name: f"model:{name}",
    )
    coder.run_one("the actual prompt to route on", preproc=True)

    assert seen["prompt"] == "the actual prompt to route on"
    assert seen["ladder"] == an_explicit_ladder


def test_no_ladder_triggers_discovery_not_a_hardcoded_default():
    """There is no DEFAULT_AUTO_LADDER any more -- ladder=None must go
    through discovery, not silently fall back to some fixed list."""
    coder = FakeCoder()
    discover_calls = []

    def fake_discover():
        discover_calls.append(1)
        return ["ollama/discovered-model"]

    enable_auto_routing(
        coder,
        discover_ladder_fn=fake_discover,
        route_fn=lambda prompt, ladder: ladder[0],
        model_factory=lambda name: name,
    )
    coder.run_one("do the thing", preproc=True)

    assert discover_calls == [1]
    assert coder.cost_autopilot_auto_ladder == ["ollama/discovered-model"]


def test_discovery_failure_propagates_instead_of_being_swallowed():
    coder = FakeCoder()

    def failing_discover():
        raise ValueError("no models available anywhere")

    with pytest.raises(ValueError, match="no models available"):
        enable_auto_routing(coder, discover_ladder_fn=failing_discover)


def test_main_model_is_set_before_the_original_run_one_is_called():
    """This is what actually fixes the spinner: send_message() builds
    its "Waiting for {model}" text from main_model before send() ever
    runs, so main_model has to be set before the original run_one --
    not inside some later call it makes -- for the spinner to show the
    routed model instead of a stale placeholder."""
    coder = FakeCoder()

    enable_auto_routing(
        coder,
        ladder=["ollama/qwen2.5:7b"],
        route_fn=lambda prompt, ladder: "ollama/qwen2.5:7b",
        model_factory=lambda name: f"model-object:{name}",
    )
    coder.run_one("do the thing", preproc=True)

    assert len(coder.run_one_calls) == 1
    assert coder.run_one_calls[0]["main_model"] == "model-object:ollama/qwen2.5:7b"


def test_a_slash_command_bypasses_routing_entirely():
    coder = FakeCoder()
    route_fn_calls = []

    enable_auto_routing(
        coder,
        ladder=["ollama/qwen2.5:7b"],
        route_fn=lambda prompt, ladder: route_fn_calls.append(1) or "ollama/qwen2.5:7b",
    )
    coder.run_one("/model ollama/qwen2.5:14b", preproc=True)

    assert route_fn_calls == []
    assert coder.main_model == "placeholder-model"  # untouched
    assert coder.run_one_calls[0]["user_message"] == "/model ollama/qwen2.5:14b"


def test_preproc_false_treats_a_leading_slash_as_a_literal_message_not_a_command():
    """Matches run_one's own real semantics: command detection only
    happens when preproc=True. A slash in the message shouldn't
    silently skip routing when preproc=False."""
    coder = FakeCoder()
    route_fn_calls = []

    def fake_route_fn(prompt, ladder):
        route_fn_calls.append(prompt)
        return "ollama/qwen2.5:7b"

    enable_auto_routing(
        coder, ladder=["ollama/qwen2.5:7b"], route_fn=fake_route_fn, model_factory=lambda name: name
    )
    coder.run_one("/not/actually/a/command", preproc=False)

    assert route_fn_calls == ["/not/actually/a/command"]


def test_an_empty_message_bypasses_routing_without_raising():
    coder = FakeCoder()
    route_fn_calls = []

    enable_auto_routing(
        coder,
        ladder=["ollama/qwen2.5:7b"],
        route_fn=lambda prompt, ladder: route_fn_calls.append(1) or "ollama/qwen2.5:7b",
    )
    coder.run_one("", preproc=True)

    assert route_fn_calls == []
    assert coder.main_model == "placeholder-model"


def test_model_construction_is_cached_across_repeated_routing_decisions():
    coder = FakeCoder()
    build_calls = []

    def model_factory(name):
        build_calls.append(name)
        return f"model:{name}"

    enable_auto_routing(
        coder,
        ladder=["ollama/qwen2.5:7b"],
        route_fn=lambda prompt, ladder: "ollama/qwen2.5:7b",
        model_factory=model_factory,
    )
    coder.run_one("message one", preproc=True)
    coder.run_one("message two", preproc=True)

    assert build_calls == ["ollama/qwen2.5:7b"]  # built once, reused the second time


def test_a_reflection_is_not_re_routed_because_run_one_is_only_called_once_per_real_turn():
    """Reflections are Aider's own same-model retry-with-error-feedback
    loop, driven entirely inside the original run_one's while loop --
    this wrapper never sees them, so a reflection can't get routed to
    a different model than the turn that triggered it."""
    coder = FakeCoder()
    route_fn_calls = []

    def fake_route_fn(prompt, ladder):
        route_fn_calls.append(prompt)
        return "ollama/qwen2.5:7b"

    enable_auto_routing(
        coder, ladder=["ollama/qwen2.5:7b"], route_fn=fake_route_fn, model_factory=lambda name: name
    )
    coder.run_one("write me a function", preproc=True)

    assert route_fn_calls == ["write me a function"]
    assert len(coder.run_one_calls) == 1


def test_announces_the_routing_decision_via_tool_output():
    coder = FakeCoder()

    enable_auto_routing(
        coder,
        ladder=["ollama/llama3.2:1b"],
        route_fn=lambda prompt, ladder: "ollama/llama3.2:1b",
        model_factory=lambda name: name,
    )
    coder.run_one("do the thing", preproc=True)

    assert any("ollama/llama3.2:1b" in msg for msg in coder.io.output)


def test_custom_ladder_is_passed_to_the_router_and_stored_on_the_coder():
    custom_ladder = ["ollama/a", "ollama/b"]
    coder = FakeCoder()
    seen_ladder = {}

    def fake_route_fn(prompt, ladder):
        seen_ladder["ladder"] = ladder
        return "ollama/a"

    enable_auto_routing(
        coder, ladder=custom_ladder, route_fn=fake_route_fn, model_factory=lambda name: name
    )
    coder.run_one("do the thing", preproc=True)

    assert seen_ladder["ladder"] == custom_ladder
    assert coder.cost_autopilot_auto_ladder == custom_ladder


# --- parse_ladder: shared by --auto-ladder and /model auto <ladder> -----


def test_parse_ladder_splits_on_commas_in_order():
    assert parse_ladder("ollama/a,ollama/b,ollama/c") == ["ollama/a", "ollama/b", "ollama/c"]


def test_parse_ladder_strips_whitespace_around_each_model():
    assert parse_ladder(" ollama/a , ollama/b ") == ["ollama/a", "ollama/b"]


def test_parse_ladder_drops_empty_entries_from_stray_commas():
    assert parse_ladder("ollama/a,,ollama/b,") == ["ollama/a", "ollama/b"]


def test_parse_ladder_accepts_a_single_model():
    assert parse_ladder("ollama/a") == ["ollama/a"]


def test_parse_ladder_rejects_a_string_with_no_usable_model_names():
    with pytest.raises(ValueError):
        parse_ladder(",, ,")
