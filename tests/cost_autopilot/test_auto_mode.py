"""Unit tests for auto_mode.py -- fakes throughout, no real router
inference or real Model construction (that's the live check, not a
unit test). route_fn and model_factory are the injection points, same
pattern as pick_starting_model's own route_fn.
"""

from aider.cost_autopilot.auto_mode import (
    DEFAULT_AUTO_LADDER,
    enable_auto_routing,
    is_auto_model_name,
)


class FakeIO:
    def __init__(self):
        self.output = []

    def tool_output(self, msg):
        self.output.append(msg)


class FakeCoder:
    """Minimal stand-in for a real Coder -- just enough surface for
    enable_auto_routing to wrap: .send, .cur_messages, .io."""

    def __init__(self):
        self.io = FakeIO()
        self.cur_messages = []
        self.send_calls = []

        def send(messages, model=None, functions=None):
            self.send_calls.append({"messages": messages, "model": model, "functions": functions})
            return "original send result"

        self.send = send


def test_is_auto_model_name_matches_case_and_whitespace_insensitively():
    assert is_auto_model_name("auto")
    assert is_auto_model_name("AUTO")
    assert is_auto_model_name("  Auto  ")
    assert not is_auto_model_name("ollama/qwen2.5:7b")
    assert not is_auto_model_name("")
    assert not is_auto_model_name(None)


def test_routes_using_the_latest_user_message_as_the_prompt():
    coder = FakeCoder()
    coder.cur_messages = [
        {"role": "user", "content": "earlier turn, ignored"},
        {"role": "assistant", "content": "earlier reply, ignored"},
        {"role": "user", "content": "the actual prompt to route on"},
    ]
    seen = {}

    def fake_route_fn(prompt, ladder):
        seen["prompt"] = prompt
        seen["ladder"] = ladder
        return "ollama/qwen2.5:7b"

    enable_auto_routing(
        coder, route_fn=fake_route_fn, model_factory=lambda name: f"model:{name}"
    )
    coder.send(["formatted", "messages"])

    assert seen["prompt"] == "the actual prompt to route on"
    assert seen["ladder"] == DEFAULT_AUTO_LADDER


def test_routed_model_is_passed_through_to_the_original_send():
    coder = FakeCoder()
    coder.cur_messages = [{"role": "user", "content": "do the thing"}]

    enable_auto_routing(
        coder,
        route_fn=lambda prompt, ladder: "ollama/qwen2.5:7b",
        model_factory=lambda name: f"model-object:{name}",
    )
    coder.send(["formatted", "messages"])

    assert len(coder.send_calls) == 1
    assert coder.send_calls[0]["model"] == "model-object:ollama/qwen2.5:7b"


def test_an_explicit_model_bypasses_routing_entirely():
    coder = FakeCoder()
    coder.cur_messages = [{"role": "user", "content": "do the thing"}]
    route_fn_calls = []

    enable_auto_routing(
        coder,
        route_fn=lambda prompt, ladder: route_fn_calls.append(1) or "ollama/qwen2.5:7b",
    )
    coder.send(["formatted", "messages"], model="explicit-override")

    assert route_fn_calls == []
    assert coder.send_calls[0]["model"] == "explicit-override"


def test_model_construction_is_cached_across_repeated_routing_decisions():
    coder = FakeCoder()
    coder.cur_messages = [{"role": "user", "content": "do the thing"}]
    build_calls = []

    def model_factory(name):
        build_calls.append(name)
        return f"model:{name}"

    enable_auto_routing(
        coder,
        route_fn=lambda prompt, ladder: "ollama/qwen2.5:7b",
        model_factory=model_factory,
    )
    coder.send(["messages one"])
    coder.send(["messages two"])

    assert build_calls == ["ollama/qwen2.5:7b"]  # built once, reused the second time


def test_announces_the_routing_decision_via_tool_output():
    coder = FakeCoder()
    coder.cur_messages = [{"role": "user", "content": "do the thing"}]

    enable_auto_routing(
        coder,
        route_fn=lambda prompt, ladder: "ollama/llama3.2:1b",
        model_factory=lambda name: name,
    )
    coder.send(["messages"])

    assert any("ollama/llama3.2:1b" in msg for msg in coder.io.output)


def test_custom_ladder_is_passed_to_the_router_and_stored_on_the_coder():
    coder = FakeCoder()
    coder.cur_messages = [{"role": "user", "content": "do the thing"}]
    custom_ladder = ["ollama/a", "ollama/b"]
    seen_ladder = {}

    def fake_route_fn(prompt, ladder):
        seen_ladder["ladder"] = ladder
        return "ollama/a"

    enable_auto_routing(
        coder, ladder=custom_ladder, route_fn=fake_route_fn, model_factory=lambda name: name
    )
    coder.send(["messages"])

    assert seen_ladder["ladder"] == custom_ladder
    assert coder.cost_autopilot_auto_ladder == custom_ladder


def test_empty_cur_messages_routes_on_an_empty_prompt_without_raising():
    coder = FakeCoder()
    coder.cur_messages = []
    seen = {}

    def fake_route_fn(prompt, ladder):
        seen["prompt"] = prompt
        return "ollama/a"

    enable_auto_routing(coder, route_fn=fake_route_fn, model_factory=lambda name: name)
    coder.send(["messages"])

    assert seen["prompt"] == ""
