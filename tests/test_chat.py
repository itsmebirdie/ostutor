from ostutor.chat.context import build_context, match_route
from ostutor.chat.llm import GroqClient
from ostutor.chat.session import ChatSession


def fake_transport(captured):
    def t(payload):
        captured.append(payload)
        return {"choices": [{"message": {"content": " hi "}}]}
    return t


def test_routes():
    assert match_route("What is a deadlock?")[0] == ["deadlock_two_locks"]
    assert match_route("explain round robin")[0] == ["--rr"]
    assert match_route("what is paging?") is None


def test_context_failure_is_silent():
    assert build_context("deadlock?", runner=lambda a: None) is None


def test_session_injects_context_but_does_not_store_it():
    cap = []
    s = ChatSession(GroqClient(transport=fake_transport(cap)),
                    context_builder=lambda m: ("deadlock", "Verdict: `deadlock`"))
    assert s.ask("what is a deadlock?") == "hi"
    msgs = cap[0]["messages"]
    assert msgs[0]["role"] == "system"
    assert any("VERIFIED ANALYSIS" in m["content"] for m in msgs)
    assert msgs[-1] == {"role": "user", "content": "what is a deadlock?"}
    assert len(s.history) == 2 and "VERIFIED" not in s.history[0]["content"]


def test_history_trimmed():
    cap = []
    s = ChatSession(GroqClient(transport=fake_transport(cap)), max_turns=2,
                    context_builder=lambda m: None)
    for i in range(6):
        s.ask(f"q{i}")
    assert len(cap[-1]["messages"]) == 1 + 4 + 1  # system + 2 turns + new msg