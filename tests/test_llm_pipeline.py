import json

import pytest

from ostutor.cache.store import Cache
from ostutor.narration.evidence import encode_evidence, estimate_tokens
from ostutor.narration.llm import FixtureStore, LLMClient, LLMConfig, LLMUnavailable, MissingFixture
from ostutor.narration.pipeline import MAX_REPROMPTS, analyze_cached, explain, parse_narration
from ostutor.slice.check import check
from ostutor.slice.explorer import analyze
from ostutor.slice.narrate import narrate
from ostutor.slice.programs import PROGRAMS

DEADLOCK = PROGRAMS["deadlock_two_locks"]


def ok_response(program, verdict):
    """A narration a well-behaved LLM would return (here: the template's)."""
    return json.dumps(narrate(program, verdict))


def make_client(replies, **cfg):
    """LLMClient whose transport serves `replies` (strings or (status, payload, headers))."""
    calls = {"n": 0, "bodies": []}
    queue = list(replies)

    def transport(url, headers, body, timeout):
        calls["n"] += 1
        calls["bodies"].append(json.loads(json.dumps(body)))  # snapshot
        item = queue.pop(0)
        if isinstance(item, tuple):
            return item
        return 200, {"choices": [{"message": {"content": item}}],
                     "usage": {"total_tokens": 100}}, {}

    client = LLMClient(LLMConfig(api_key="k", **cfg), transport=transport,
                       sleep=lambda s: None)
    return client, calls


def test_evidence_is_compact_and_complete():
    v = analyze(DEADLOCK)
    ev = encode_evidence(DEADLOCK, v)
    assert "label: deadlock" in ev and "step 2: T2 lock b" in ev and "critical_point" in ev
    assert estimate_tokens(ev) <= 1000


def test_good_llm_output_is_accepted_and_cached():
    v = analyze(DEADLOCK)
    client, calls = make_client([ok_response(DEADLOCK, v)])
    cache = Cache()
    first = explain(DEADLOCK, v, client, cache)
    assert first.source == "llm" and first.checked and calls["n"] == 1
    second = explain(DEADLOCK, v, client, cache)  # same question twice -> zero tokens
    assert second.source == "cache" and calls["n"] == 1
    assert cache.hits["narration"] == 1


def test_bad_output_is_reprompted_with_violations_then_accepted():
    v = analyze(DEADLOCK)
    bad = json.loads(ok_response(DEADLOCK, v))
    bad["summary"] = bad["summary"].replace("`deadlock`", "`safe`", 1)
    client, calls = make_client([json.dumps(bad), ok_response(DEADLOCK, v)])
    e = explain(DEADLOCK, v, client)
    assert e.source == "llm" and e.attempts == 2
    second_prompt = calls["bodies"][1]["messages"][1]["content"]
    assert "label:" in second_prompt and "Violated rules" in second_prompt


def test_persistent_failure_falls_back_to_template_after_two_reprompts():
    v = analyze(DEADLOCK)
    bad = json.loads(ok_response(DEADLOCK, v))
    bad["claims"].append({"step": 99, "text": "`T1` locks `a`."})
    client, calls = make_client([json.dumps(bad)] * 5)
    e = explain(DEADLOCK, v, client, Cache())
    assert e.source == "template_fallback" and e.attempts == MAX_REPROMPTS + 1 == calls["n"]
    assert e.violations and check(e.narration, v, DEADLOCK).ok
    assert e.badge == "template fall-back"


def test_llm_cannot_smuggle_in_a_number():
    v = analyze(DEADLOCK)
    bad = json.loads(ok_response(DEADLOCK, v))
    bad["summary"] += " This happens in 73% of runs."
    assert not check(bad, v, DEADLOCK).ok


def test_non_json_reply_is_rejected_then_recovers():
    v = analyze(DEADLOCK)
    client, _ = make_client(["Sure! Here is an explanation...",
                             "```json\n" + ok_response(DEADLOCK, v) + "\n```"])
    assert explain(DEADLOCK, v, client).source == "llm"


def test_parse_narration_shapes():
    assert parse_narration("{}")[0] is None
    assert parse_narration('{"claims": [{"step": "x"}], "summary": "s"}')[0] is None
    n, errs = parse_narration('{"claims": [], "summary": "s"}')
    assert n == {"claims": [], "summary": "s"} and not errs


def test_no_key_or_outage_degrades_to_template():
    v = analyze(DEADLOCK)
    nokey = LLMClient(LLMConfig(api_key=""))
    e = explain(DEADLOCK, v, nokey)
    assert e.source == "template_fallback" and "no API key" in e.note

    client, _ = make_client([(401, {"error": {"message": "bad key"}}, {})])
    e = explain(DEADLOCK, v, client)
    assert e.source == "template_fallback" and "authentication" in e.note


def test_rate_limit_is_retried_with_backoff():
    v = analyze(DEADLOCK)
    waits = []
    replies = [(429, {"error": {"message": "slow down"}}, {"retry-after": "2"}),
               ok_response(DEADLOCK, v)]
    client, calls = make_client(replies)
    client.sleep = waits.append
    assert explain(DEADLOCK, v, client).source == "llm"
    assert waits == [2.0] and calls["n"] == 2


def test_daily_quota_does_not_retry():
    client, calls = make_client([(429, {"error": {"message": "Rate limit reached: "
                                                  "tokens per day (TPD)"}}, {})])
    with pytest.raises(LLMUnavailable):
        client.complete("s", "u")
    assert calls["n"] == 1


def test_json_mode_unsupported_is_retried_without_it():
    client, calls = make_client([(400, {"error": {"message": "response_format unsupported"}}, {}),
                                 "hello"])
    assert client.complete("s", "u") == "hello"
    assert "response_format" in calls["bodies"][0]
    assert "response_format" not in calls["bodies"][1]


def test_fixtures_record_then_replay_without_network(tmp_path):
    v = analyze(DEADLOCK)
    client, calls = make_client([ok_response(DEADLOCK, v)])
    client.fixtures = FixtureStore(tmp_path, "record")
    assert explain(DEADLOCK, v, client).source == "llm"

    offline = LLMClient(LLMConfig(api_key=""), fixtures=FixtureStore(tmp_path, "replay"))
    e = explain(DEADLOCK, v, offline)
    assert e.source == "llm" and offline.calls == 0

    empty = LLMClient(LLMConfig(api_key=""), fixtures=FixtureStore(tmp_path / "none", "replay"))
    with pytest.raises(MissingFixture):
        empty.complete("a", "b")


def test_verdict_cache_roundtrip_and_hit_count():
    cache = Cache()
    v1 = analyze_cached(DEADLOCK, cache)
    v2 = analyze_cached(DEADLOCK, cache)
    assert v1 == v2 and cache.hits["verdict"] == 1 and cache.misses["verdict"] == 1
    assert v2.witness.steps[1].thread == "T2"
    other = analyze_cached(DEADLOCK, cache, loop_unroll=3)  # different bounds -> new key
    assert cache.misses["verdict"] == 2 and other.bounds["loop_unroll"] == 3


def test_sqlite_cache_persists_on_disk(tmp_path):
    path = str(tmp_path / "c.sqlite")
    c = Cache(path); analyze_cached(DEADLOCK, c); c.close()
    c2 = Cache(path); analyze_cached(DEADLOCK, c2)
    assert c2.hits["verdict"] == 1
    c2.close()


# ---- retired models / reasoning models -------------------------------------
def test_retired_model_gives_an_actionable_error():
    msg = "The model `llama-3.3-70b-versatile` does not exist or you do not have access to it."
    client, calls = make_client([(404, {"error": {"message": msg}}, {})], model="old-model")
    with pytest.raises(LLMUnavailable) as e:
        client.complete("s", "u")
    assert "old-model" in str(e.value) and "ostutor.cli models" in str(e.value)
    assert calls["n"] == 1  # not retried


def test_default_model_is_not_the_retired_llama():
    assert LLMConfig().model == "openai/gpt-oss-120b"


def test_reasoning_effort_is_sent_only_to_gpt_oss_models():
    c1, calls1 = make_client(["x"], model="openai/gpt-oss-120b")
    c1.complete("s", "u")
    assert calls1["bodies"][0]["reasoning_effort"] == "low"
    c2, calls2 = make_client(["x"], model="qwen/qwen3.6-27b")
    c2.complete("s", "u")
    assert "reasoning_effort" not in calls2["bodies"][0]


def test_empty_reply_from_a_reasoning_model_is_treated_as_bad_output_not_a_crash():
    v = analyze(DEADLOCK)
    client, _ = make_client([(200, {"choices": [{"message": {"content": None}}]}, {}),
                             ok_response(DEADLOCK, v)])
    assert explain(DEADLOCK, v, client).source == "llm"
