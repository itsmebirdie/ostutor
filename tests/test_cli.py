"""In-process CLI tests (fast, and counted by coverage)."""
import json
from pathlib import Path

from ostutor.cli import main
from ostutor.dsl.parser import parse
from ostutor.narration.llm import FixtureStore, LLMClient, LLMConfig
from ostutor.narration.pipeline import explain
from ostutor.slice.explorer import analyze
from ostutor.slice.narrate import narrate

DATA = Path(__file__).parent.parent / "data" / "programs"
DEADLOCK = DATA / "deadlock_two_locks.ost"


def test_analyze_text_output(capsys):
    assert main(["analyze", str(DEADLOCK)]) == 0
    out = capsys.readouterr().out
    assert "label        : deadlock" in out and "reproduced" in out
    assert "[template]" in out and "checker validates facts" in out


def test_analyze_json_output(capsys):
    assert main(["analyze", str(DATA / "safe_with_lock.ost"), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["verdict"]["label"] == "safe" and data["narration_source"] == "template"


def test_missing_file_and_parse_error_exit_2(capsys, tmp_path):
    assert main(["analyze", str(tmp_path / "nope.ost")]) == 2
    bad = tmp_path / "b.ost"
    bad.write_text("thread T1:\n  lock zz\n")
    assert main(["analyze", str(bad)]) == 2
    assert "not declared" in capsys.readouterr().err


def test_state_budget_exit_3(capsys):
    assert main(["analyze", str(DEADLOCK), "--budget", "3"]) == 3
    assert "Search stopped" in capsys.readouterr().err


def test_db_cache_makes_second_run_a_hit(capsys, tmp_path):
    db = str(tmp_path / "c.sqlite")
    assert main(["analyze", str(DEADLOCK), "--db", db]) == 0
    assert main(["analyze", str(DEADLOCK), "--db", db]) == 0
    assert (tmp_path / "c.sqlite").exists()


def test_llm_path_runs_fully_offline_from_a_recorded_fixture(capsys, tmp_path):
    """Record one fake LLM answer, then replay it through the real CLI with no key."""
    program = parse(DEADLOCK.read_text()).program
    verdict = analyze(program)
    canned = json.dumps(narrate(program, verdict))

    def transport(url, headers, body, timeout):
        return 200, {"choices": [{"message": {"content": canned}}],
                     "usage": {"total_tokens": 321}}, {}

    recorder = LLMClient(LLMConfig(api_key="k"), FixtureStore(tmp_path, "record"),
                         transport=transport)
    assert explain(program, verdict, recorder).source == "llm"

    code = main(["analyze", str(DEADLOCK), "--llm", "--fixtures", str(tmp_path),
                 "--fixture-mode", "replay"])
    out = capsys.readouterr().out
    assert code == 0 and "checked (LLM)" in out and "llm calls    : 0" in out


def test_llm_flag_without_key_degrades(capsys):
    assert main(["analyze", str(DEADLOCK), "--llm"]) == 0
    assert "template fall-back" in capsys.readouterr().out


def test_ping_without_key_fails_cleanly(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # so no developer .env is picked up
    assert main(["ping"]) == 1
    out = capsys.readouterr().out
    assert "NOT SET" in out and "FAILED" in out
