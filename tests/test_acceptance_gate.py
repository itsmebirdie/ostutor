"""The review slice's acceptance gate (build plan 2.8), automated.

Items 1 and 5 run the real command line in a subprocess, exactly as in the demo.
Items 2-4 run in-process over every program.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ostutor.slice.check import check
from ostutor.slice.explorer import analyze
from ostutor.slice.narrate import narrate
from ostutor.slice.programs import PROGRAMS
from ostutor.slice.replay import replay

ROOT = Path(__file__).parent.parent
ANSI = re.compile(r"\x1b\[[0-9;]*m")
HAND_VERIFIED = {"deadlock_two_locks": "deadlock", "race_no_lock": "race",
                 "safe_with_lock": "safe"}


def run(args, tmp_path):
    env = {k: v for k, v in os.environ.items()
           if k not in ("GROQ_API_KEY", "OSTUTOR_API_KEY")}
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONIOENCODING"] = "utf-8"  # Windows consoles default to a legacy code page
    r = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True, encoding="utf-8",
                       env=env, cwd=tmp_path, timeout=60)
    return r.returncode, ANSI.sub("", r.stdout), ANSI.sub("", r.stderr)


def test_gate1_demo_deadlock_prints_everything(tmp_path):
    code, out, _ = run(["ostutor.slice.demo", "deadlock_two_locks"], tmp_path)
    assert code == 0
    for needle in ("label        : deadlock", "T1: lock a", "T2: lock b", "states       : 19",
                   "critical point", "CHECKED", "reproduced deadlock"):
        assert needle in out, needle


@pytest.mark.parametrize("name,label", HAND_VERIFIED.items())
def test_gate2_hand_verified_labels(name, label):
    assert analyze(PROGRAMS[name]).label == label


@pytest.mark.parametrize("name", HAND_VERIFIED)
def test_gate3_every_witness_replays(name):
    v = analyze(PROGRAMS[name])
    if v.witness is None:
        assert v.label == "safe"
    else:
        assert replay(PROGRAMS[name], v.witness, expected=v.label).ok


@pytest.mark.parametrize("name", HAND_VERIFIED)
def test_gate4_checker_rejects_four_tamperings_on_every_program(name):
    p = PROGRAMS[name]
    v = analyze(p)
    assert check(narrate(p, v), v, p).ok  # untampered template passes

    def tampered(mutate):
        n = narrate(p, v)
        mutate(n)
        return not check(n, v, p).ok

    wrong = "safe" if v.label != "safe" else "deadlock"
    if v.witness:  # the safe program has no witness steps to invent against
        assert tampered(lambda n: n["claims"].append({"step": 99, "text": "x"}))
    assert tampered(lambda n: n.update(summary=n["summary"] + " `T9` runs."))
    assert tampered(lambda n: n.update(summary=n["summary"] + " After 1019 states."))
    assert tampered(lambda n: n.update(summary=n["summary"].replace(
        f"`{v.label}`", f"`{wrong}`", 1)))


def test_gate5_round_robin_mutant_is_named_with_first_divergence(tmp_path):
    code, out, _ = run(["ostutor.slice.demo", "--rr"], tmp_path)
    assert code == 0
    assert "mutant_match" in out and "time unit 5" in out
    assert "expected P3, got P1" in out


def test_new_cli_end_to_end_without_a_key(tmp_path):
    prog = ROOT / "data" / "programs" / "dining_3_deadlock.ost"
    code, out, err = run(["ostutor.cli", "analyze", str(prog), "--llm"], tmp_path)
    assert code == 0, err
    assert "label        : deadlock" in out
    assert "template fall-back" in out and "no API key" in out  # graceful degradation


def test_cli_parse_error_is_friendly_and_exit_code_2(tmp_path):
    bad = tmp_path / "bad.ost"
    bad.write_text("mutex a\nthread T1:\n  sleep 5\n")
    code, _, err = run(["ostutor.cli", "analyze", str(bad)], tmp_path)
    assert code == 2 and "line 3" in err and "timing" in err


@pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX shell pipeline")
def test_cli_survives_a_reader_that_closes_the_pipe_early(tmp_path):
    """`ostutor analyze ... | grep -q ...` must not print a BrokenPipeError traceback."""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    prog = ROOT / "data" / "programs" / "deadlock_two_locks.ost"
    r = subprocess.run(["bash", "-c", f"{sys.executable} -m ostutor.cli analyze {prog} | head -1"],
                       capture_output=True, text=True, env=env, cwd=tmp_path, timeout=60)
    assert r.stdout.startswith("label") and "Traceback" not in r.stderr and "Broken" not in r.stderr
