"""Command line: paste a program file, get a verdict, witness and checked narration.

  python -m ostutor.cli analyze data/programs/deadlock_two_locks.ost
  python -m ostutor.cli analyze prog.ost --llm              # LLM narration, checked
  python -m ostutor.cli analyze prog.ost --llm --db cache.sqlite
  python -m ostutor.cli ping                                # test your API key
  python -m ostutor.cli models                              # list usable model ids
  python -m ostutor.cli warm                                # pre-record demo narrations
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ostutor.cache.store import Cache
from ostutor.dsl.parser import ParseError, canonical_hash, parse
from ostutor.explorer import replay
from ostutor.narration.llm import (
    FixtureStore,
    LLMClient,
    LLMConfig,
    LLMUnavailable,
    list_models,
)
from ostutor.narration.pipeline import analyze_cached, explain
from ostutor.slice.explorer import ProgramError, StateBudgetExceeded


def cmd_analyze(a) -> int:
    cache = Cache(a.db) if a.db else None
    try:
        return _analyze(a, cache)
    finally:
        if cache:
            cache.close()  # release the SQLite file (matters on Windows)


def _analyze(a, cache) -> int:
    try:
        parsed = parse(Path(a.file).read_text(), a.unroll)
    except OSError as e:
        print(f"cannot read {a.file}: {e}", file=sys.stderr)
        return 2
    except ParseError as e:
        print(f"Parse error, {e}", file=sys.stderr)
        return 2
    program = parsed.program
    try:
        v = analyze_cached(program, cache, parsed.loop_unroll, a.budget, a.tier, a.fairness,
                           a.pct_runs, a.seed)
    except ProgramError as e:
        print(f"Program error: {e}", file=sys.stderr)
        return 2
    except StateBudgetExceeded as e:
        print(f"Search stopped: {e}. Use --tier auto (default) to fall back to symmetry "
              "reduction or sampling, or raise --budget.", file=sys.stderr)
        return 3

    llm = None
    if a.llm:
        fx = FixtureStore(a.fixtures, a.fixture_mode) if a.fixtures else None
        llm = LLMClient(LLMConfig.from_env(), fixtures=fx)
    ex = explain(program, v, llm, cache)

    if a.json:
        print(json.dumps({"verdict": v.to_dict(), "narration": ex.narration,
                          "narration_source": ex.source, "note": ex.note}, indent=2))
        return 0

    print(f"label        : {v.label}")
    print(f"completeness : {v.completeness}  (bounds: {v.bounds})")
    print(f"states       : {v.states_explored}")
    print(f"program hash : {canonical_hash(program)[:19]}... (canonical)")
    if v.witness:
        print("witness (shortest):")
        for s in v.witness.steps:
            blocked = f"   blocked after: {', '.join(s.blocked)}" if s.blocked else ""
            print(f"  {s.i}. {s.thread}: {s.op} {s.object}{blocked}")
        r = replay(program, v.witness, expected=v.label)
        print(f"replay       : {'reproduced ' + str(r.failure) if r.ok else 'FAILED - ' + r.reason}")
    if v.critical_point:
        cp = v.critical_point
        print(f"critical pt  : after step {cp['decision_after_step']}: bad choice "
              f"{cp['bad_choice']} vs safe alternative {cp['safe_alternative']}")
    print(f"narration [{ex.badge}]:")
    for c in ex.narration["claims"]:
        print(f"  [step {c['step']}] {c['text']}")
    print(f"  summary: {ex.narration['summary']}")
    print("  (the checker validates facts, not causal reasoning)")
    if ex.note:
        print(f"note         : {ex.note}")
    if llm:
        print(f"llm calls    : {llm.calls}   tokens: {llm.tokens_used}")
    return 0


def cmd_ping(a) -> int:
    cfg = LLMConfig.from_env()
    print(f"endpoint: {cfg.base_url}\nmodel   : {cfg.model}\n"
          f"key     : {'set (' + cfg.api_key[:4] + '...)' if cfg.api_key else 'NOT SET'}")
    client = LLMClient(cfg)
    cfg.max_tokens, cfg.json_mode = 20, False
    try:
        reply = client.complete("Reply with one word.", "Say OK.")
    except LLMUnavailable as e:
        print(f"FAILED: {e}")
        return 1
    print(f"reply   : {reply.strip()!r}   tokens used: {client.tokens_used}\nLLM access works.")
    return 0


def cmd_warm(a) -> int:
    """Pre-record LLM narrations for every example so the demo can run offline."""
    cfg = LLMConfig.from_env()
    cache = Cache(a.db)
    fx = FixtureStore(a.fixtures, "auto")
    llm = LLMClient(cfg, fixtures=fx)
    bad = 0
    try:
        for f in sorted(Path(a.programs).glob("*.ost")):
            parsed = parse(f.read_text())
            v = analyze_cached(parsed.program, cache, parsed.loop_unroll)
            ex = explain(parsed.program, v, llm, cache)
            ok = ex.source in ("llm", "cache")
            bad += not ok
            print(f"{f.stem:24} {ex.badge:22} {ex.note}")
    finally:
        cache.close()
    print(f"\nLLM calls: {llm.calls}, tokens: {llm.tokens_used}. Fixtures in {a.fixtures}/")
    if bad:
        print(f"{bad} program(s) fell back to the template; re-run, or check `ping`/`models`.")
    return 1 if bad else 0


def cmd_models(a) -> int:
    cfg = LLMConfig.from_env()
    try:
        ids = list_models(cfg)
    except LLMUnavailable as e:
        print(f"FAILED: {e}")
        return 1
    print(f"models available to your key at {cfg.base_url}:")
    for m in ids:
        print(f"  {m}{'   <- current OSTUTOR_MODEL' if m == cfg.model else ''}")
    if cfg.model not in ids:
        print(f"\nWARNING: '{cfg.model}' is not in this list. Set OSTUTOR_MODEL=<id> in .env.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ostutor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("analyze", help="analyze a .ost program file")
    p.add_argument("file")
    p.add_argument("--llm", action="store_true", help="narrate with the LLM (checked)")
    p.add_argument("--db", help="SQLite cache file (verdicts + narrations)")
    p.add_argument("--fixtures", help="directory for recorded LLM responses")
    p.add_argument("--fixture-mode", choices=["auto", "record", "replay"], default="auto")
    p.add_argument("--unroll", type=int, default=2, help="bound for 'while' loops")
    p.add_argument("--budget", type=int, default=300_000, help="state budget")
    p.add_argument("--tier", choices=["auto", "A", "B", "C"], default="auto",
                   help="A exhaustive, B symmetry-reduced, C PCT sampling (default: auto)")
    p.add_argument("--fairness", choices=["weak", "strong"], default="weak",
                   help="fairness assumption for starvation analysis")
    p.add_argument("--pct-runs", type=int, default=2000, help="Tier C random schedules")
    p.add_argument("--seed", type=int, default=0, help="Tier C random seed")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_analyze)
    w = sub.add_parser("warm", help="pre-record LLM narrations for all examples (demo prep)")
    w.add_argument("--programs", default="data/programs")
    w.add_argument("--db", default="demo_cache.sqlite")
    w.add_argument("--fixtures", default="demo_fixtures")
    w.set_defaults(fn=cmd_warm)
    sub.add_parser("models", help="list model ids your key can use").set_defaults(fn=cmd_models)
    q = sub.add_parser("ping", help="check that your API key works")
    q.set_defaults(fn=cmd_ping)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except BrokenPipeError:  # the reader of our output (head, grep -q) went away: not an error
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except (OSError, ValueError):
            pass
        return 0


if __name__ == "__main__":
    sys.exit(main())
