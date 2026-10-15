"""Command line: paste a program file, get a verdict, witness and checked narration.

  python -m ostutor.cli analyze data/programs/deadlock_two_locks.ost
  python -m ostutor.cli analyze prog.ost --llm              # LLM narration, checked
  python -m ostutor.cli analyze prog.ost --llm --db cache.sqlite
  python -m ostutor.cli ping                                # test your API key
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ostutor.cache.store import Cache
from ostutor.dsl.parser import ParseError, canonical_hash, parse
from ostutor.narration.llm import FixtureStore, LLMClient, LLMConfig, LLMUnavailable
from ostutor.narration.pipeline import analyze_cached, explain
from ostutor.slice.explorer import ProgramError, StateBudgetExceeded
from ostutor.slice.replay import replay


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
        v = analyze_cached(program, cache, parsed.loop_unroll, a.budget)
    except ProgramError as e:
        print(f"Program error: {e}", file=sys.stderr)
        return 2
    except StateBudgetExceeded as e:
        print(f"Search stopped: {e}. Tier B/C reduction is not built yet; "
              "try a smaller program or --budget.", file=sys.stderr)
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
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_analyze)
    q = sub.add_parser("ping", help="check that your API key works")
    q.set_defaults(fn=cmd_ping)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
