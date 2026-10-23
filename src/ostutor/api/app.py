"""FastAPI service (Build Plan 8.7): /analyze_code, /simulate, /diagnose, /ask, /health.

    uvicorn ostutor.api.app:app --reload

Student input is validated before any computation; malformed programs or problems get a
422 with a friendly message (never a stack trace). The LLM is optional everywhere: when no
key is set, narration and answers fall back to templates and extracts.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ostutor.cache.store import Cache
from ostutor.diagnosis.engine import diagnose
from ostutor.dsl.parser import ParseError, canonical_hash, parse
from ostutor.explorer import replay
from ostutor.mutants.catalogue import Problem, solve
from ostutor.narration.llm import LLMClient, LLMConfig
from ostutor.narration.pipeline import analyze_cached, explain
from ostutor.rag.answer import answer as rag_answer
from ostutor.rag.chunk import chunk_pages
from ostutor.rag.index import HashingEmbedder, Index
from ostutor.rag.ingest import load_documents
from ostutor.rag.retrieve import Retriever
from ostutor.router.rules import route
from ostutor.sims.common import ValidationError
from ostutor.slice.explorer import ProgramError, StateBudgetExceeded

ROOT = Path(__file__).resolve().parents[3]


class CodeRequest(BaseModel):
    code: str = Field(..., max_length=20_000)
    tier: str = Field("auto", pattern="^(auto|A|B|C)$")
    fairness: str = Field("weak", pattern="^(weak|strong)$")
    narrate: bool = False   # True: LLM narration (checked); falls back to templates
    budget: int = Field(300_000, ge=10, le=2_000_000)


class ProblemRequest(BaseModel):
    problem: dict[str, Any]


class DiagnoseRequest(BaseModel):
    problem: dict[str, Any]
    answer: Any
    use_llm: bool = False


class AskRequest(BaseModel):
    message: str = Field(..., max_length=20_000)
    problem: dict[str, Any] | None = None
    answer: Any = None


def _unprocessable(kind: str, message: str, **extra) -> HTTPException:
    return HTTPException(422, detail={"error": kind, "message": message, **extra})


def create_app(retriever: Retriever | None = None, llm: LLMClient | None = None,
               cache: Cache | None = None) -> FastAPI:
    app = FastAPI(title="OS Tutor", version="0.4.0")
    state: dict[str, Any] = {"retriever": retriever, "cache": cache or Cache(":memory:")}

    def get_llm() -> LLMClient | None:
        if llm is not None:
            return llm
        cfg = LLMConfig.from_env()
        return LLMClient(cfg) if cfg.api_key else None

    def get_retriever() -> Retriever:
        if state["retriever"] is None:
            corpus = os.environ.get("OSTUTOR_CORPUS", str(ROOT / "data" / "corpus_sample"))
            chunks = chunk_pages(load_documents([corpus]))
            state["retriever"] = Retriever(Index(chunks, HashingEmbedder()))
        return state["retriever"]

    @app.get("/health")
    def health():
        cfg = LLMConfig.from_env()
        return {"status": "ok", "llm_key_set": bool(cfg.api_key or llm), "model": cfg.model}

    def run_code(req: CodeRequest) -> dict:
        try:
            parsed = parse(req.code)
            v = analyze_cached(parsed.program, state["cache"], parsed.loop_unroll, req.budget,
                               req.tier, req.fairness)
        except ParseError as e:
            raise _unprocessable("parse", e.message, line=e.line) from e
        except (ProgramError, ValueError) as e:
            raise _unprocessable("program", str(e)) from e
        except StateBudgetExceeded as e:
            raise _unprocessable("budget", f"{e}; use tier 'auto' to allow reduction or "
                                           "sampling, or raise the budget") from e
        # narrate=True always goes through the LLM path so a missing key is REPORTED as a
        # fall-back (with the reason) instead of silently returning a template
        client = (get_llm() or LLMClient(LLMConfig.from_env())) if req.narrate else None
        ex = explain(parsed.program, v, client, state["cache"])
        rp = replay(parsed.program, v.witness, v.label) if v.witness else None
        return {"verdict": v.to_dict(), "narration": ex.narration,
                "narration_source": ex.source, "narration_note": ex.note,
                "checked": True,
                "replay": {"ok": rp.ok, "reason": rp.reason} if rp else None,
                "canonical_hash": canonical_hash(parsed.program)}

    @app.post("/analyze_code")
    def analyze_code(req: CodeRequest):
        return run_code(req)

    def run_problem(p: dict) -> Problem:
        try:
            return Problem.from_dict(p)
        except ValidationError as e:
            raise _unprocessable("problem", str(e)) from e

    @app.post("/simulate")
    def simulate(req: ProblemRequest):
        try:
            return solve(run_problem(req.problem)).to_dict()
        except ValidationError as e:
            raise _unprocessable("problem", str(e)) from e

    @app.post("/diagnose")
    def diagnose_endpoint(req: DiagnoseRequest):
        problem = run_problem(req.problem)
        try:
            d = diagnose(problem, req.answer, get_llm() if req.use_llm else None)
        except ValidationError as e:
            raise _unprocessable("problem", str(e)) from e
        return {"diagnosis": d.to_dict(), "correct_answer": solve(problem).answer}

    @app.post("/ask")
    def ask(req: AskRequest):
        r = route(req.message)
        info = {"layer": r.layer, "reason": r.reason, "topic": r.topic, "rules": r.rules}
        if r.layer == "analyze_code":
            return {"route": info, "result": run_code(CodeRequest(code=req.message, tier="auto", fairness="weak",
                                                        narrate=False, budget=300_000))}
        if r.layer in ("simulate", "diagnose"):
            if req.problem is None:
                return {"route": info, "result": None, "needs_structured_input": True,
                        "message": "This looks like a scheduling/memory problem. Send the "
                                   "problem as structured data in 'problem' (topic, "
                                   "algorithm, params)" + (" and the student's 'answer'."
                                   if r.layer == "diagnose" else ".")}
            if r.layer == "diagnose" and req.answer is not None:
                return {"route": info, "result": diagnose_endpoint(
                    DiagnoseRequest(problem=req.problem, answer=req.answer))}
            return {"route": info, "result": solve(run_problem(req.problem)).to_dict()}
        a = rag_answer(req.message, get_retriever(), get_llm())
        return {"route": info, "result": {
            "answer": a.text, "found": a.found, "source": a.source,
            "citations": a.citations, "note": a.note}}

    return app


app = create_app()
