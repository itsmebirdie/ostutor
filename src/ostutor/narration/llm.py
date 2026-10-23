"""LLM client for narration. Standard library only (urllib), no SDK needed.

Talks to any OpenAI-compatible chat-completions endpoint. Default: Groq.
Configuration (environment variables or a .env file in the working directory):

    OSTUTOR_API_KEY   (or GROQ_API_KEY)   the secret key
    OSTUTOR_BASE_URL  default https://api.groq.com/openai/v1
    OSTUTOR_MODEL     default openai/gpt-oss-120b

Fixtures (Build Plan 10.1): every response can be recorded once and replayed in
tests so CI spends zero tokens. Modes: "replay" (never touches the network),
"record" (always call, then save), "auto" (use a saved response if present).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
# Groq retired llama-3.3-70b-versatile on 2026-08-16 and recommends gpt-oss-120b / qwen3.6-27b.
# Model ids change often: run `python -m ostutor.cli models` to list what your key can use.
DEFAULT_MODEL = "openai/gpt-oss-120b"


class LLMUnavailable(RuntimeError):
    """No key, bad key, network down, or rate limit not recoverable. Callers fall back."""


class MissingFixture(LLMUnavailable):
    pass


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader (KEY=value lines); never overrides real env variables."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, val = line.partition("=")
        os.environ.setdefault(k.strip(), val.strip().strip('"').strip("'"))


@dataclass
class LLMConfig:
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    max_tokens: int = 2000  # reasoning models (gpt-oss) also spend tokens "thinking"
    temperature: float = 0.0
    timeout: float = 30.0
    json_mode: bool = True
    max_retries: int = 3
    reasoning_effort: str = "low"  # sent only to gpt-oss models; keeps narration cheap

    @classmethod
    def from_env(cls) -> LLMConfig:
        load_dotenv()
        return cls(
            api_key=os.environ.get("OSTUTOR_API_KEY") or os.environ.get("GROQ_API_KEY", ""),
            base_url=os.environ.get("OSTUTOR_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            model=os.environ.get("OSTUTOR_MODEL", DEFAULT_MODEL))


@dataclass
class FixtureStore:
    directory: Path
    mode: str = "auto"  # "replay" | "record" | "auto"

    def __post_init__(self):
        self.directory = Path(self.directory)

    @staticmethod
    def key(model: str, system: str, user: str) -> str:
        return hashlib.sha256(json.dumps([model, system, user]).encode()).hexdigest()[:24]

    def get(self, key: str) -> str | None:
        f = self.directory / f"{key}.json"
        return json.loads(f.read_text())["response"] if f.is_file() else None

    def put(self, key: str, system: str, user: str, response: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f"{key}.json").write_text(
            json.dumps({"system": system, "user": user, "response": response}, indent=1))


Transport = Callable[[str, dict, dict, float], tuple[int, dict, dict]]


def _urllib_transport(url: str, headers: dict, body: dict, timeout: float):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers,
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode()), dict(r.headers)
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode())
        except Exception:
            payload = {}
        return e.code, payload, dict(e.headers or {})
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMUnavailable(f"network error: {e}") from e


@dataclass
class LLMClient:
    config: LLMConfig = field(default_factory=LLMConfig.from_env)
    fixtures: FixtureStore | None = None
    transport: Transport = _urllib_transport
    sleep: Callable[[float], None] = time.sleep
    calls: int = 0  # live network calls made (for budgeting/tests)
    tokens_used: int = 0

    def complete(self, system: str, user: str) -> str:
        fx = self.fixtures
        key = FixtureStore.key(self.config.model, system, user) if fx else ""
        if fx and fx.mode in ("replay", "auto"):
            saved = fx.get(key)
            if saved is not None:
                return saved
            if fx.mode == "replay":
                raise MissingFixture(f"no recorded response for {key}")
        text = self._call(system, user)
        if fx and fx.mode in ("record", "auto"):
            fx.put(key, system, user, text)
        return text

    def _call(self, system: str, user: str) -> str:
        c = self.config
        if not c.api_key:
            raise LLMUnavailable("no API key (set OSTUTOR_API_KEY or GROQ_API_KEY)")
        body = {"model": c.model, "temperature": c.temperature, "max_tokens": c.max_tokens,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}
        if c.json_mode:
            body["response_format"] = {"type": "json_object"}
        if "gpt-oss" in c.model and c.reasoning_effort:
            body["reasoning_effort"] = c.reasoning_effort
        headers = {"Authorization": f"Bearer {c.api_key}",
                   "Content-Type": "application/json",
                   # Groq sits behind Cloudflare, which can reject urllib's default agent
                   "User-Agent": "ostutor/0.2 (+https://github.com/itsmebirdie/ostutor)"}
        url = f"{c.base_url}/chat/completions"
        last = ""
        for attempt in range(c.max_retries + 1):
            status, payload, hdrs = self.transport(url, headers, body, c.timeout)
            self.calls += 1
            if status == 200:
                try:
                    self.tokens_used += int(payload.get("usage", {}).get("total_tokens", 0))
                    return payload["choices"][0]["message"].get("content") or ""
                except (KeyError, IndexError, TypeError) as e:
                    raise LLMUnavailable(f"unexpected response shape: {payload!r}") from e
            err = (payload.get("error") or {}).get("message", "") if isinstance(payload, dict) else ""
            last = f"HTTP {status} {err}".strip()
            if status == 404 or "decommissioned" in err or "model_not_found" in str(payload):
                raise LLMUnavailable(
                    f"model '{c.model}' is not available ({last}). It may have been retired; "
                    "run `python -m ostutor.cli models` and set OSTUTOR_MODEL in .env")
            if status in (401, 403):
                raise LLMUnavailable(f"authentication failed ({last}); check your API key")
            if status == 400 and c.json_mode and "response_format" in err:
                body.pop("response_format", None)  # provider lacks JSON mode; retry plain
                continue
            if status == 429 or status >= 500:
                if "per day" in err.lower() or "tokens per day" in err.lower():
                    raise LLMUnavailable(f"daily quota exhausted ({last})")
                wait = _retry_after(hdrs) or min(2 ** attempt, 20)
                if attempt < c.max_retries:
                    self.sleep(wait)
                    continue
            break
        raise LLMUnavailable(f"request failed: {last}")


def list_models(config: LLMConfig, timeout: float = 20.0) -> list[str]:
    """Ask the provider which model ids this key can use (GET {base_url}/models)."""
    if not config.api_key:
        raise LLMUnavailable("no API key (set OSTUTOR_API_KEY or GROQ_API_KEY)")
    req = urllib.request.Request(
        f"{config.base_url}/models",
        headers={"Authorization": f"Bearer {config.api_key}",
                 "User-Agent": "ostutor/0.2 (+https://github.com/itsmebirdie/ostutor)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise LLMUnavailable(f"HTTP {e.code} listing models") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMUnavailable(f"network error: {e}") from e
    return sorted(m["id"] for m in data.get("data", []))


def _retry_after(headers: dict) -> float | None:
    for k, v in headers.items():
        if k.lower() == "retry-after":
            try:
                return min(float(v), 60.0)
            except ValueError:
                return None
    return None
