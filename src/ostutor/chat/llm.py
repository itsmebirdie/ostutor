"""Minimal Groq client (OpenAI-compatible endpoint). Standard library only."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"


class LLMError(RuntimeError):
    pass


class GroqClient:
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 timeout: float = 60.0, transport=None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key and transport is None:
            raise LLMError("GROQ_API_KEY is not set. Export it or put it in your environment.")
        self.model = model or os.environ.get("OSTUTOR_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        self._transport = transport or self._http  # injectable for tests

    def complete(self, messages: list[dict], temperature: float = 0.4,
                 max_tokens: int = 700) -> str:
        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        data = self._transport(payload)
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected response from Groq: {data!r}") from e

    def _http(self, payload: dict) -> dict:
        req = urllib.request.Request(
            GROQ_URL, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}",
                     "User-Agent": "ostutor/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise LLMError(f"Groq HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"Could not reach Groq: {e.reason}") from e