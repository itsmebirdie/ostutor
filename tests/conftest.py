"""Test-suite guard rails: tests never use the network or a real API key."""
import socket

import pytest

SECRET_VARS = ("GROQ_API_KEY", "OSTUTOR_API_KEY", "OSTUTOR_BASE_URL", "OSTUTOR_MODEL")


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    for var in SECRET_VARS:
        monkeypatch.delenv(var, raising=False)

    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
