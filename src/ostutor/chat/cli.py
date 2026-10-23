"""Interactive chat:  python -m ostutor.chat.cli"""
from __future__ import annotations

from .llm import GroqClient, LLMError
from .session import ChatSession

HELP = "Commands: /reset clears the conversation, /quit exits."


def main() -> int:
    try:
        client = GroqClient()
    except LLMError as e:
        print(e)
        return 1
    session = ChatSession(client)
    print(f"OS Tutor ({client.model}). Ask me anything about Operating Systems.\n{HELP}\n")
    while True:
        try:
            msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not msg:
            continue
        if msg in ("/quit", "/exit"):
            return 0
        if msg == "/reset":
            session.reset()
            print("(conversation cleared)")
            continue
        try:
            print(f"\ntutor> {session.ask(msg)}\n")
        except LLMError as e:
            print(f"\n[error] {e}\n")


if __name__ == "__main__":
    raise SystemExit(main())