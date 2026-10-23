from __future__ import annotations

from .context import build_context
from .prompts import CONTEXT_TEMPLATE, SYSTEM_PROMPT


class ChatSession:
    """Holds conversation history. Grounding context is attached to the model
    call only, never stored, so history stays small and clean."""

    def __init__(self, client, max_turns: int = 10, context_builder=build_context):
        self.client = client
        self.max_turns = max_turns
        self.context_builder = context_builder
        self.history: list[dict] = []

    def ask(self, user_message: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += self.history[-2 * self.max_turns:]
        ctx = self.context_builder(user_message)
        if ctx:
            topic, body = ctx
            messages.append({"role": "system",
                             "content": CONTEXT_TEMPLATE.format(topic=topic, body=body)})
        messages.append({"role": "user", "content": user_message})

        reply = self.client.complete(messages)
        self.history += [{"role": "user", "content": user_message},
                         {"role": "assistant", "content": reply}]
        return reply

    def reset(self):
        self.history.clear()