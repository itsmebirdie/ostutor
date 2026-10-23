"""Streamlit UI.  Run from the repo root:  streamlit run src/ostutor/chat/app.py"""
import os

import streamlit as st

from ostutor.chat.llm import DEFAULT_MODEL, GroqClient, LLMError
from ostutor.chat.session import ChatSession

# The base prompt asks for plain text (terminal). The UI renders markdown, so override.
MARKDOWN_SUFFIX = (
    "\n\nFormatting override: the output is rendered as markdown. You may use short "
    "bullet lists, **bold** for key terms and fenced code blocks. Still no headers or "
    "horizontal rules, and stay within the word limit."
)

st.set_page_config(page_title="OS Tutor", page_icon="🖥️")
st.title("🖥️ OS Tutor")
st.caption("Ask anything about Operating Systems.")

with st.sidebar:
    key = st.text_input("Groq API key", type="password",
                        value=os.environ.get("GROQ_API_KEY", ""),
                        help="Or set GROQ_API_KEY before launching.")
    model = st.text_input("Model", value=os.environ.get("OSTUTOR_MODEL", DEFAULT_MODEL))
    if st.button("New conversation"):
        st.session_state.pop("chat", None)
        st.session_state.pop("messages", None)
        st.rerun()

if not key:
    st.info("Enter your Groq API key in the sidebar to start.")
    st.stop()

# Rebuild the session if key/model changed; history is preserved in `messages`.
sig = (key, model)
if st.session_state.get("sig") != sig or "chat" not in st.session_state:
    st.session_state.chat = ChatSession(GroqClient(api_key=key, model=model),
                                        system_suffix=MARKDOWN_SUFFIX)
    st.session_state.chat.history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.get("messages", [])]
    st.session_state.sig = sig
st.session_state.setdefault("messages", [])

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("e.g. explain deadlocks"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                reply = st.session_state.chat.ask(prompt)
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except LLMError as e:
            st.error(str(e))
            st.session_state.messages.pop()  # drop unanswered user turn