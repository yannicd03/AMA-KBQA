"""Conversion between the agent's canonical OpenAI-format message dicts
(``BaseKBQAAgent._messages``) and LangChain ``BaseMessage`` objects (graph
state's ``messages`` key).

``self._messages`` stays the canonical history read by the benchmark
harness, the frontend, and (in Phase 2) compaction/journal code — see
``langgraph-rewrite.md`` Architecture decisions, "Messages". The graph engine
converts in at the start of a run and back out at the end; nothing in
between the two conversions touches ``self._messages`` directly.

Not used for text-mode agents (``framework/text_tool_calls.py``): those store
tool calls as ``<tool_call>...</tool_call>`` text inside ``role=user``/
``role=assistant`` messages, which ``convert_to_messages`` has no concept of.
``BaseKBQAAgent._run_tool_loop`` falls back to the legacy loop entirely when
``self._text_tool_call_mode`` is true, regardless of the configured engine.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AnyMessage, convert_to_messages
from langchain_core.messages.utils import convert_to_openai_messages


def to_lc_messages(messages: List[Dict[str, Any]]) -> List[AnyMessage]:
    """Convert OpenAI-format message dicts into LangChain ``BaseMessage``s."""
    return convert_to_messages(messages)


def from_lc_messages(messages: List[AnyMessage]) -> List[Dict[str, Any]]:
    """Inverse of :func:`to_lc_messages`.

    ``convert_to_openai_messages`` round-trips every message shape this repo
    produces EXCEPT one: an assistant message that carries only
    ``tool_calls`` (no text) comes back with ``content == ""``, where this
    repo's dicts (and the raw OpenAI API response they were built from) use
    ``content: None`` for that case. Restore it here so
    ``convert_to_openai_messages(convert_to_messages(msgs)) == msgs`` holds
    for every message shape ``BaseKBQAAgent`` produces — see
    ``tests/graph/test_message_roundtrip.py``.
    """
    openai_messages = convert_to_openai_messages(messages)
    for msg in openai_messages:
        if (
            msg.get("role") == "assistant"
            and msg.get("tool_calls")
            and msg.get("content") == ""
        ):
            msg["content"] = None
    return openai_messages
