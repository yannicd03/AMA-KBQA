"""Synthetic fast-path tool calls must use ids every provider accepts.

Mistral's chat template (KIT's vLLM) keeps only the last 9 characters of a
tool-call id and rejects anything but [a-zA-Z0-9]; the old
"fast_path_<n>_<tool>" id failed there with a 400 ("_FindNode").
"""

import re
from types import SimpleNamespace

from ama_kbqa.framework.base_agent import BaseKBQAAgent

_MISTRAL_ID = re.compile(r"^[A-Za-z0-9]{9}$")


def _stub():
    return SimpleNamespace(_text_tool_call_mode=False, _messages=[], tool_call_durations=[])


def test_fast_path_tool_call_id_is_mistral_compatible():
    agent = _stub()
    BaseKBQAAgent._append_fast_path_tool_messages(agent, "FindNode", {"name": "Inception"}, "{}")

    call, result = agent._messages
    call_id = call["tool_calls"][0]["id"]
    assert _MISTRAL_ID.match(call_id), call_id
    assert result["tool_call_id"] == call_id


def test_fast_path_tool_call_ids_are_unique():
    agent = _stub()
    for _ in range(5):
        BaseKBQAAgent._append_fast_path_tool_messages(agent, "FindNode", {}, "{}")

    ids = [m["tool_calls"][0]["id"] for m in agent._messages if m["role"] == "assistant"]
    assert len(set(ids)) == len(ids)
