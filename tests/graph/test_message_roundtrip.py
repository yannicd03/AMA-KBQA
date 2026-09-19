"""Message round-trip: OpenAI-format dicts -> LangChain messages -> back.

``ama_kbqa.graph.messages`` converts ``BaseKBQAAgent._messages`` in at the
start of a graph run and back out at the end. The graph engine only holds
LangChain messages transiently in state; ``self._messages`` stays the
canonical history every other consumer (harness, frontend, Phase 2
compaction/journal code) reads. If the conversion is lossy, that canonical
history silently drifts every time the graph engine runs.
"""

from __future__ import annotations

from ama_kbqa.graph.messages import from_lc_messages, to_lc_messages


def _sample_messages() -> list[dict]:
    return [
        {"role": "system", "content": "You are a KBQA agent."},
        {"role": "user", "content": "Who directed Inception?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "FindNode",
                        "arguments": '{"semantic_node_name": "Inception"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "FindNode",
            "content": '{"matches": [{"original_id": "Q1", "name": "Inception"}]}',
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "GetRelationDetails",
                        "arguments": '{"base_node_id": "Q1", "relation_name": "director"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "name": "GetRelationDetails",
            "content": '{"triples": [{"related_id": "Q2"}]}',
        },
        {"role": "assistant", "content": "Christopher Nolan directed Inception."},
    ]


def test_round_trip_is_lossless_for_a_full_tool_loop_transcript():
    original = _sample_messages()
    lc_messages = to_lc_messages(original)
    restored = from_lc_messages(lc_messages)
    assert restored == original


def test_round_trip_preserves_plain_system_user_only_stack():
    original = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    assert from_lc_messages(to_lc_messages(original)) == original


def test_round_trip_preserves_assistant_final_answer_with_no_tool_calls():
    original = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "The answer is 42."},
    ]
    assert from_lc_messages(to_lc_messages(original)) == original


def test_assistant_tool_call_only_message_keeps_content_none_not_empty_string():
    """Regression guard for the one real shape mismatch: LangChain's own
    convert_to_openai_messages() renders a tool-calls-only assistant message
    with content=="" — this repo's dicts (and the raw OpenAI response they
    come from) use content=None for that case."""
    original = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "FindNode", "arguments": "{}"},
                }
            ],
        },
    ]
    restored = from_lc_messages(to_lc_messages(original))
    assert restored == original
    assert restored[0]["content"] is None
