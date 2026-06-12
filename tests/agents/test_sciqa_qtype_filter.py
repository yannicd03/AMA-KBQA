"""Tests for the SciQA per-qtype tool filter (allowed_tools_for_qtype)."""

import re
from pathlib import Path

from ama_kbqa.agents.sciqa_agent.agent import (
    CORE_TOOLS,
    QTYPE_TOOL_MAP,
    SciQAAgent,
    allowed_tools_for_qtype,
)


def _server_tool_names() -> set:
    """Extract the actual MCP tool names from sciqa_server.py source."""
    server_path = (
        Path(__file__).resolve().parents[2] / "ama_kbqa" / "server" / "sciqa_server.py"
    )
    source = server_path.read_text(encoding="utf-8")
    return set(re.findall(r"^async def (\w+)\(", source, flags=re.MULTILINE))


def test_all_mapped_tools_exist_on_server():
    """Guard against drift: every filtered-in name must be a real server tool."""
    server_tools = _server_tool_names()
    mapped = set(CORE_TOOLS)
    for extra in QTYPE_TOOL_MAP.values():
        mapped |= extra
    missing = mapped - server_tools
    assert not missing, f"Mapped tools not found in sciqa_server.py: {missing}"


def test_known_qtype_returns_core_plus_extra():
    allowed = allowed_tools_for_qtype("Aggregation")
    assert allowed is not None
    assert CORE_TOOLS <= allowed
    assert "AggregateComparisonValues" in allowed
    assert "DiagnoseComparisonAggregation" in allowed
    # A Factoid-only tool must not leak into Aggregation questions.
    assert "FindCoAuthors" not in allowed


def test_every_mapped_qtype_keeps_journal_and_sparql():
    """The journal and the raw-SPARQL escape hatch stay available everywhere."""
    for qtype in QTYPE_TOOL_MAP:
        allowed = allowed_tools_for_qtype(qtype)
        assert {"ManageJournal", "GetJournalSummary", "RunORKGSPARQL"} <= allowed


def test_general_qtype_allows_all_tools():
    assert allowed_tools_for_qtype("General") is None


def test_unknown_qtype_allows_all_tools():
    assert allowed_tools_for_qtype("Temporal") is None
    assert allowed_tools_for_qtype("") is None
    assert allowed_tools_for_qtype(None) is None


def test_multilabel_qtype_unions_tools():
    allowed = allowed_tools_for_qtype("Factoid\nSuperlative")
    assert allowed is not None
    assert allowed == CORE_TOOLS | QTYPE_TOOL_MAP["Factoid"] | QTYPE_TOOL_MAP["Superlative"]


def test_multilabel_with_comma_separator():
    allowed = allowed_tools_for_qtype("Factoid, Count")
    assert allowed == CORE_TOOLS | QTYPE_TOOL_MAP["Factoid"] | QTYPE_TOOL_MAP["Count"]


def test_multilabel_with_unknown_component_keeps_known():
    allowed = allowed_tools_for_qtype("Factoid\nTemporal")
    assert allowed == CORE_TOOLS | QTYPE_TOOL_MAP["Factoid"]


def test_lowercase_label_is_normalized():
    assert allowed_tools_for_qtype("count") == CORE_TOOLS | QTYPE_TOOL_MAP["Count"]


def test_agent_method_delegates_to_helper():
    assert SciQAAgent._get_allowed_tools_for_qtype(
        None, "Comparison"
    ) == allowed_tools_for_qtype("Comparison")
