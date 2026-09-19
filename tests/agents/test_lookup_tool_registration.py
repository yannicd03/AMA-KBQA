"""Guard against the new lexical-lookup tools being invisible to the agent.

_get_allowed_tools_for_qtype resolves CORE_TOOLS | QTYPE_TOOL_MAP[qtype]; a
tool absent from CORE_TOOLS (and not in every qtype's extra set) is simply
never offered to the model for that question type. Both LookupEntityByName
(KQAPro) and LookupResourceByLabel (SciQA) must be in CORE_TOOLS so they're
available regardless of question type — the whole point is that the agent
can reach for them any time semantic search looks confidently wrong.
"""

import re
from pathlib import Path

from ama_kbqa.agents.kqapro_agent.agent import CORE_TOOLS as KQAPRO_CORE_TOOLS
from ama_kbqa.agents.kqapro_agent.agent import QTYPE_TOOL_MAP as KQAPRO_QTYPE_TOOL_MAP
from ama_kbqa.agents.kqapro_agent.agent import KQAProAgent
from ama_kbqa.agents.sciqa_agent.agent import CORE_TOOLS as SCIQA_CORE_TOOLS
from ama_kbqa.agents.sciqa_agent.agent import allowed_tools_for_qtype
from ama_kbqa.server import kqapro_server as kqa
from ama_kbqa.server import sciqa_server as sci


def _kqapro_server_tool_names() -> set:
    source = (
        Path(__file__).resolve().parents[2] / "ama_kbqa" / "server" / "kqapro_server.py"
    ).read_text(encoding="utf-8")
    return set(re.findall(r"^def (\w+)\(", source, flags=re.MULTILINE))


def test_lookup_entity_by_name_is_a_core_kqapro_tool():
    assert "LookupEntityByName" in KQAPRO_CORE_TOOLS
    assert hasattr(kqa, "LookupEntityByName")


def test_lookup_entity_by_name_survives_every_kqapro_qtype_filter():
    """CORE_TOOLS is unioned into every mapped qtype's allowed set — assert
    this holds for the concrete resolver, not just set membership in the
    raw dict. An unmapped qtype resolves to None ("all tools"), which
    trivially includes it too."""
    agent = KQAProAgent.__new__(KQAProAgent)  # bypass __init__ (no MCP server needed)
    for qtype in KQAPRO_QTYPE_TOOL_MAP:
        allowed = agent._get_allowed_tools_for_qtype(qtype)
        assert allowed is not None
        assert "LookupEntityByName" in allowed

    assert agent._get_allowed_tools_for_qtype("SomeUnmappedType") is None


def test_lookup_entity_by_name_registered_as_real_server_tool():
    """Guard against drift: CORE_TOOLS entries must exist as real server tools."""
    server_tools = _kqapro_server_tool_names()
    assert "LookupEntityByName" in server_tools


def test_lookup_resource_by_label_is_a_core_sciqa_tool():
    assert "LookupResourceByLabel" in SCIQA_CORE_TOOLS
    assert hasattr(sci, "LookupResourceByLabel")


def test_lookup_resource_by_label_survives_every_sciqa_qtype_filter():
    for qtype in ("Factoid", "Count", "List", "Boolean", "Comparison", "Superlative", "Aggregation", "General"):
        allowed = allowed_tools_for_qtype(qtype)
        if allowed is None:  # "General"/unmapped intentionally means "all tools"
            continue
        assert "LookupResourceByLabel" in allowed
