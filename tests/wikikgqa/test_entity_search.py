"""Tests for the gated entity-linking tool (without-mentions track).

The SearchEntities tool must be ABSENT on with-mentions runs (lean tool list) and
PRESENT only when the ``entity_search`` config flag sets WIKIKGQA_ENTITY_SEARCH.
The live Wikidata search API is stubbed so these stay offline.
"""

from __future__ import annotations

import asyncio
import importlib
import os

import ama_kbqa.server.wikidata_server as server


def _reload_with_flag(value: str | None):
    """Reload the server module with WIKIKGQA_ENTITY_SEARCH set/unset."""
    if value is None:
        os.environ.pop("WIKIKGQA_ENTITY_SEARCH", None)
    else:
        os.environ["WIKIKGQA_ENTITY_SEARCH"] = value
    return importlib.reload(server)


def _tool_names(mod) -> set[str]:
    return set(asyncio.run(mod.mcp.get_tools()))


def test_tool_absent_when_flag_off():
    mod = _reload_with_flag(None)
    assert mod._ENTITY_SEARCH_ENABLED is False
    assert "SearchEntities" not in _tool_names(mod)


def test_tool_absent_for_falsey_flag():
    mod = _reload_with_flag("0")
    assert mod._ENTITY_SEARCH_ENABLED is False
    assert "SearchEntities" not in _tool_names(mod)


def test_tool_present_when_flag_on():
    mod = _reload_with_flag("1")
    assert mod._ENTITY_SEARCH_ENABLED is True
    assert "SearchEntities" in _tool_names(mod)
    _reload_with_flag(None)  # restore lean default for other tests


def test_search_formats_candidates(monkeypatch):
    mod = _reload_with_flag("1")
    monkeypatch.setattr(
        mod, "_wbsearch",
        lambda term, kind, lang: [
            {"id": "Q42", "label": "Douglas Adams", "description": "writer"},
            {"id": "Q28421831", "label": "Douglas Adams", "description": "engineer"},
        ],
    )
    out = mod.SearchEntities("Douglas Adams", None, type="item", language="en")
    assert "Q42" in out and "writer" in out
    assert "Q28421831" in out  # multiple candidates surfaced for disambiguation
    _reload_with_flag(None)


def test_search_empty_is_actionable(monkeypatch):
    mod = _reload_with_flag("1")
    monkeypatch.setattr(mod, "_wbsearch", lambda term, kind, lang: [])
    out = mod.SearchEntities("zzzznotathing", None, type="item", language="en")
    assert "No item candidates" in out
    _reload_with_flag(None)


def test_agent_sets_env_flag_and_injects_guidance():
    """The entity_search config parameter must set the env flag and add linking guidance."""
    from ama_kbqa.agents.wikidata_agent.prompts import ENTITY_SEARCH_GUIDANCE

    # Construct only the prompt-relevant state without a full agent init/MCP spawn.
    class _Stub:
        _conventions = "minimal"
        _entity_search = True
        _get_system_prompt = (
            importlib.import_module("ama_kbqa.agents.wikidata_agent.agent")
            .WikidataAgent._get_system_prompt
        )

    prompt = _Stub._get_system_prompt(_Stub())
    assert ENTITY_SEARCH_GUIDANCE.strip()[:30] in prompt
    assert "SearchEntities" in prompt
