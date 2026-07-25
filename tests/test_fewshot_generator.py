"""Tests for the output-path fixes in ama_kbqa.fewshot_generator.

Companion to tests/utils/test_trace_utils.py — that file covers the
tool-trace exporter (``export_fewshot_examples_from_traces``); this file
covers the LLM-based generator's save helpers and directory resolution
(``_save_qtype_example`` / ``_save_general_example`` / ``_save_tool_tip`` /
``_resolve_agent_fewshot_dir``), which previously always wrote into the
hardcoded KQAPro directory regardless of ``agent_name``, using unsanitized
qtype labels as filenames.
"""
from __future__ import annotations

import json
from pathlib import Path

import ama_kbqa.fewshot_generator as fg

REPO_ROOT = Path(__file__).resolve().parents[1]
KQAPRO_DEFAULT_DIR = REPO_ROOT / "db" / "datasets" / "kqapro" / "fewshot-examples"


def _qtype_example(qtype: str, question: str = "Q?") -> "fg.FewshotQTypeExample":
    return fg.FewshotQTypeExample(
        question=question,
        answer="42",
        qtype=qtype,
        trace=[{"tool": "FindNode", "args": "x", "result": "y"}],
        lesson="lesson",
        pitfall="pitfall",
        tool_count=2,
        was_correct=True,
        collected_at="2026-07-25T00:00:00",
    )


def _general_example(title: str = "Title") -> "fg.FewshotGeneralExample":
    return fg.FewshotGeneralExample(
        title=title,
        guidance="guidance",
        applies_to=["all"],
        derived_from_qtype="Count",
        was_correct=True,
        collected_at="2026-07-25T00:00:00",
    )


def _tool_tip(tool_name: str = "RunSPARQL") -> "fg.ToolTip":
    return fg.ToolTip(
        tool_name=tool_name,
        problem_pattern="pattern",
        guidance="guidance",
        derived_from_question="Q?",
        collected_at="2026-07-25T00:00:00",
    )


class TestSaveHelpersUseExplicitDir:
    """The save helpers must write into the *passed* dir, never a hardcoded one."""

    def test_save_qtype_example_sanitizes_newline_qtype(self, tmp_path):
        example = _qtype_example("Factoid\nSuperlative")
        saved = fg._save_qtype_example(example, tmp_path)
        assert saved is True

        written = list(tmp_path.iterdir())
        assert len(written) == 1
        assert written[0].name == "Factoid_Superlative.json"
        assert "\n" not in written[0].name

        data = json.loads(written[0].read_text())
        assert data[0]["question"] == "Q?"
        assert data[0]["qtype"] == "Factoid\nSuperlative"  # content untouched

    def test_save_qtype_example_plain_qtype_unchanged(self, tmp_path):
        fg._save_qtype_example(_qtype_example("Count"), tmp_path)
        assert (tmp_path / "Count.json").exists()

    def test_save_qtype_example_dedups_by_question(self, tmp_path):
        fg._save_qtype_example(_qtype_example("Count", "Q1"), tmp_path)
        second = fg._save_qtype_example(_qtype_example("Count", "Q1"), tmp_path)
        assert second is False
        data = json.loads((tmp_path / "Count.json").read_text())
        assert len(data) == 1

    def test_save_general_example_writes_to_given_dir(self, tmp_path):
        fg._save_general_example(_general_example(), tmp_path)
        assert (tmp_path / "_general.json").exists()

    def test_save_tool_tip_writes_to_given_dir(self, tmp_path):
        fg._save_tool_tip(_tool_tip(), tmp_path)
        assert (tmp_path / "_tool_tips.json").exists()


class TestResolveAgentFewshotDir:
    def test_kqapro_agent_resolves_to_legacy_default(self):
        assert fg._resolve_agent_fewshot_dir("kqapro") == KQAPRO_DEFAULT_DIR

    def test_sciqa_agent_resolves_outside_kqapro_dir(self):
        sciqa_dir = fg._resolve_agent_fewshot_dir("sciqa")
        assert sciqa_dir != KQAPRO_DEFAULT_DIR
        assert "kqapro" not in str(sciqa_dir)
        assert sciqa_dir == REPO_ROOT / "db" / "datasets" / "sciqa" / "fewshot-examples"

    def test_monkeypatched_fewshot_dir_overrides_any_agent(self, tmp_path, monkeypatch):
        """Mirrors run_fewshot_generator.py's shadow-directory replay mechanism:
        it monkeypatches the module-level FEWSHOT_DIR to redirect writes away
        from live training data. That override must win regardless of
        agent_name, so post-hoc replay stays safe for every agent."""
        shadow_dir = tmp_path / "shadow"
        monkeypatch.setattr(fg, "FEWSHOT_DIR", shadow_dir)

        assert fg._resolve_agent_fewshot_dir("kqapro") == shadow_dir
        assert fg._resolve_agent_fewshot_dir("sciqa") == shadow_dir
        assert fg._resolve_agent_fewshot_dir("anything") == shadow_dir
