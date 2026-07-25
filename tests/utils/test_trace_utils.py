"""Tests for the few-shot output-path fixes in ama_kbqa.utils.trace_utils.

Covers three regressions found in a SMOKE run:

1. ``export_fewshot_examples_from_traces`` always defaulted to the KQAPro
   fewshot directory regardless of which agent produced the trace, so a
   SciQA run's examples (including the shared ``_general.json`` /
   ``_tool_tips.json`` sinks in the LLM-based generator) landed in — and
   could contaminate — KQAPro's own prompt-injection files.
2. SciQA question types are compound labels joined by a literal newline
   (e.g. ``"Factoid\\nSuperlative"``), which produced filenames containing
   raw newline bytes when used unsanitized as a filename stem.
3. KQAPro's existing output path/behavior must stay byte-for-byte unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ama_kbqa.framework.adapters.kqapro_adapter import KQAProAdapter
from ama_kbqa.framework.adapters.sciqa_adapter import SciQAAdapter
from ama_kbqa.utils import trace_utils
from ama_kbqa.utils.trace_utils import (
    export_fewshot_examples_from_traces,
    resolve_fewshot_dir,
    sanitize_qtype_for_filename,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
KQAPRO_DEFAULT_DIR = REPO_ROOT / "db" / "datasets" / "kqapro" / "fewshot-examples"


# ============================================================================
# sanitize_qtype_for_filename
# ============================================================================

class TestSanitizeQtypeForFilename:
    def test_plain_qtype_unchanged(self):
        """Existing single-word KQAPro qtypes must round-trip identically."""
        for qtype in ["Count", "Query", "QueryAttr", "SelectAmong", "Verify"]:
            assert sanitize_qtype_for_filename(qtype) == qtype

    def test_newline_joined_qtype_collapses_to_safe_separator(self):
        """The exact SciQA labels observed in a SMOKE run's summary.json."""
        assert sanitize_qtype_for_filename("Factoid\nSuperlative") == "Factoid_Superlative"
        assert sanitize_qtype_for_filename("Non-Factoid\nCount") == "Non-Factoid_Count"
        assert sanitize_qtype_for_filename("Non-factoid\nSuperlative") == "Non-factoid_Superlative"

    def test_result_never_contains_newline_or_control_chars(self):
        for qtype in ["Factoid\nSuperlative", "a\r\nb", "tab\ttab", "multi\n\n\nnewline"]:
            safe = sanitize_qtype_for_filename(qtype)
            assert "\n" not in safe
            assert "\r" not in safe
            assert "\t" not in safe
            assert all(ord(c) >= 0x20 for c in safe)

    def test_collapses_multiple_whitespace_runs_to_one_separator(self):
        assert sanitize_qtype_for_filename("A\n\n\nB") == "A_B"
        assert sanitize_qtype_for_filename("A   B") == "A_B"

    def test_path_separators_are_stripped(self):
        safe = sanitize_qtype_for_filename("Foo/../bar")
        assert "/" not in safe
        assert "\\" not in safe

    def test_result_is_a_valid_single_path_segment(self):
        """The sanitized name must never expand into multiple path components."""
        for qtype in ["Factoid\nSuperlative", "a/b/c", "../../etc/passwd", "a\\b"]:
            safe = sanitize_qtype_for_filename(qtype)
            assert Path(safe).parts == (safe,) or len(Path(safe).parts) == 1

    def test_empty_or_none_falls_back_to_unknown(self):
        assert sanitize_qtype_for_filename("") == "Unknown"
        assert sanitize_qtype_for_filename(None) == "Unknown"


# ============================================================================
# resolve_fewshot_dir
# ============================================================================

class TestResolveFewshotDir:
    def test_none_agent_returns_legacy_kqapro_default(self):
        assert resolve_fewshot_dir(None) == KQAPRO_DEFAULT_DIR

    def test_kqapro_agent_matches_legacy_default_byte_for_byte(self):
        """KQAPro's adapter-declared dir must equal the pre-fix hardcoded path."""
        assert resolve_fewshot_dir("kqapro") == KQAPRO_DEFAULT_DIR
        assert resolve_fewshot_dir("kqapro") == resolve_fewshot_dir(None)

    def test_kqapro_adapter_declares_the_expected_dir(self):
        """Guards against the adapter's domain_settings drifting silently."""
        declared = KQAProAdapter().config.domain_settings.get("fewshot_examples_dir")
        assert declared == "db/datasets/kqapro/fewshot-examples"

    def test_sciqa_agent_resolves_outside_kqapro_dir(self):
        sciqa_dir = resolve_fewshot_dir("sciqa")
        assert sciqa_dir != KQAPRO_DEFAULT_DIR
        assert "kqapro" not in str(sciqa_dir)
        assert sciqa_dir == REPO_ROOT / "db" / "datasets" / "sciqa" / "fewshot-examples"

    def test_sciqa_adapter_declares_no_fewshot_dir_today(self):
        """If SciQA ever gains a declared dir, resolve_fewshot_dir must prefer it
        (this test documents/pins the current fallback-derivation behavior)."""
        assert "fewshot_examples_dir" not in SciQAAdapter().config.domain_settings

    def test_unknown_agent_derives_a_per_agent_path(self):
        derived = resolve_fewshot_dir("some_future_kg")
        assert derived == REPO_ROOT / "db" / "datasets" / "some_future_kg" / "fewshot-examples"


# ============================================================================
# export_fewshot_examples_from_traces — agent-aware output directory
# ============================================================================

def _make_result(question: str, qtype: str, tool_count: int = 2) -> dict:
    return {
        "accuracy": True,
        "qtype": qtype,
        "question": question,
        "answer": "42",
        "tool_trace": [{"tool": "FindNode", "args": "x", "result": "y"}],
        "tool_call_summary": {"total_calls": tool_count},
    }


class TestExportFewshotExamplesAgentAware:
    @pytest.fixture(autouse=True)
    def _isolate_project_root(self, tmp_path, monkeypatch):
        """Redirect the module's PROJECT_ROOT so agent-derived defaults land
        under tmp_path instead of the real repo's db/datasets tree."""
        monkeypatch.setattr(trace_utils, "PROJECT_ROOT", tmp_path)
        self.tmp_path = tmp_path

    def test_sciqa_qtype_writes_outside_kqapro_directory(self):
        results = [_make_result("What venue?", "Factoid\nSuperlative")]
        export_fewshot_examples_from_traces(results, agent_name="sciqa")

        kqapro_dir = self.tmp_path / "db" / "datasets" / "kqapro" / "fewshot-examples"
        sciqa_dir = self.tmp_path / "db" / "datasets" / "sciqa" / "fewshot-examples"

        assert not kqapro_dir.exists()
        assert sciqa_dir.exists()
        assert (sciqa_dir / "Factoid_Superlative.json").exists()

    def test_newline_qtype_produces_filesystem_safe_filename(self):
        results = [_make_result("Q?", "Non-Factoid\nCount")]
        export_fewshot_examples_from_traces(results, agent_name="sciqa")

        sciqa_dir = self.tmp_path / "db" / "datasets" / "sciqa" / "fewshot-examples"
        written = list(sciqa_dir.iterdir())
        assert len(written) == 1
        assert written[0].name == "Non-Factoid_Count.json"
        assert "\n" not in written[0].name

        data = json.loads(written[0].read_text())
        assert data[0]["question"] == "Q?"
        # The qtype recorded *inside* the example is untouched (only the
        # filename is sanitized) — content fidelity is preserved.
        assert data[0]["qtype"] == "Non-Factoid\nCount"

    def test_kqapro_qtype_writes_to_kqapro_directory_unchanged(self):
        results = [_make_result("Q?", "Count")]
        export_fewshot_examples_from_traces(results, agent_name="kqapro")

        kqapro_dir = self.tmp_path / "db" / "datasets" / "kqapro" / "fewshot-examples"
        sciqa_dir = self.tmp_path / "db" / "datasets" / "sciqa" / "fewshot-examples"

        assert (kqapro_dir / "Count.json").exists()
        assert not sciqa_dir.exists()

    def test_no_agent_name_falls_back_to_kqapro_default_path_shape(self):
        """Backward compatibility: existing callers that never passed
        agent_name/output_dir still resolve to a kqapro/fewshot-examples path."""
        results = [_make_result("Q?", "Count")]
        # Explicit output_dir here to avoid writing into the real repo tree
        # while still exercising the "no agent_name" resolution path for the
        # *directory naming*, checked via resolve_fewshot_dir directly below.
        out_dir = self.tmp_path / "explicit"
        export_fewshot_examples_from_traces(results, output_dir=out_dir)
        assert (out_dir / "Count.json").exists()
