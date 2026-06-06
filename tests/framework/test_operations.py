"""Tests for the abstract atomic operation contract and adapter bindings."""

import re
from pathlib import Path

from ama_kbqa.framework.operations import (
    ATOMIC_OPERATIONS,
    validate_bindings,
)
from ama_kbqa.framework.adapters.kqapro_adapter import KQAProAdapter
from ama_kbqa.framework.adapters.sciqa_adapter import SciQAAdapter

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestOperationRegistry:
    """The contract itself: registry shape and validation logic."""

    REQUIRED = {
        "find_entity", "get_label", "get_labels", "get_summary",
        "get_relation_targets", "reverse_lookup", "follow_path",
        "compare", "count", "verify_numeric", "run_sparql",
    }
    OPTIONAL = {"aggregate", "frequent_values", "select_extreme"}

    def test_registry_contains_core_operations(self):
        assert self.REQUIRED | self.OPTIONAL == set(ATOMIC_OPERATIONS)

    def test_two_level_contract(self):
        """Required/optional split must match the declared tiers."""
        required = {n for n, op in ATOMIC_OPERATIONS.items() if op.required}
        optional = {n for n, op in ATOMIC_OPERATIONS.items() if not op.required}
        assert required == self.REQUIRED
        assert optional == self.OPTIONAL

    def test_optional_ops_not_reported_missing(self):
        """A binding map covering only required ops is a valid contract."""
        bindings = {name: "X" for name in self.REQUIRED}
        report = validate_bindings(bindings)
        assert report.ok
        assert report.missing == ()

    def test_registry_kinds_are_valid(self):
        for op in ATOMIC_OPERATIONS.values():
            assert op.kind in ("atomic", "deterministic", "escape_hatch")
            assert op.name
            assert op.description
            assert op.canonical_params

    def test_validate_bindings_complete(self):
        bindings = {name: f"Tool{i}" for i, name in enumerate(ATOMIC_OPERATIONS)}
        report = validate_bindings(bindings)
        assert report.ok
        assert report.missing == ()
        assert report.unknown == ()

    def test_validate_bindings_detects_missing(self):
        bindings = {"get_label": "GetNodeLabel"}
        report = validate_bindings(bindings)
        assert not report.ok
        assert "find_entity" in report.missing
        assert "count" in report.missing

    def test_validate_bindings_detects_unknown(self):
        bindings = {name: "X" for name in ATOMIC_OPERATIONS}
        bindings["teleport_entity"] = "Teleport"
        report = validate_bindings(bindings)
        assert not report.ok
        assert report.unknown == ("teleport_entity",)


class TestAdapterCoverage:
    """Both shipped adapters must fully cover the contract."""

    def test_kqapro_covers_contract(self):
        report = KQAProAdapter().validate_operation_coverage()
        assert report.ok, f"missing={report.missing}, unknown={report.unknown}"

    def test_sciqa_covers_contract(self):
        report = SciQAAdapter().validate_operation_coverage()
        assert report.ok, f"missing={report.missing}, unknown={report.unknown}"


class TestBindingsPointToRealTools:
    """Every bound tool name must exist as a function def in its MCP server.

    Checked at source level (regex on the server file) so the test does not
    need the servers' runtime dependencies.
    """

    @staticmethod
    def _defined_tools(server_file: str) -> set:
        source = (REPO_ROOT / "ama_kbqa" / "server" / server_file).read_text()
        return set(re.findall(r"^(?:async )?def ([A-Z]\w+)\(", source, re.MULTILINE))

    def test_kqapro_bindings_exist_in_server(self):
        tools = self._defined_tools("kqapro_server.py")
        for op, tool_name in KQAProAdapter().get_operation_bindings().items():
            assert tool_name in tools, f"{op} bound to unknown tool {tool_name}"

    def test_sciqa_bindings_exist_in_server(self):
        tools = self._defined_tools("sciqa_server.py")
        for op, tool_name in SciQAAdapter().get_operation_bindings().items():
            assert tool_name in tools, f"{op} bound to unknown tool {tool_name}"
