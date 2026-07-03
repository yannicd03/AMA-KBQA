"""Tests for the benchmark CLI's argument wiring and run manifest.

Both ``run_benchmark`` and the generators it constructs talk to the network, so
these tests monkeypatch ``run_benchmark`` and ``WikiKGQADataset.load`` to keep
everything offline and only check the pure wiring: which generator gets built
with which flags, and what ``_write_run_manifest`` persists.

Two live benchmark runs may be writing into benchmark_results/ at the same
time; nothing here touches that directory (manifests are written to tmp_path).
"""

from __future__ import annotations

import argparse
import json

import ama_kbqa.wikikgqa.benchmark as benchmark_mod
from ama_kbqa.wikikgqa.dataset import WikiKGQADataset
from ama_kbqa.wikikgqa.generator import AgentSparqlGenerator, MentionSparqlGenerator


def _patch_offline(monkeypatch):
    """Stub out dataset loading and the actual run so main() never hits the network."""
    monkeypatch.setattr(
        benchmark_mod.WikiKGQADataset,
        "load",
        classmethod(lambda cls, path: WikiKGQADataset(dataset_id="fake", questions=[])),
    )
    captured: dict = {}

    def fake_run_benchmark(dataset, generator, out_dir, limit=None, verbose=True):
        captured["dataset"] = dataset
        captured["generator"] = generator
        captured["out_dir"] = out_dir
        captured["limit"] = limit
        return {}

    monkeypatch.setattr(benchmark_mod, "run_benchmark", fake_run_benchmark)
    return captured


def test_main_passes_ask_votes_to_agent_generator(monkeypatch, tmp_path):
    captured = _patch_offline(monkeypatch)

    rc = benchmark_mod.main(
        ["--data", "unused.json", "--generator", "agent", "--ask-votes", "3", "--out-dir", str(tmp_path)]
    )
    assert rc == 0
    gen = captured["generator"]
    assert isinstance(gen, AgentSparqlGenerator)
    assert gen.ask_votes == 3


def test_main_ask_votes_default_value_is_one(monkeypatch, tmp_path):
    captured = _patch_offline(monkeypatch)

    rc = benchmark_mod.main(["--data", "unused.json", "--generator", "agent", "--out-dir", str(tmp_path)])
    assert rc == 0
    assert captured["generator"].ask_votes == 1


def test_main_ask_votes_ignored_for_mention_generator(monkeypatch, tmp_path):
    # --ask-votes is agent-only; the mention generator has no such knob.
    captured = _patch_offline(monkeypatch)

    rc = benchmark_mod.main(
        ["--data", "unused.json", "--generator", "mention", "--ask-votes", "5", "--out-dir", str(tmp_path)]
    )
    assert rc == 0
    assert isinstance(captured["generator"], MentionSparqlGenerator)
    assert not hasattr(captured["generator"], "ask_votes")


def test_main_writes_run_manifest(monkeypatch, tmp_path):
    _patch_offline(monkeypatch)

    benchmark_mod.main(
        ["--data", "unused.json", "--generator", "agent", "--ask-votes", "2", "--out-dir", str(tmp_path)]
    )
    manifest_path = tmp_path / "run_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["args"]["ask_votes"] == 2
    assert manifest["args"]["generator"] == "agent"
    assert "timestamp" in manifest
    assert "git_commit" in manifest
    assert "endpoint" in manifest


def test_write_run_manifest_creates_out_dir_and_expected_keys(tmp_path):
    out_dir = tmp_path / "nested" / "run"
    args = argparse.Namespace(data="d.json", ask_votes=4, generator="agent")

    benchmark_mod._write_run_manifest(out_dir, args, "https://example.org/sparql")

    manifest_path = out_dir / "run_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["endpoint"] == "https://example.org/sparql"
    assert manifest["args"] == {"data": "d.json", "ask_votes": 4, "generator": "agent"}
    assert "timestamp" in manifest
    assert "git_commit" in manifest  # may be None outside a git checkout, but the key exists
