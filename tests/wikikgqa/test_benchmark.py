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
from ama_kbqa.wikikgqa.dataset import WikiKGQADataset, WikiKGQAQuestion
from ama_kbqa.wikikgqa.endpoint import SparqlResult
from ama_kbqa.wikikgqa.generator import AgentSparqlGenerator, GeneratedQuery, MentionSparqlGenerator


def _patch_offline(monkeypatch):
    """Stub out dataset loading and the actual run so main() never hits the network."""
    monkeypatch.setattr(
        benchmark_mod.WikiKGQADataset,
        "load",
        classmethod(lambda cls, path: WikiKGQADataset(dataset_id="fake", questions=[])),
    )
    captured: dict = {}

    def fake_run_benchmark(dataset, generator, out_dir, limit=None, verbose=True, auto_escalate=True):
        captured["dataset"] = dataset
        captured["generator"] = generator
        captured["out_dir"] = out_dir
        captured["limit"] = limit
        captured["auto_escalate"] = auto_escalate
        return {}

    monkeypatch.setattr(benchmark_mod, "run_benchmark", fake_run_benchmark)
    return captured


def test_main_passes_votes_to_agent_generator(monkeypatch, tmp_path):
    captured = _patch_offline(monkeypatch)

    rc = benchmark_mod.main(
        ["--data", "unused.json", "--generator", "agent", "--votes", "3", "--out-dir", str(tmp_path)]
    )
    assert rc == 0
    gen = captured["generator"]
    assert isinstance(gen, AgentSparqlGenerator)
    assert gen.votes == 3


def test_main_votes_default_is_one(monkeypatch, tmp_path):
    captured = _patch_offline(monkeypatch)

    rc = benchmark_mod.main(["--data", "unused.json", "--generator", "agent", "--out-dir", str(tmp_path)])
    assert rc == 0
    assert captured["generator"].votes == 1


def test_main_votes_ignored_for_mention_generator(monkeypatch, tmp_path):
    # --votes is agent-only; the mention generator has no such knob.
    captured = _patch_offline(monkeypatch)

    rc = benchmark_mod.main(
        ["--data", "unused.json", "--generator", "mention", "--votes", "5", "--out-dir", str(tmp_path)]
    )
    assert rc == 0
    assert isinstance(captured["generator"], MentionSparqlGenerator)
    assert not hasattr(captured["generator"], "votes")


def test_main_writes_run_manifest(monkeypatch, tmp_path):
    _patch_offline(monkeypatch)

    benchmark_mod.main(
        ["--data", "unused.json", "--generator", "agent", "--votes", "2", "--out-dir", str(tmp_path)]
    )
    manifest_path = tmp_path / "run_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["args"]["votes"] == 2
    assert manifest["args"]["generator"] == "agent"
    assert "timestamp" in manifest
    assert "git_commit" in manifest
    assert "endpoint" in manifest


def test_write_run_manifest_creates_out_dir_and_expected_keys(tmp_path):
    out_dir = tmp_path / "nested" / "run"
    args = argparse.Namespace(data="d.json", votes=4, generator="agent")

    benchmark_mod._write_run_manifest(out_dir, args, "https://example.org/sparql")

    manifest_path = out_dir / "run_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["endpoint"] == "https://example.org/sparql"
    assert manifest["args"] == {"data": "d.json", "votes": 4, "generator": "agent"}
    assert "timestamp" in manifest
    assert "git_commit" in manifest  # may be None outside a git checkout, but the key exists


def test_main_auto_escalate_default_on(monkeypatch, tmp_path):
    captured = _patch_offline(monkeypatch)

    rc = benchmark_mod.main(["--data", "unused.json", "--generator", "agent", "--out-dir", str(tmp_path)])
    assert rc == 0
    assert captured["auto_escalate"] is True


def test_main_no_auto_escalate_flag_disables(monkeypatch, tmp_path):
    captured = _patch_offline(monkeypatch)

    rc = benchmark_mod.main(
        ["--data", "unused.json", "--generator", "agent", "--no-auto-escalate", "--out-dir", str(tmp_path)]
    )
    assert rc == 0
    assert captured["auto_escalate"] is False


# --- empty-answer escalation (run_benchmark's post-pass re-run) -----------------
#
# Every gold answer in this benchmark is non-empty, so an empty submission answer
# is known-wrong (see scripts/assemble_voted_submission.py's --patch policy, which
# _escalate_empty_answers automates). These exercise run_benchmark directly with a
# scripted AgentSparqlGenerator subclass so no network/LLM call is ever made.


def _question(qid: int, text: str = "Some question?") -> WikiKGQAQuestion:
    return WikiKGQAQuestion(id=qid, questions={"en": text}, mentions=[])


def _dataset(*qids: int) -> WikiKGQADataset:
    return WikiKGQADataset(dataset_id="fake", questions=[_question(qid) for qid in qids])


def _empty_result() -> SparqlResult:
    return SparqlResult(ok=True, json={"head": {"vars": ["x"]}, "results": {"bindings": []}})


def _filled_result(value: str = "42") -> SparqlResult:
    return SparqlResult(
        ok=True,
        json={"head": {"vars": ["x"]}, "results": {"bindings": [{"x": {"type": "literal", "value": value}}]}},
    )


class _ScriptedAgentGenerator(AgentSparqlGenerator):
    """AgentSparqlGenerator stub: returns scripted SparqlResults per qid, consumed
    in call order (main pass first, escalation re-run second), and records every
    ``generate()`` call as ``(qid, tool_budget at call time)`` -- no network/LLM
    call is ever made. isinstance(AgentSparqlGenerator) still holds, so the real
    escalation path in benchmark.py is exercised unmodified.
    """

    def __init__(self, script: dict[int, list[SparqlResult]], **kwargs):
        super().__init__(**kwargs)
        self._script = {qid: list(results) for qid, results in script.items()}
        self.calls: list[tuple[int, int]] = []

    def generate(self, question: WikiKGQAQuestion) -> GeneratedQuery:
        self.calls.append((question.id, self.tool_budget))
        queue = self._script.get(question.id, [])
        result = queue.pop(0) if queue else _empty_result()
        return GeneratedQuery(qid=question.id, sparql="SELECT ?x WHERE {}", result=result)


def test_escalation_reruns_empty_answer_once_with_raised_budget(tmp_path):
    gen = _ScriptedAgentGenerator({1: [_empty_result(), _filled_result()]}, tool_budget=20)

    benchmark_mod.run_benchmark(_dataset(1), gen, out_dir=tmp_path, verbose=False)

    # Main pass at the configured budget, exactly one escalation re-run at +15.
    assert gen.calls == [(1, 20), (1, 35)]
    assert gen.tool_budget == 20  # restored after the escalation pass


def test_escalation_patches_nonempty_result_into_submission(tmp_path):
    gen = _ScriptedAgentGenerator({1: [_empty_result(), _filled_result("42")]}, tool_budget=20)

    benchmark_mod.run_benchmark(_dataset(1), gen, out_dir=tmp_path, verbose=False)

    submission = json.loads((tmp_path / "submission.json").read_text(encoding="utf-8"))
    assert submission["questions"][0]["answers"] == ["42"]
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["escalated_qids"] == [1]
    assert summary["escalated_filled_qids"] == [1]
    assert summary["escalated_still_empty_qids"] == []


def test_escalation_still_empty_stays_and_is_logged(tmp_path, capsys):
    gen = _ScriptedAgentGenerator({1: [_empty_result(), _empty_result()]}, tool_budget=20)

    benchmark_mod.run_benchmark(_dataset(1), gen, out_dir=tmp_path, verbose=True)

    submission = json.loads((tmp_path / "submission.json").read_text(encoding="utf-8"))
    assert submission["questions"][0]["answers"] == []
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["escalated_qids"] == [1]
    assert summary["escalated_filled_qids"] == []
    assert summary["escalated_still_empty_qids"] == [1]
    out = capsys.readouterr().out
    assert "still empty" in out
    assert "q1" in out


def test_escalation_never_reruns_already_nonempty_answers(tmp_path):
    gen = _ScriptedAgentGenerator(
        {1: [_filled_result("7")], 2: [_empty_result(), _filled_result("8")]}, tool_budget=20,
    )

    benchmark_mod.run_benchmark(_dataset(1, 2), gen, out_dir=tmp_path, verbose=False)

    # q1 was non-empty on the first pass: exactly one call, never escalated.
    assert [c for c in gen.calls if c[0] == 1] == [(1, 20)]
    # q2 was empty: main pass + one escalation call at the raised budget.
    assert [c for c in gen.calls if c[0] == 2] == [(2, 20), (2, 35)]
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["escalated_qids"] == [2]


def test_no_auto_escalate_flag_disables_the_rerun(tmp_path):
    gen = _ScriptedAgentGenerator({1: [_empty_result(), _filled_result()]}, tool_budget=20)

    benchmark_mod.run_benchmark(_dataset(1), gen, out_dir=tmp_path, verbose=False, auto_escalate=False)

    assert gen.calls == [(1, 20)]  # no escalation call made
    submission = json.loads((tmp_path / "submission.json").read_text(encoding="utf-8"))
    assert submission["questions"][0]["answers"] == []
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "escalated_qids" not in summary  # additive key absent: shape unchanged


def test_escalation_skipped_for_non_agent_mention_generator(tmp_path):
    """The mention (non-agent) generator path is unaffected: run_benchmark only
    escalates isinstance(AgentSparqlGenerator) generators, so a mention-style
    generator (no ``tool_budget`` attribute at all) must never be re-run even
    when its answer is empty."""
    calls: list[int] = []

    class _FakeMentionGenerator:
        def generate(self, question):
            calls.append(question.id)
            return GeneratedQuery(qid=question.id, sparql=None, result=None)

    benchmark_mod.run_benchmark(_dataset(1), _FakeMentionGenerator(), out_dir=tmp_path, verbose=False)

    assert calls == [1]  # exactly one call: no escalation attempted
    submission = json.loads((tmp_path / "submission.json").read_text(encoding="utf-8"))
    assert submission["questions"][0]["answers"] == []
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "escalated_qids" not in summary
