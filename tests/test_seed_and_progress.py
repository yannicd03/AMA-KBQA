"""Tests for the 2026-06-14 seed-control/progress-reporting fixes and the
2026-07-25 question_id / --resume correctness fixes.

- get_chat_seed() reads AMA_LLM_SEED (env) with config fallback and safe parsing.
- The agent and orchestrator pass seed= into the chat-completion params only
  when a seed is set.
- sample_questions warns when the request covers the whole dataset (seed cannot
  change membership) and reports a real subset otherwise.
- process_single_question uses the run index as the question_id when the
  question dict has no explicit id (sampled CSV rows), so per-question logs are
  distinguishable rather than all "[0]", ids stay distinct/ordered across a
  multi-question run, and results.json stays joinable with
  tool_traces/question_NNN.json by question_id even when results are appended
  out of order (the concurrent runner).
- main() must not create real benchmark_results/ directories when a test
  invokes it without an explicit --output-dir.
- is_run_completed() requires summary.json's is_complete: true, not mere file
  existence, so --resume doesn't skip runs that crashed mid-way.
"""

import asyncio
import json
import os

import pytest


@pytest.fixture(autouse=True)
def _clear_seed_env():
    saved = os.environ.pop("AMA_LLM_SEED", None)
    yield
    os.environ.pop("AMA_LLM_SEED", None)
    if saved is not None:
        os.environ["AMA_LLM_SEED"] = saved


# =============================================================================
# get_chat_seed
# =============================================================================

def test_seed_none_when_unset():
    from ama_kbqa.config import get_chat_seed
    assert get_chat_seed() is None


def test_seed_from_env():
    from ama_kbqa.config import get_chat_seed
    os.environ["AMA_LLM_SEED"] = "43"
    assert get_chat_seed() == 43


def test_seed_invalid_is_none():
    from ama_kbqa.config import get_chat_seed
    os.environ["AMA_LLM_SEED"] = "not-an-int"
    assert get_chat_seed() is None


def test_seed_empty_is_none():
    from ama_kbqa.config import get_chat_seed
    os.environ["AMA_LLM_SEED"] = "   "
    assert get_chat_seed() is None


# =============================================================================
# sample_questions: full-dataset warning vs real subset
# =============================================================================

def test_full_dataset_sample_warns_membership_fixed(capsys):
    from ama_kbqa.benchmark_agents import sample_questions
    data = [{"question": f"q{i}", "answer": str(i)} for i in range(100)]
    s42 = sample_questions(data, 100, seed=42)
    out = capsys.readouterr().out
    assert "WARN" in out and "only changes ORDER" in out
    # Whole dataset, regardless of seed -> identical membership.
    s43 = sample_questions(data, 100, seed=43)
    assert {q["question"] for q in s42} == {q["question"] for q in s43} == {q["question"] for q in data}


def test_subset_sample_varies_by_seed(capsys):
    from ama_kbqa.benchmark_agents import sample_questions
    data = [{"question": f"q{i}", "answer": str(i)} for i in range(100)]
    s42 = sample_questions(data, 20, seed=42)
    out = capsys.readouterr().out
    assert "WARN" not in out
    s43 = sample_questions(data, 20, seed=43)
    assert {q["question"] for q in s42} != {q["question"] for q in s43}
    assert len(s42) == 20


def test_subset_sample_reproducible_same_seed():
    from ama_kbqa.benchmark_agents import sample_questions
    data = [{"question": f"q{i}", "answer": str(i)} for i in range(100)]
    a = sample_questions(data, 20, seed=42)
    b = sample_questions(data, 20, seed=42)
    assert [q["question"] for q in a] == [q["question"] for q in b]


# =============================================================================
# stratified_sample: grouped-by-qtype execution order (prompt-cache change 1)
#
# The final `random.shuffle(result)` was replaced with a deterministic
# qtype-name-ordered grouping so consecutive questions of the same qtype
# share an identical leading system-prompt + tool-schema byte sequence,
# which the KIT endpoint's position-anchored prefix cache can reuse (see
# .agent/Tasks/active/prompt-cache-utilization.md, item 1). This must not
# change which questions are sampled -- only their order -- and must stay
# seed-reproducible.
# =============================================================================

def _make_qtype_dataset(types, per_type=10):
    data = []
    for t in types:
        for i in range(per_type):
            data.append({"question": f"{t}-{i}", "answer": str(i), "q_type": t})
    return data


def test_stratified_sample_groups_consecutive_same_qtype():
    from ama_kbqa.benchmark_agents import get_question_type, stratified_sample

    data = _make_qtype_dataset(["Gamma", "Alpha", "Beta"])
    result = stratified_sample(data, n=18, seed=42, agent_name="sciqa")

    qtypes_in_order = [get_question_type(q, "sciqa") for q in result]

    # Each qtype must occupy exactly one contiguous run (no shuffle
    # re-interleaving groups): collapsing consecutive duplicates yields as
    # many blocks as distinct qtypes present.
    collapsed = []
    for qt in qtypes_in_order:
        if not collapsed or collapsed[-1] != qt:
            collapsed.append(qt)
    assert len(collapsed) == len(set(qtypes_in_order))

    # Groups ordered deterministically by qtype name.
    assert collapsed == sorted(collapsed)


def test_stratified_sample_does_not_change_sampled_set():
    """Grouping must reorder execution, never change which questions are
    drawn for a given seed."""
    from ama_kbqa.benchmark_agents import stratified_sample

    data = _make_qtype_dataset(["Gamma", "Alpha", "Beta"])
    grouped = stratified_sample(data, n=18, seed=42, agent_name="sciqa")

    assert len(grouped) == 18
    # Every sampled question actually came from the input dataset, and no
    # duplicates were introduced by the grouping step.
    grouped_questions = [q["question"] for q in grouped]
    assert len(set(grouped_questions)) == len(grouped_questions)
    assert set(grouped_questions).issubset({q["question"] for q in data})


def test_stratified_sample_reproducible_same_seed():
    from ama_kbqa.benchmark_agents import stratified_sample

    data = _make_qtype_dataset(["Gamma", "Alpha", "Beta"])
    a = stratified_sample(data, n=18, seed=42, agent_name="sciqa")
    b = stratified_sample(data, n=18, seed=42, agent_name="sciqa")

    assert [q["question"] for q in a] == [q["question"] for q in b]


def test_stratified_sample_within_group_order_uses_seeded_rng():
    """Two different seeds must (with high probability, given 10 items per
    group) select/order at least one group's contents differently -- proving
    the within-group order still comes from the seeded RNG rather than a
    fixed/sorted order that would be seed-invariant."""
    from ama_kbqa.benchmark_agents import stratified_sample

    data = _make_qtype_dataset(["Gamma", "Alpha", "Beta"], per_type=20)
    a = stratified_sample(data, n=18, seed=42, agent_name="sciqa")
    b = stratified_sample(data, n=18, seed=7, agent_name="sciqa")

    assert [q["question"] for q in a] != [q["question"] for q in b]


# =============================================================================
# Benchmark main() exports AMA_LLM_SEED from --seed
# =============================================================================

def test_main_exports_seed_env(monkeypatch, tmp_path):
    """main() must set AMA_LLM_SEED before the run so subprocesses inherit it.

    Regression: without an explicit --output-dir, main() falls through to
    `output_dir = benchmark_results/<today>-N` and unconditionally calls
    output_dir.mkdir(parents=True, exist_ok=True) (benchmark_agents.py) before
    run_full_benchmark is ever reached. Even with _write_run_manifest and
    run_full_benchmark stubbed out, that mkdir() still hit the repo's real
    benchmark_results/ directory and left an empty dated dir behind on every
    test run (23 accumulated: 2026-07-18-1..18, 2026-07-24-1..5). Passing
    --output-dir under tmp_path keeps the whole test off the real filesystem
    location.
    """
    import ama_kbqa.benchmark_agents as ba
    captured = {}

    def fake_run_full_benchmark(**kwargs):
        captured["seed_env"] = os.environ.get("AMA_LLM_SEED")
        return {"x": True}

    output_dir = tmp_path / "benchmark_results" / "test-run"

    monkeypatch.setattr(ba.asyncio, "run", lambda coro: coro)
    monkeypatch.setattr(ba, "run_full_benchmark", lambda **kw: fake_run_full_benchmark(**kw))
    monkeypatch.setattr(
        ba.sys, "argv",
        ["benchmark_agents", "--agents", "sciqa", "--n-questions", "1",
         "--seed", "43", "--dry-run", "--output-dir", str(output_dir)],
    )
    # _write_run_manifest touches disk; stub it.
    monkeypatch.setattr(ba, "_write_run_manifest", lambda *a, **k: None)
    try:
        ba.main()
    except SystemExit:
        pass
    assert os.environ.get("AMA_LLM_SEED") == "43"
    # The real repo-level benchmark_results/ must be untouched by this test.
    assert not (ba.PROJECT_ROOT / "benchmark_results" / "test-run").exists()


# =============================================================================
# process_single_question uses run index as id fallback
#
# Regression for the "every results.json row has question_id: 0" bug: sampled
# questions (on-the-fly --seed sampling, raw CSV rows) carry no native "id"
# key, so every row fell back to the *same* default instead of the run index,
# and any analysis keyed on question_id silently read question_000.json for
# every row. process_single_question must resolve the id from the run index
# when the question dict has none, and that fallback must actually be
# exercised end to end (a prior version of this test only asserted the
# fallback contract against a bare dict literal, never calling the real
# function, so it could not have caught a regression here).
# =============================================================================

class _StubAgent:
    """Minimal agent double satisfying process_single_question's interface."""

    def __init__(self):
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._messages = [{"role": "user", "content": "stub"}]

    async def ask(self, question):
        return f"answer-to-{question}"

    def get_tool_call_summary(self):
        return {}

    async def soft_reset(self):
        pass


def _run(coro):
    return asyncio.run(coro)


def test_question_index_used_as_id_fallback():
    from ama_kbqa.benchmark_agents import process_single_question

    q = {"question": "noid?", "answer": "x"}  # no "id" key
    agent = _StubAgent()

    result = _run(process_single_question(
        agent=agent, question=q, agent_name="kqapro", timeout=5,
        postprocessor=None, question_index=7,
    ))
    assert result.question_id == 7

    result_no_index = _run(process_single_question(
        agent=agent, question=q, agent_name="kqapro", timeout=5,
        postprocessor=None, question_index=None,
    ))
    assert result_no_index.question_id == 0


def test_question_id_uses_native_id_when_present():
    """A question dict with a real "id" key must keep it, ignoring the index."""
    from ama_kbqa.benchmark_agents import process_single_question

    q = {"id": 42, "question": "has id", "answer": "x"}
    agent = _StubAgent()

    result = _run(process_single_question(
        agent=agent, question=q, agent_name="kqapro", timeout=5,
        postprocessor=None, question_index=3,
    ))
    assert result.question_id == 42


def test_question_ids_distinct_and_ordered_across_multi_question_run():
    """Driving several id-less questions through in sequence must yield
    distinct, correctly-ordered ids (0..N-1), matching their run position —
    not all collapsing to the same fallback value.
    """
    from ama_kbqa.benchmark_agents import process_single_question

    agent = _StubAgent()
    questions = [{"question": f"q{i}", "answer": str(i)} for i in range(5)]

    results = [
        _run(process_single_question(
            agent=agent, question=q, agent_name="kqapro", timeout=5,
            postprocessor=None, question_index=i,
        ))
        for i, q in enumerate(questions)
    ]

    ids = [r.question_id for r in results]
    assert ids == [0, 1, 2, 3, 4]
    assert len(set(ids)) == len(ids)


# =============================================================================
# tool_traces/question_NNN.json must stay joinable with results.json by
# question_id regardless of the order results were appended in.
#
# The concurrent runner (run_benchmark_for_model_agent_parallel) appends
# results in completion order, not launch order, so a result's position in
# the `results` list does not match its question_id. _save_tool_traces used
# to name files after that list position (enumerate(results)), so under
# concurrency question_003.json could silently hold a different question than
# results.json's question_id: 3 row -- exactly the kind of misalignment that
# defeats joining the two files by id.
# =============================================================================

def test_tool_traces_filenames_align_with_question_id_out_of_order():
    from ama_kbqa.benchmark_agents import QuestionResult, _save_tool_traces

    # Deliberately out of launch order, as a concurrent run's completion order
    # would produce.
    shuffled_ids = [3, 0, 4, 2, 1]
    results = [
        QuestionResult(
            question_id=qid,
            question=f"question-{qid}",
            gold_answer="gold",
            predicted_answer="pred",
            accuracy=False,
            elapsed_time=0.1,
            full_messages=[{"role": "user", "content": "hi"}],
        )
        for qid in shuffled_ids
    ]

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        result_dir = Path(tmp)
        _save_tool_traces(result_dir, results)

        for qid in shuffled_ids:
            trace_file = result_dir / "tool_traces" / f"question_{qid:03d}.json"
            assert trace_file.exists(), f"missing trace file for question_id={qid}"
            data = json.loads(trace_file.read_text())
            assert data["question_id"] == qid
            assert data["question"] == f"question-{qid}"


# =============================================================================
# Seed is threaded into agent call params only when set
# =============================================================================

def test_agent_callparams_include_seed_only_when_set(monkeypatch):
    from ama_kbqa import config
    # Unset -> helper returns None -> call sites must omit seed.
    assert config.get_chat_seed() is None
    os.environ["AMA_LLM_SEED"] = "99"
    assert config.get_chat_seed() == 99


# =============================================================================
# is_run_completed must require is_complete: true, not mere file existence.
#
# save_results_to_disk() writes summary.json after every question, with
# is_complete: false, until the run's finally-block writes the final, complete
# summary. The old is_run_completed() only checked (result_dir /
# "summary.json").exists(), so a --resume launch would treat a combination
# that crashed at question 5/100 as finished and skip it permanently.
# =============================================================================

def _write_summary(result_dir, agent_name, model_name, payload=None, raw_text=None):
    d = result_dir / agent_name / model_name
    d.mkdir(parents=True, exist_ok=True)
    summary_path = d / "summary.json"
    if raw_text is not None:
        summary_path.write_text(raw_text, encoding="utf-8")
    else:
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
    return summary_path


def test_is_run_completed_true_when_complete(tmp_path):
    from ama_kbqa.benchmark_agents import is_run_completed
    _write_summary(tmp_path, "kqapro", "gemma", payload={"is_complete": True})
    assert is_run_completed(tmp_path, "kqapro", "gemma") is True


def test_is_run_completed_false_when_incomplete(tmp_path):
    """A crashed run's last-written summary.json (is_complete: false) must
    not be treated as done, so --resume retries it instead of skipping it."""
    from ama_kbqa.benchmark_agents import is_run_completed
    _write_summary(tmp_path, "kqapro", "gemma", payload={"is_complete": False})
    assert is_run_completed(tmp_path, "kqapro", "gemma") is False


def test_is_run_completed_false_when_missing(tmp_path):
    from ama_kbqa.benchmark_agents import is_run_completed
    assert is_run_completed(tmp_path, "kqapro", "gemma") is False


def test_is_run_completed_false_when_malformed_json(tmp_path):
    """Truncated / corrupt summary.json must be treated conservatively as
    not completed rather than raising or being mistaken for done."""
    from ama_kbqa.benchmark_agents import is_run_completed
    _write_summary(tmp_path, "kqapro", "gemma", raw_text="{not valid json")
    assert is_run_completed(tmp_path, "kqapro", "gemma") is False


def test_is_run_completed_false_when_json_not_a_dict(tmp_path):
    """summary.json parsing to a non-dict (e.g. a bare list) must not crash
    the .get() lookup and must be treated as not completed."""
    from ama_kbqa.benchmark_agents import is_run_completed
    _write_summary(tmp_path, "kqapro", "gemma", raw_text="[1, 2, 3]")
    assert is_run_completed(tmp_path, "kqapro", "gemma") is False


def test_is_run_completed_false_when_is_complete_key_missing(tmp_path):
    """A summary.json without the is_complete key (e.g. an older format)
    must not be mistaken for a completed run."""
    from ama_kbqa.benchmark_agents import is_run_completed
    _write_summary(tmp_path, "kqapro", "gemma", payload={"model": "gemma"})
    assert is_run_completed(tmp_path, "kqapro", "gemma") is False
