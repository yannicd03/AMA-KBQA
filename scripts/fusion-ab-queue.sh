#!/usr/bin/env bash
# Fusion A/B queue — arms B (rrf) and C (dbsf).
# Design and decision rule: .agent/Tasks/active/retrieval-fusion-ab.md
#
# Sequence:
#   1. wait for the running dense_rerank sweep (arm A) to finish
#   2. deploy origin/dev (ff-only; will not clobber the modified fewshot pool)
#   3. run arm B then arm C over an IDENTICAL pinned question set
#
# Arms are selected by AMA_RETRIEVAL_* env overrides, never by editing
# config.toml, so the deployed config is irrelevant to arm assignment.
#
# Order is deliberate: B (the incumbent rrf) runs first, so the arm we
# hypothesise in favour of (C, dbsf) gets the later and, if the shared KIT
# endpoint degrades overnight, the *less* flattering slot.
#
# CONFOUND WARNING: the orca stack is currently stopped. If it is restarted
# between arms rather than before or after both, B and C are no longer
# comparable (memory pressure costs ~0.92 s/q of tool time).
set -u
cd ~/AMAKBQA-main

IDX=$(seq -s, 0 99)
STAMP=2026-07-26

say() { echo "=== $(date -Is) $* ==="; }

# ---- 1. wait for arm A ------------------------------------------------
say "queue armed; waiting for the dense_rerank sweep to finish"
while true; do
  if grep -q "SWEEP COMPLETE" ~/densererank-sweep.log 2>/dev/null; then
    say "arm A complete, proceeding"
    break
  fi
  if ! pgrep -f "dense-rerank-sweep.sh" >/dev/null 2>&1; then
    say "ABORT: sweep process is gone but no SWEEP COMPLETE marker was written"
    exit 1
  fi
  sleep 120
done

# ---- 2. deploy --------------------------------------------------------
say "deploying origin/dev"
git fetch origin      || { say "ABORT: git fetch failed"; exit 1; }
git merge --ff-only origin/dev || { say "ABORT: ff-only merge failed (local divergence)"; exit 1; }
say "deployed at $(git rev-parse --short HEAD)"

# ---- 3. arms ----------------------------------------------------------
export AMA_RETRIEVAL_HYBRID_ENABLED=true
export AMA_RETRIEVAL_RERANKER_ENABLED=true

run_arm() {  # $1 = fusion method, $2 = arm label
  local fusion="$1" label="$2"
  export AMA_RETRIEVAL_FUSION="$fusion"

  say "ARM $label START (fusion=$fusion)"
  echo "--- effective retrieval env ---"
  env | grep '^AMA_RETRIEVAL_' | sort
  echo "--- memory ---"
  free -g | head -2
  echo "------------------------------"

  .venv/bin/python ama_kbqa/benchmark_agents.py \
    --agents kqapro --models gemma-4-31b-kit \
    --n-questions 500 --seed 42 --question-indices "$IDX" \
    --postprocessing llm_judge --timeout 600 \
    --output-dir "benchmark_results/fusion-${fusion}-kqapro-n100pinned-${STAMP}"
  say "ARM $label kqapro done"

  .venv/bin/python ama_kbqa/benchmark_agents.py \
    --agents sciqa --models gemma-4-31b-kit \
    --n-questions 100 --seed 42 --question-indices "$IDX" \
    --postprocessing llm_judge --timeout 600 \
    --output-dir "benchmark_results/fusion-${fusion}-sciqa-n100pinned-${STAMP}"
  say "ARM $label sciqa done"
}

run_arm rrf  B
run_arm dbsf C

say "FUSION A/B COMPLETE"
