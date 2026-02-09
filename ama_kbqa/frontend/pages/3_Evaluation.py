"""Evaluation Dashboard - view batch run results, charts, and per-question details."""

import pandas as pd
import streamlit as st

from ama_kbqa.frontend.utils.styling import inject_css
from ama_kbqa.frontend.utils.batch_results_loader import (
    list_batches,
    load_judgments,
    load_results,
    load_summary,
)

inject_css()

st.title("Evaluation Dashboard")

# ── Batch selector ───────────────────────────────────────────────────────────
batches = list_batches()

if not batches:
    st.info("No batch results found. Run a batch from the Batch Processing page first.")
    st.stop()

selected_batch = st.selectbox("Select batch run", batches, index=0)

# ── Load data ────────────────────────────────────────────────────────────────
summary = load_summary(selected_batch)
results = load_results(selected_batch)
judgments_data = load_judgments(selected_batch)

stats = summary.get("statistics", {})
config = summary.get("config", {})

# ── Config info ──────────────────────────────────────────────────────────────
with st.expander("Batch Configuration", expanded=False):
    cfg_cols = st.columns(3)
    cfg_cols[0].markdown(f"**Seed:** {config.get('seed', '?')}")
    cfg_cols[1].markdown(f"**Postprocessing:** {config.get('postprocessing_mode', '?')}")
    cfg_cols[2].markdown(f"**Timestamp:** {summary.get('timestamp', '?')[:19]}")

# ── Summary cards ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{stats.get('accuracy_rate', 0):.0%}")
c2.metric("Total Questions", stats.get("total_questions", 0))
c3.metric("Avg Duration", f"{stats.get('average_duration_seconds', 0):.1f}s")
c4.metric("Total Tokens", f"{stats.get('total_tokens_used', 0):,}")

# ── Charts ───────────────────────────────────────────────────────────────────
st.markdown("---")

chart_col1, chart_col2 = st.columns(2)

# Accuracy by question type
acc_by_type = stats.get("accuracy_by_question_type", {})
if acc_by_type:
    with chart_col1:
        st.markdown("##### Accuracy by Question Type")
        df_acc = pd.DataFrame(
            {"Question Type": list(acc_by_type.keys()), "Accuracy": list(acc_by_type.values())}
        )
        st.bar_chart(df_acc, x="Question Type", y="Accuracy", height=300)

# Question type distribution
type_dist = stats.get("question_type_distribution", {})
if type_dist:
    with chart_col2:
        st.markdown("##### Question Type Distribution")
        df_dist = pd.DataFrame(
            {"Question Type": list(type_dist.keys()), "Count": list(type_dist.values())}
        )
        st.bar_chart(df_dist, x="Question Type", y="Count", height=300)

# Duration distribution & tool call breakdown (from per-question results)
if results:
    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.markdown("##### Duration Distribution")
        durations = []
        for r in results:
            dur_str = r.get("duration", "0s")
            try:
                durations.append(float(str(dur_str).rstrip("s")))
            except (ValueError, TypeError):
                pass
        if durations:
            df_dur = pd.DataFrame({"Duration (s)": durations})
            st.bar_chart(df_dur, height=300)

    # Tool call breakdown from results
    with chart_col4:
        st.markdown("##### Tokens per Question")
        tokens = [r.get("tokens_used", 0) for r in results if r.get("tokens_used")]
        questions_short = [
            (r.get("question", "")[:40] + "...") if len(r.get("question", "")) > 40
            else r.get("question", "")
            for r in results if r.get("tokens_used")
        ]
        if tokens:
            df_tok = pd.DataFrame({"Question": questions_short, "Tokens": tokens})
            st.bar_chart(df_tok, x="Question", y="Tokens", height=300)

# ── Per-question results table ───────────────────────────────────────────────
st.markdown("---")
st.markdown("### Per-Question Results")

if results:
    table_data = []
    for r in results:
        table_data.append({
            "Question": r.get("question", "")[:80],
            "Gold Answer": str(r.get("answer", ""))[:60],
            "Predicted": str(r.get("predicted_answer", ""))[:60],
            "Correct": "✅" if r.get("accuracy") else "❌",
            "Type": r.get("qtype", ""),
            "Duration": r.get("duration", ""),
            "Tokens": r.get("tokens_used", 0),
        })

    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True, hide_index=True)
else:
    st.info("No per-question results available for this batch.")

# ── Expandable judgments ─────────────────────────────────────────────────────
if judgments_data and "judgments" in judgments_data:
    st.markdown("---")
    st.markdown("### LLM Judge Evaluations")

    judge_stats = judgments_data.get("statistics", {})
    if judge_stats:
        jc1, jc2 = st.columns(2)
        jc1.metric("Avg Argumentation Score", f"{judge_stats.get('average_argumentation_score', 0):.2f}/5")
        jc2.metric("Judged Questions", judge_stats.get("total_judgments", 0))

    for j in judgments_data["judgments"]:
        judgment = j.get("judgment", {})
        icon = "✅" if judgment.get("is_correct") else "❌"
        score = judgment.get("argumentation_score", "?")
        label = f'{icon} {j.get("question", "")[:70]}... (Score: {score}/5)'

        with st.expander(label, expanded=False):
            st.markdown(f"**Gold:** {j.get('gold_answer', '')}")
            st.markdown(f"**Predicted:** {j.get('predicted_answer', '')[:200]}")
            st.markdown(f"**Correct:** {judgment.get('is_correct', '?')}")
            st.markdown(f"**Score:** {score}/5")
            st.markdown(f"**Reasoning:** {judgment.get('correctness_reasoning', '')}")
            st.markdown(f"**Argumentation Quality:** {judgment.get('argumentation_quality', '')}")
            if judgment.get("suggested_improvement"):
                st.markdown(f"**Suggestion:** {judgment['suggested_improvement']}")

# ── Failed questions ─────────────────────────────────────────────────────────
failed = summary.get("failed_questions", [])
if failed:
    st.markdown("---")
    st.markdown("### Failed Questions")
    for fq in failed:
        with st.expander(f"❌ {fq.get('question', '')[:80]}", expanded=False):
            st.code(fq.get("error", "Unknown error"))
