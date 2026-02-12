"""Batch Processing page - configure and run batch benchmarks."""

import subprocess
import sys
import re
import threading
import time
from pathlib import Path

import streamlit as st

from ama_kbqa.frontend.utils.styling import inject_css, ansi_to_html
from ama_kbqa.frontend.utils.batch_results_loader import list_batches, load_summary

inject_css()

st.title("Batch Processing")

# ── Session state init ───────────────────────────────────────────────────────
if "batch_running" not in st.session_state:
    st.session_state.batch_running = False
if "batch_output" not in st.session_state:
    st.session_state.batch_output = ""
if "batch_proc" not in st.session_state:
    st.session_state.batch_proc = None
if "batch_thread" not in st.session_state:
    st.session_state.batch_thread = None
if "batch_finished" not in st.session_state:
    st.session_state.batch_finished = False
if "batch_return_code" not in st.session_state:
    st.session_state.batch_return_code = None

# Thread-safe shared dict (avoids accessing st.session_state from bg thread)
if "batch_shared" not in st.session_state:
    st.session_state.batch_shared = {"output": "", "return_code": None, "finished": False}


def _read_output(proc, shared):
    """Background thread: read subprocess stdout line by line into shared dict."""
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            shared["output"] += line
    except Exception:
        pass
    finally:
        proc.stdout.close()
        proc.wait()
        shared["return_code"] = proc.returncode
        shared["finished"] = True


# ── Configuration form ───────────────────────────────────────────────────────
with st.form("batch_config"):
    col1, col2 = st.columns(2)

    with col1:
        agent = st.selectbox("Agent", ["kqapro", "sciqa"], index=0)
        n_questions = st.number_input("Sample Size", min_value=1, max_value=500, value=10)
        seed = st.number_input("Seed", min_value=0, value=42)
        stratified = st.checkbox("Stratified Sampling")

    with col2:
        # Evaluation methods differ per agent
        if agent == "kqapro":
            eval_options = ["choice", "sparql", "llm_judge"]
        else:
            eval_options = ["llm_judge", "simple"]
        eval_method = st.selectbox("Evaluation Method", eval_options)

        if agent == "sciqa":
            dataset = st.selectbox("Dataset", ["handcrafted", "auto"])
        else:
            dataset = None

        no_fewshot = st.checkbox("Disable Few-Shot Examples")
        questionnaire = st.text_input("Questionnaire File (optional)", value="",
                                      help="Path to a pre-generated questionnaire JSON file")

    submitted = st.form_submit_button(
        "Start Batch Run",
        disabled=st.session_state.batch_running,
        type="primary",
    )

# ── Launch subprocess ────────────────────────────────────────────────────────
if submitted and not st.session_state.batch_running:
    st.session_state.batch_output = ""
    st.session_state.batch_finished = False
    st.session_state.batch_return_code = None

    # Build command using unified benchmark_agents module
    cmd = [
        sys.executable, "-m",
        "ama_kbqa.benchmark_agents",
        "--agents", agent,
        "--n-questions", str(n_questions),
        "--seed", str(seed),
        "--postprocessing", eval_method,
    ]

    if agent == "sciqa" and dataset:
        cmd += ["--dataset", dataset]

    if no_fewshot:
        cmd.append("--no-fewshot")

    if stratified:
        cmd.append("--stratified")

    if questionnaire.strip():
        cmd += ["--questionnaire", questionnaire.strip()]

    repo_root = Path(__file__).resolve().parents[3]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(repo_root),
    )

    st.session_state.batch_proc = proc
    st.session_state.batch_running = True
    st.session_state.batch_shared = {"output": "", "return_code": None, "finished": False}

    t = threading.Thread(
        target=_read_output,
        args=(proc, st.session_state.batch_shared),
        daemon=True,
    )
    t.start()
    st.session_state.batch_thread = t

    st.rerun()

# ── Stop button ──────────────────────────────────────────────────────────────
if st.session_state.batch_running:
    if st.button("Stop Batch Run", type="secondary"):
        proc = st.session_state.batch_proc
        if proc and proc.poll() is None:
            proc.terminate()
            st.session_state.batch_running = False
            st.session_state.batch_finished = True
            st.session_state.batch_return_code = -1
            st.warning("Batch run terminated by user.")

# ── Sync shared dict → session state ────────────────────────────────────────
shared = st.session_state.batch_shared
if shared["output"]:
    st.session_state.batch_output = shared["output"]
if shared["finished"] and st.session_state.batch_running:
    st.session_state.batch_return_code = shared["return_code"]
    st.session_state.batch_finished = True
    st.session_state.batch_running = False


def _resolve_cr(text: str) -> str:
    """Simulate carriage-return behaviour: for each line, keep only the
    content after the last \\r so tqdm updates collapse to the latest value."""
    out_lines = []
    for line in text.split("\n"):
        parts = line.split("\r")
        # Keep the last non-empty segment (the most recent overwrite)
        resolved = ""
        for part in parts:
            if part:
                resolved = part
        out_lines.append(resolved)
    return "\n".join(out_lines)


# ── Live progress display ────────────────────────────────────────────────────
if st.session_state.batch_running or st.session_state.batch_output:
    output_text = st.session_state.batch_output

    # Parse tqdm progress: look for percentage pattern like " 30%|" or fraction "5/10"
    # tqdm outputs lines like: " 30%|███       | 3/10 [00:15<00:35, ...]"
    tqdm_pct = re.findall(r'(\d+)%\|', output_text)
    tqdm_frac = re.findall(r'\|\s*(\d+)/(\d+)\s*\[', output_text)
    if tqdm_frac:
        current, total = int(tqdm_frac[-1][0]), int(tqdm_frac[-1][1])
        if total > 0:
            st.progress(current / total, text=f"Question {current}/{total}")
    elif tqdm_pct:
        pct = int(tqdm_pct[-1])
        st.progress(min(pct / 100.0, 1.0), text=f"{pct}%")

    # Resolve \r (tqdm overwrites) and show console output (last ~5000 chars)
    cleaned = _resolve_cr(output_text)
    display_text = cleaned[-5000:] if len(cleaned) > 5000 else cleaned
    colored = ansi_to_html(display_text)
    # Auto-scroll script keeps the console pinned to the bottom
    scroll_js = (
        '<script>var c=document.querySelector(".console-container");'
        "if(c)c.scrollTop=c.scrollHeight;</script>"
    )
    st.markdown(
        f'<div class="console-container">{colored}</div>{scroll_js}',
        unsafe_allow_html=True,
    )

    # Auto-refresh while running
    if st.session_state.batch_running:
        time.sleep(2)
        st.rerun()

# ── Completion display ───────────────────────────────────────────────────────
if st.session_state.batch_finished and not st.session_state.batch_running:
    rc = st.session_state.batch_return_code
    if rc == 0:
        st.success("Batch run completed successfully!")
        # Load latest results
        batches = list_batches()
        if batches:
            latest = batches[0]
            try:
                summary = load_summary(latest)
                stats = summary.get("statistics", {})
                c1, c2, c3, c4 = st.columns(4)
                acc = stats.get("accuracy_rate") or stats.get("accuracy", 0)
                c1.metric("Accuracy", f"{acc:.0%}")
                c2.metric("Questions", stats.get("total_questions", "?"))
                avg_dur = stats.get("average_duration_seconds") or stats.get("avg_time_seconds", 0)
                c3.metric("Avg Duration", f"{avg_dur:.1f}s")
                tok = stats.get("total_tokens_used") or stats.get("total_tokens", 0)
                c4.metric("Total Tokens", f"{tok:,}")
                label = latest.get("label", latest) if isinstance(latest, dict) else latest
                st.info(f"Results saved as **{label}**. View details on the Evaluation page.")
            except Exception as e:
                st.warning(f"Could not load results: {e}")
    elif rc == -1:
        st.warning("Batch run was stopped.")
    else:
        st.error(f"Batch run failed (exit code {rc}).")
