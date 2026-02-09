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


def _read_output(proc):
    """Background thread: read subprocess stdout line by line."""
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            st.session_state.batch_output += line
    except Exception:
        pass
    finally:
        proc.stdout.close()
        proc.wait()
        st.session_state.batch_return_code = proc.returncode
        st.session_state.batch_finished = True
        st.session_state.batch_running = False


# ── Configuration form ───────────────────────────────────────────────────────
with st.form("batch_config"):
    col1, col2 = st.columns(2)

    with col1:
        agent = st.selectbox("Agent", ["kqapro", "sciqa"], index=0)
        n_questions = st.number_input("Sample Size", min_value=1, max_value=500, value=10)
        seed = st.number_input("Seed", min_value=0, value=42)

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

    # Build command
    cmd = [
        sys.executable, "-m",
        f"ama_kbqa.agents.{agent}_agent.batch_runner",
        "--n_questions", str(n_questions),
        "--seed", str(seed),
    ]

    if agent == "kqapro":
        cmd += ["--postprocessing_mode", eval_method]
    else:
        cmd += ["--postprocessing", eval_method]
        if dataset:
            cmd += ["--dataset", dataset]

    if no_fewshot:
        cmd.append("--no-fewshot")

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

    t = threading.Thread(target=_read_output, args=(proc,), daemon=True)
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

# ── Live progress display ────────────────────────────────────────────────────
if st.session_state.batch_running or st.session_state.batch_output:
    output_text = st.session_state.batch_output

    # Parse progress from [N/M] patterns
    progress_matches = re.findall(r'\[(\d+)/(\d+)\]', output_text)
    if progress_matches:
        current, total = int(progress_matches[-1][0]), int(progress_matches[-1][1])
        if total > 0:
            st.progress(current / total, text=f"Question {current}/{total}")

    # Show console output (last ~5000 chars)
    display_text = output_text[-5000:] if len(output_text) > 5000 else output_text
    colored = ansi_to_html(display_text)
    st.markdown(
        f'<div class="console-container">{colored}</div>',
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
                c1.metric("Accuracy", f"{stats.get('accuracy_rate', 0):.0%}")
                c2.metric("Questions", stats.get("total_questions", "?"))
                c3.metric("Avg Duration", f"{stats.get('average_duration_seconds', 0):.1f}s")
                c4.metric("Total Tokens", f"{stats.get('total_tokens_used', 0):,}")
                st.info(f"Results saved as **{latest}**. View details on the Evaluation page.")
            except Exception as e:
                st.warning(f"Could not load results: {e}")
    elif rc == -1:
        st.warning("Batch run was stopped.")
    else:
        st.error(f"Batch run failed (exit code {rc}).")
