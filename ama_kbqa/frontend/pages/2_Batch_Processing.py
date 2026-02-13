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

# ── Available models (mirrors BENCHMARK_MODELS in benchmark_agents.py) ──────
AVAILABLE_MODELS = [
    "minimax-m2.1",
    "minimax-m2.5",
    "glm-4.7",
    "kimi-k2.5",
    "deepseek-v3.2",
    "gpt-oss-120b",
    "qwen3-32b",
    "nemotron-3-nano-30b",
    "gpt-oss-120b-kit",
    "qwen3-vl-235b-kit",
    "gpt-4.1-mini-kit",
]

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
if "batch_start_time" not in st.session_state:
    st.session_state.batch_start_time = None

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
        stratified = st.checkbox("Stratified Sampling", value=True)

    with col2:
        # Evaluation methods differ per agent
        if agent == "kqapro":
            eval_options = ["llm_judge", "choice", "sparql"]
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

    # ── Multi-Model Selection ────────────────────────────────────────────────
    models = st.multiselect(
        "Models (multi-model mode)",
        options=AVAILABLE_MODELS,
        default=[],
        help="Select models to benchmark. Leave empty to use the model from config.toml (single-model mode).",
    )

    # ── Advanced Options ─────────────────────────────────────────────────────
    with st.expander("Advanced Options"):
        adv1, adv2 = st.columns(2)
        with adv1:
            timeout = st.number_input(
                "Timeout per question (seconds)",
                min_value=10,
                max_value=3600,
                value=300,
                help="Maximum time allowed for each question before it's marked as timed out.",
            )
            output_dir = st.text_input(
                "Output Directory (optional)",
                value="",
                help="Custom output path. Leave blank for benchmark_results/<YYYY-MM-DD-N>.",
            )
        with adv2:
            resume = st.checkbox(
                "Resume",
                help="Skip model/agent combinations that already have results in the output directory.",
            )
            export_csv = st.checkbox(
                "Export CSV",
                help="Generate a CSV summary alongside the JSON results.",
            )
            dry_run = st.checkbox(
                "Dry Run",
                help="Preview what would run without actually executing any benchmarks.",
            )

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

    if models:
        cmd += ["--models"] + models

    if timeout != 300:
        cmd += ["--timeout", str(timeout)]

    if output_dir.strip():
        cmd += ["--output-dir", output_dir.strip()]

    if resume:
        cmd.append("--resume")

    if export_csv:
        cmd.append("--export-csv")

    if dry_run:
        cmd.append("--dry-run")

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
    st.session_state.batch_start_time = time.time()
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

    def _fmt_duration(seconds: float) -> str:
        """Format seconds into a human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        m, s = divmod(int(seconds), 60)
        if m < 60:
            return f"{m}m {s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m"

    elapsed = time.time() - st.session_state.batch_start_time if st.session_state.batch_start_time else 0

    if tqdm_frac:
        current, total = int(tqdm_frac[-1][0]), int(tqdm_frac[-1][1])
        if total > 0:
            progress_text = f"Question {current}/{total}"
            if current > 0 and elapsed > 0:
                avg_per_q = elapsed / current
                remaining = avg_per_q * (total - current)
                progress_text += f"  |  Elapsed: {_fmt_duration(elapsed)}  |  ETA: ~{_fmt_duration(remaining)}"
            st.progress(current / total, text=progress_text)
    elif tqdm_pct:
        pct = int(tqdm_pct[-1])
        progress_text = f"{pct}%"
        if pct > 0 and elapsed > 0:
            remaining = elapsed * (100 - pct) / pct
            progress_text += f"  |  Elapsed: {_fmt_duration(elapsed)}  |  ETA: ~{_fmt_duration(remaining)}"
        st.progress(min(pct / 100.0, 1.0), text=progress_text)

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
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                acc = stats.get("accuracy_rate") or stats.get("accuracy", 0)
                c1.metric("Accuracy", f"{acc:.0%}")
                c2.metric("Questions", stats.get("total_questions", "?"))
                avg_dur = stats.get("average_duration_seconds") or stats.get("avg_time_seconds", 0)
                c3.metric("Avg Duration", f"{avg_dur:.1f}s")
                avg_tok = stats.get("avg_tokens") or 0
                c4.metric("Avg Tokens", f"{avg_tok:,.0f}")
                total_cost = stats.get("estimated_cost_usd", 0)
                c5.metric("Total Cost", f"${total_cost:.4f}")
                n_q = stats.get("total_questions", 1) or 1
                c6.metric("Avg Cost", f"${total_cost / n_q:.4f}")
                label = latest.get("label", latest) if isinstance(latest, dict) else latest
                st.info(f"Results saved as **{label}**. View details on the Evaluation page.")
            except Exception as e:
                st.warning(f"Could not load results: {e}")
    elif rc == -1:
        st.warning("Batch run was stopped.")
    else:
        st.error(f"Batch run failed (exit code {rc}).")
