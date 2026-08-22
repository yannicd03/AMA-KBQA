"""Phase 0 prefix-cache gate: reproduce the cold/warm/grown TTFT protocol from
.agent/Tasks/active/prompt-cache-utilization.md §1, once via the raw `openai`
client (what BaseKBQAAgent uses today) and once via the LangGraph/ChatOpenAI
path, side by side, against the KIT endpoint only.

3 trials x 3 cases (cold / warm-repeat / grown) x 2 paths = 18 requests total
(well under the 30-request budget). Uses the same real 15-tool, 4-message
request body captured by scripts/langgraph_spike.py (db/kqapro_questionnaire.json
question id=1) as the base prompt, so the gate measures the SAME shape of
request the agent actually sends, not a synthetic filler prompt.

Usage:
    uv run python scripts/prefix_cache_gate.py

Writes:
    <scratch>/spike/cache_gate.md
    <scratch>/spike/cache_gate_raw.json   (raw per-trial numbers)

If KIT_API_KEY is empty or chat_provider != "kit", this reports that and exits
without sending any request (no fabricated numbers).
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(
    "/tmp/claude-1000/-home-yannic-code-AMAKBQA/3259b8c1-7636-4a66-93dc-2b2da2449a37/scratchpad"
)
SPIKE_DIR = SCRATCH / "spike"
SPIKE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

N_TRIALS = 3
MAX_TOKENS = 300  # tool_choice="required" -> short tool-call completion; bounds decode time.


def _preflight() -> Tuple[bool, str]:
    from ama_kbqa.config import load_config

    config = load_config()
    provider = config["llm"]["chat_provider"]
    if provider != "kit":
        return False, f"[llm].chat_provider is {provider!r}, not 'kit' — skipping (KIT-only gate)."
    key = os.environ.get("KIT_API_KEY", "")
    if not key.strip():
        return False, "KIT_API_KEY is empty/unset — skipping (no fabricated numbers)."
    return True, ""


def _load_base() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    path = SPIKE_DIR / "old_request.json"
    if not path.exists():
        raise RuntimeError(
            f"{path} not found — run scripts/langgraph_spike.py first "
            "(this gate reuses its captured real request body as the base prompt)."
        )
    with open(path) as f:
        d = json.load(f)
    return d["messages"], d["tools"], d.get("tool_choice", "required")


def _variant_messages(base_messages: List[Dict[str, Any]], nonce: str) -> List[Dict[str, Any]]:
    """Deep-copy base_messages with a unique nonce prepended to the system
    message (position 0 = the most cache-defeating spot, per
    prompt-cache-utilization.md §1 Test B, so this reliably produces a
    genuinely cold prefix each time)."""
    msgs = json.loads(json.dumps(base_messages))
    msgs[0] = dict(msgs[0])
    msgs[0]["content"] = f"[gate-nonce:{nonce}]\n" + msgs[0]["content"]
    return msgs


def _grown_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Same prefix + one extra assistant tool-call / tool-result pair,
    mirroring the shape of the real agent loop growing by one iteration."""
    grown = list(msgs)
    call_id = f"gate_{uuid.uuid4().hex[:8]}"
    grown.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "GetNodeLabel",
                        "arguments": json.dumps({"node_id": "Q1"}),
                    },
                }
            ],
        }
    )
    grown.append(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": "GetNodeLabel",
            "content": "Q1: synthetic placeholder label (cache-gate filler, not a real lookup).",
        }
    )
    return grown


# ---------------------------------------------------------------------------
# RAW path (openai SDK directly — what BaseKBQAAgent._llm_call uses)
# ---------------------------------------------------------------------------

def _build_raw_client():
    from ama_kbqa.config import _create_client, load_config

    config = load_config()
    provider = config["llm"]["chat_provider"]
    return _create_client(provider, model_type="chat", max_retries=0)


def _raw_ttft(client, model: str, messages, tools, tool_choice) -> Tuple[Optional[float], Optional[int]]:
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        max_tokens=MAX_TOKENS,
        temperature=1.0,
        timeout=90.0,
        stream=True,
        stream_options={"include_usage": True},
    )
    first: Optional[float] = None
    prompt_tokens: Optional[int] = None
    for chunk in stream:
        if first is None:
            first = time.perf_counter()
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            prompt_tokens = usage.prompt_tokens
    ttft = (first - t0) if first is not None else None
    return ttft, prompt_tokens


# ---------------------------------------------------------------------------
# LangGraph/ChatOpenAI path
# ---------------------------------------------------------------------------

def _build_lc_model_with_tools(tools, tool_choice):
    import httpx
    from langchain_openai import ChatOpenAI

    from ama_kbqa.config import load_config, get_chat_model_name

    config = load_config()
    kit_cfg = config["kit"]
    model = ChatOpenAI(
        model=get_chat_model_name(),
        base_url=kit_cfg["base_url"],
        api_key=os.environ["KIT_API_KEY"],
        timeout=httpx.Timeout(connect=20.0, read=60.0, write=10.0, pool=5.0),
        max_retries=0,
        temperature=1.0,
        max_tokens=MAX_TOKENS,
    )
    return model.bind_tools(tools, tool_choice=tool_choice)


def _to_lc_messages(messages: List[Dict[str, Any]]):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    lc_msgs = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            lc_msgs.append(SystemMessage(content=m.get("content") or ""))
        elif role == "user":
            lc_msgs.append(HumanMessage(content=m.get("content") or ""))
        elif role == "assistant":
            tool_calls = [
                {
                    "name": tc["function"]["name"],
                    "args": json.loads(tc["function"]["arguments"]),
                    "id": tc["id"],
                }
                for tc in (m.get("tool_calls") or [])
            ]
            lc_msgs.append(AIMessage(content=m.get("content") or "", tool_calls=tool_calls))
        elif role == "tool":
            lc_msgs.append(
                ToolMessage(content=m.get("content") or "", tool_call_id=m.get("tool_call_id", ""))
            )
    return lc_msgs


def _lc_ttft(model_with_tools, messages) -> Tuple[Optional[float], Optional[int]]:
    lc_msgs = _to_lc_messages(messages)
    t0 = time.perf_counter()
    first: Optional[float] = None
    prompt_tokens: Optional[int] = None
    for chunk in model_with_tools.stream(lc_msgs):
        if first is None:
            first = time.perf_counter()
        um = getattr(chunk, "usage_metadata", None)
        if um:
            prompt_tokens = um.get("input_tokens")
    ttft = (first - t0) if first is not None else None
    return ttft, prompt_tokens


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _median(xs: List[Optional[float]]) -> Optional[float]:
    xs2 = sorted(x for x in xs if x is not None)
    if not xs2:
        return None
    n = len(xs2)
    mid = n // 2
    return xs2[mid] if n % 2 else (xs2[mid - 1] + xs2[mid]) / 2


def main() -> None:
    ok, reason = _preflight()
    if not ok:
        print(f"SKIPPING cache gate: {reason}")
        with open(SPIKE_DIR / "cache_gate.md", "w") as f:
            f.write(f"# Prefix-cache gate (KIT)\n\nSKIPPED: {reason}\n")
        return

    base_messages, tools, tool_choice = _load_base()
    model_name = None
    from ama_kbqa.config import get_chat_model_name

    model_name = get_chat_model_name()

    raw_client = _build_raw_client()
    lc_model_with_tools = _build_lc_model_with_tools(tools, tool_choice)

    results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "raw": {"cold": [], "warm": [], "grown": []},
        "langgraph": {"cold": [], "warm": [], "grown": []},
    }

    request_count = 0
    for path_name in ("raw", "langgraph"):
        for i in range(N_TRIALS):
            nonce = f"{path_name}-{i}-{uuid.uuid4().hex[:8]}"
            cold_msgs = _variant_messages(base_messages, nonce)

            def call(msgs):
                nonlocal request_count
                request_count += 1
                if path_name == "raw":
                    return _raw_ttft(raw_client, model_name, msgs, tools, tool_choice)
                return _lc_ttft(lc_model_with_tools, msgs)

            print(f"[{path_name}] trial {i}: cold ...")
            ttft, pt = call(cold_msgs)
            results[path_name]["cold"].append({"ttft": ttft, "prompt_tokens": pt})
            print(f"  ttft={ttft} prompt_tokens={pt}")

            print(f"[{path_name}] trial {i}: warm (immediate repeat) ...")
            ttft, pt = call(cold_msgs)
            results[path_name]["warm"].append({"ttft": ttft, "prompt_tokens": pt})
            print(f"  ttft={ttft} prompt_tokens={pt}")

            grown_msgs = _grown_messages(cold_msgs)
            print(f"[{path_name}] trial {i}: grown ...")
            ttft, pt = call(grown_msgs)
            results[path_name]["grown"].append({"ttft": ttft, "prompt_tokens": pt})
            print(f"  ttft={ttft} prompt_tokens={pt}")

    print(f"\nTotal requests sent: {request_count}")

    with open(SPIKE_DIR / "cache_gate_raw.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = ["# Prefix-cache gate (KIT endpoint), raw openai client vs LangGraph/ChatOpenAI\n"]
    lines.append(f"Model: `{model_name}` | trials per case: {N_TRIALS} | total requests: {request_count}\n")
    lines.append(
        "Base prompt: the real captured 15-tool, 4-message QueryAttr request body "
        "(`scratchpad/spike/old_request.json`), byte-identical across both paths. "
        "Each COLD trial prepends a unique nonce to the system message (position 0 — "
        "the most cache-defeating spot). WARM immediately repeats that exact prompt. "
        "GROWN appends one assistant tool-call + tool-result pair to the warmed prompt.\n"
    )

    for path_name in ("raw", "langgraph"):
        lines.append(f"\n## {path_name}\n")
        lines.append("| case | trial | ttft (s) | prompt_tokens |")
        lines.append("|---|---|---|---|")
        for case in ("cold", "warm", "grown"):
            for i, row in enumerate(results[path_name][case]):
                ttft_s = f"{row['ttft']:.2f}" if row["ttft"] is not None else "n/a"
                pt_s = row["prompt_tokens"] if row["prompt_tokens"] is not None else "n/a"
                lines.append(f"| {case} | {i} | {ttft_s} | {pt_s} |")
        cold_med = _median([r["ttft"] for r in results[path_name]["cold"]])
        warm_med = _median([r["ttft"] for r in results[path_name]["warm"]])
        grown_med = _median([r["ttft"] for r in results[path_name]["grown"]])
        ratio = (warm_med / cold_med) if (cold_med and warm_med) else None
        grown_ratio = (grown_med / cold_med) if (cold_med and grown_med) else None
        lines.append(
            f"\nmedian TTFT: cold={cold_med:.2f}s warm={warm_med:.2f}s grown={grown_med:.2f}s"
            if all(x is not None for x in (cold_med, warm_med, grown_med))
            else "\nmedian TTFT: insufficient data"
        )
        if ratio is not None:
            lines.append(f"warm/cold ratio: {ratio:.2f}")
        if grown_ratio is not None:
            lines.append(f"grown/cold ratio: {grown_ratio:.2f}")

    # Side-by-side comparison
    lines.append("\n## raw vs langgraph, side by side (median TTFT, seconds)\n")
    lines.append("| case | raw | langgraph |")
    lines.append("|---|---|---|")
    for case in ("cold", "warm", "grown"):
        raw_med = _median([r["ttft"] for r in results["raw"][case]])
        lc_med = _median([r["ttft"] for r in results["langgraph"][case]])
        raw_s = f"{raw_med:.2f}" if raw_med is not None else "n/a"
        lc_s = f"{lc_med:.2f}" if lc_med is not None else "n/a"
        lines.append(f"| {case} | {raw_s} | {lc_s} |")

    report = "\n".join(lines) + "\n"
    with open(SPIKE_DIR / "cache_gate.md", "w") as f:
        f.write(report)

    print("\n" + report)
    print(f"Wrote: {SPIKE_DIR / 'cache_gate.md'}")
    print(f"Wrote: {SPIKE_DIR / 'cache_gate_raw.json'}")


if __name__ == "__main__":
    main()
