"""Phase 0 go/no-go spike: prove a minimal plain-LangGraph tool loop sends the
KIT chat-completions endpoint a request that is byte-equivalent (tools array +
messages + top-level params) to what the current ``BaseKBQAAgent``/``KQAProAgent``
sends for the same question.

See .agent/Tasks/active/langgraph-rewrite.md Section 5, Phase 0.

Usage:
    uv run python scripts/langgraph_spike.py

Writes:
    <scratch>/spike/old_request.json   - raw body of the OLD path's first tool-loop call
    <scratch>/spike/new_request.json   - raw body of the NEW (LangGraph) path's first call
    <scratch>/spike/request_diff.md    - structured diff + assessment

Makes exactly ONE real network call against the configured chat provider (the
classification call in the OLD path). Both paths' tool-loop/model LLM calls
are captured by swapping the underlying httpx transport for a mock that
records the request body and returns a synthetic no-tool-call completion
instead of hitting the network, so no tokens are spent on either "first
tool-loop call" and neither path's retry logic gets involved.

Finding recorded here rather than re-derived at every read: the installed
`openai` SDK (bumped to 3.3.1 by langchain-openai's dependency floor; see
pyproject.toml/uv.lock) vendors its OWN httpx fork as `httpx2`
(`openai/_base_client.py: import httpx2`), not the top-level `httpx` package.
Both the OLD path's raw OpenAI client and the NEW path's `ChatOpenAI` (which
also goes through `openai.OpenAI` internally) are therefore `httpx2.Client`
instances under `.  _client`, so transport-swapping/event-hook code must use
`httpx2.MockTransport`/`httpx2.Response`, not `httpx.MockTransport`. This is
one of the "LangChain/dependency quirks" flagged in the Phase 0 report.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(
    "/tmp/claude-1000/-home-yannic-code-AMAKBQA/3259b8c1-7636-4a66-93dc-2b2da2449a37/scratchpad"
)
SPIKE_DIR = SCRATCH / "spike"
SPIKE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)  # config.py resolves config.toml relative to REPO_ROOT anyway,
# but MCP server subprocess launch / relative paths in the codebase assume cwd=repo root.

QUESTION_ID = 1  # "The person born as Chloe Ardelia Wofford has which KLfG ... designation?"
# db/kqapro_questionnaire.json: program's last function is QueryAttr, i.e. the
# gold-derived qtype that should classify into the agent's "QueryAttr" bucket.


import httpx2  # the openai SDK's internal httpx fork — see module docstring.  # noqa: E402

_FAKE_COMPLETION = {
    "id": "spike-fake-completion",
    "object": "chat.completion",
    "created": 0,
    "model": "spike-fake-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "SPIKE_CAPTURE_STOP", "tool_calls": None},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def _install_capture_transport(openai_httpx_client, store: Dict[str, Any], key: str, skip_if=None):
    """Swap ``openai_httpx_client``'s transport for a mock that captures the
    JSON body of the first matching chat-completions request into
    ``store[key]`` and answers it with a synthetic no-tool-call completion
    instead of hitting the network. Every other request (e.g. the
    classification call, which ``skip_if`` identifies) is forwarded to the
    real transport unchanged, so it still gets a real response.

    Using a mock response (rather than raising from an event hook) matters:
    raising gets wrapped by the openai SDK into a generic
    ``APIConnectionError("Connection error.")``, whose message matches
    ``chatkit.retry.TRANSIENT_MARKERS`` ("connection error") — so
    ``TransientRetry`` retries it, and the retry's underlying request is NOT
    re-intercepted (this repo's ``_llm_call`` retry wraps the whole
    ``client.chat.completions.create`` call), so it goes out for real. A
    synthetic 200 response avoids that failure mode entirely: both paths
    finish the call normally (no tool_calls -> loop ends on iteration 1).
    """
    real_transport = openai_httpx_client._transport

    def handler(request: httpx2.Request) -> httpx2.Response:
        path = str(request.url.path)
        if path.endswith("/chat/completions"):
            try:
                body = json.loads(request.content.decode("utf-8"))
            except Exception:
                body = None
            if body is not None and (skip_if is None or not skip_if(body)):
                # Capture only the FIRST such call's body, but fake-answer
                # every one (not just the first) — the OLD path's
                # zero-tool-call retry (config.toml [agent]
                # zero_tool_call_retry_max) would otherwise re-issue a real
                # network call once our synthetic zero-tool-call response
                # comes back, since only the byte-captured call is special.
                if key not in store:
                    store[key] = body
                return httpx2.Response(200, json=_FAKE_COMPLETION)
        return real_transport.handle_request(request)

    openai_httpx_client._transport = httpx2.MockTransport(handler)


def _load_question(question_id: int) -> str:
    with open(REPO_ROOT / "db" / "kqapro_questionnaire.json") as f:
        data = json.load(f)
    for q in data["questions"]:
        if q["id"] == question_id:
            return q["question"]
    raise ValueError(f"question id {question_id} not found")


# ---------------------------------------------------------------------------
# OLD PATH: real KQAProAgent.ask(), fast path disabled, first tool-loop call
# captured via an httpx hook on the agent's real OpenAI client.
# ---------------------------------------------------------------------------

async def capture_old_request(question: str) -> Dict[str, Any]:
    from ama_kbqa.agents.kqapro_agent.agent import KQAProAgent

    agent = KQAProAgent(name="spike_old", session_id="spike_old")

    # Force the full tool loop to run (skip the fast path) so there IS a
    # tool-loop LLM call to capture, regardless of what the classifier's qtype
    # ends up being for this question.
    async def _no_fast_path(*_args, **_kwargs):
        return None

    agent._try_fast_path = _no_fast_path  # type: ignore[method-assign]

    # Record the qtype actually used for tool filtering (whatever the real
    # classifier produced), so the NEW path can filter tools identically.
    captured_qtype: Dict[str, str] = {}
    orig_allowed_tools = agent._get_allowed_tools_for_qtype

    def _wrapped_allowed_tools(qtype: str):
        captured_qtype["qtype"] = qtype
        return orig_allowed_tools(qtype)

    agent._get_allowed_tools_for_qtype = _wrapped_allowed_tools  # type: ignore[method-assign]

    store: Dict[str, Any] = {}
    skip_classify = lambda body: "response_format" in body  # noqa: E731
    _install_capture_transport(agent.client._client, store, "old_request", skip_if=skip_classify)

    try:
        answer = await agent.ask(question)
        print(f"[old] ask() returned normally: {answer!r}")
    except Exception as e:  # pragma: no cover - diagnostic escape hatch
        if "old_request" not in store:
            raise
        print(f"[old] non-fatal exception after capture: {e!r}")
    finally:
        if agent.mcp is not None:
            await agent.mcp.close()

    if "old_request" not in store:
        raise RuntimeError("OLD path never reached the tool-loop LLM call")

    body = store["old_request"]
    # NOTE: do NOT stash `agent._messages` here as "the call's messages" — it's
    # a mutable list the agent keeps appending to for the rest of the run (incl.
    # the zero-tool-call retry nudges triggered by the synthetic no-tool-call
    # response from `_install_capture_transport`), so by the time `ask()`
    # returns it no longer reflects iteration 1. `body["messages"]` (below) is
    # the immutable JSON snapshot of the exact bytes sent for THIS call and is
    # the correct thing to reuse for the NEW path's input.
    body["_spike_meta"] = {"qtype": captured_qtype.get("qtype")}
    return body


# ---------------------------------------------------------------------------
# NEW PATH: plain LangGraph StateGraph (model node + ToolNode), tools loaded
# via langchain_mcp_adapters, tool SCHEMAS taken from the existing
# MCPClient.convert_tools_to_openai_format() (not the adapters' auto schema),
# messages/tool-filter reproduced via the same KQAProAgent helper methods.
# ---------------------------------------------------------------------------

async def capture_new_request(
    old_messages: List[Dict[str, Any]],
    old_qtype: str,
    tool_choice: str,
) -> Dict[str, Any]:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_mcp_adapters.sessions import StdioConnection
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, START, END
    from langgraph.prebuilt import ToolNode
    import httpx  # only used for the Timeout object, matching config.py's own usage

    from ama_kbqa.agents.kqapro_agent.agent import KQAProAgent
    from ama_kbqa.framework.mcp_client import MCPClient
    from ama_kbqa.config import (
        get_chat_model_name,
        get_chat_temperature,
        get_chat_max_tokens,
        get_chat_seed,
        get_provider_preferences,
        load_config,
    )

    # --- MCP tools over stdio via langchain_mcp_adapters (2a) ---
    agent_for_helpers = KQAProAgent(name="spike_new_helper", session_id="spike_new_helper")
    server_path = agent_for_helpers.get_mcp_server_path()

    connection = StdioConnection(
        transport="stdio",
        command=sys.executable,
        args=[server_path],
        env=os.environ.copy(),
    )
    mcp_client = MultiServerMCPClient({"kqapro": connection})

    async with mcp_client.session("kqapro") as session:
        raw_tools_result = await session.list_tools()
        raw_mcp_tools = raw_tools_result.tools
        # Also load LangChain BaseTool wrappers for the ToolNode (proves 2a end
        # to end, i.e. real tool *execution* would work), even though we do NOT
        # use their auto-generated schemas for the model-facing tool defs below.
        lc_tools = await load_mcp_tools(session)

    # --- Tool schemas: EXACT bytes from convert_tools_to_openai_format(),
    # applied to the raw MCP tool list, then filtered by the same qtype/
    # denylist logic KQAProAgent uses (imported, not reimplemented). ---
    schema_helper = MCPClient(server_path, "spike_schema_helper")
    all_openai_tools = schema_helper.convert_tools_to_openai_format(raw_mcp_tools)

    allowed = agent_for_helpers._get_allowed_tools_for_qtype(old_qtype)
    if allowed is not None:
        openai_tools = [t for t in all_openai_tools if t["function"]["name"] in allowed]
    else:
        openai_tools = all_openai_tools
    denied = agent_for_helpers._get_denied_tool_names()
    if denied:
        openai_tools = [t for t in openai_tools if t["function"]["name"] not in denied]

    if agent_for_helpers.mcp is not None:
        await agent_for_helpers.mcp.close()

    # --- Messages: reconstruct LangChain BaseMessage objects from the OLD
    # path's literal message dicts (system prompt + STATIC qtype context +
    # query + PER-QUESTION context) — same content, byte for byte. ---
    role_to_cls = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
    lc_messages = []
    for m in old_messages:
        cls = role_to_cls.get(m.get("role"))
        if cls is None:
            continue
        lc_messages.append(cls(content=m.get("content") or ""))

    # --- Provider/client config: mirror ama_kbqa.config._create_client exactly. ---
    config = load_config()
    provider = config["llm"]["chat_provider"]
    provider_config = config[provider]
    base_url = provider_config["base_url"]
    env_var_map = {"openrouter": "OPENROUTER_API_KEY", "kit": "KIT_API_KEY"}
    api_key = provider_config.get("api_key") or os.environ.get(env_var_map.get(provider, ""), "")

    timeout = httpx.Timeout(connect=20.0, read=60.0, write=10.0, pool=5.0)
    model_name = get_chat_model_name()

    model_kwargs: Dict[str, Any] = {}
    seed = get_chat_seed()
    if seed is not None:
        model_kwargs["seed"] = seed
    provider_prefs = get_provider_preferences()
    extra_body = {"provider": provider_prefs} if provider_prefs else None

    model = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=0,
        temperature=get_chat_temperature(),
        max_tokens=get_chat_max_tokens(),
        extra_body=extra_body,
        **model_kwargs,
    )

    store: Dict[str, Any] = {}
    _install_capture_transport(model.root_client._client, store, "new_request")

    model_with_tools = model.bind_tools(openai_tools, tool_choice=tool_choice)

    tool_node = ToolNode(lc_tools)

    class GraphState(dict):
        pass

    def call_model(state):
        response = model_with_tools.invoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    def should_continue(state):
        last = state["messages"][-1]
        if state.get("iterations", 0) >= 3:
            return "end"
        if getattr(last, "tool_calls", None):
            return "tools"
        return "end"

    graph_builder = StateGraph(dict)
    graph_builder.add_node("model", call_model)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_edge(START, "model")
    graph_builder.add_conditional_edges("model", should_continue, {"tools": "tools", "end": END})
    graph_builder.add_edge("tools", "model")
    graph = graph_builder.compile()

    try:
        final_state = await graph.ainvoke({"messages": lc_messages, "iterations": 0})
        print(f"[new] graph.ainvoke returned normally, {len(final_state['messages'])} messages")
    except Exception as e:  # pragma: no cover - diagnostic escape hatch
        if "new_request" not in store:
            raise
        print(f"[new] non-fatal exception after capture: {e!r}")

    if "new_request" not in store:
        raise RuntimeError("NEW path never reached the model call")

    return store["new_request"]


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def _diff_tools(old_tools: List[Dict], new_tools: List[Dict]) -> List[str]:
    lines = []
    old_names = [t["function"]["name"] for t in old_tools]
    new_names = [t["function"]["name"] for t in new_tools]
    if old_names != new_names:
        lines.append(f"- **tool order/set differs**: old={old_names} new={new_names}")
    else:
        lines.append(f"- tool order/set: IDENTICAL ({len(old_names)} tools)")
    for name in set(old_names) & set(new_names):
        ot = next(t for t in old_tools if t["function"]["name"] == name)
        nt = next(t for t in new_tools if t["function"]["name"] == name)
        if ot != nt:
            diffs = []
            if ot["type"] != nt["type"]:
                diffs.append(f"type: {ot['type']!r} vs {nt['type']!r}")
            of, nf = ot["function"], nt["function"]
            if of.get("description") != nf.get("description"):
                diffs.append(
                    f"description differs (old={len(of.get('description') or '')} chars, "
                    f"new={len(nf.get('description') or '')} chars)"
                )
            if of.get("parameters") != nf.get("parameters"):
                diffs.append("parameters (inputSchema) differ")
            extra_old = set(of.keys()) - set(nf.keys())
            extra_new = set(nf.keys()) - set(of.keys())
            if extra_old:
                diffs.append(f"old-only function keys: {sorted(extra_old)}")
            if extra_new:
                diffs.append(f"new-only function keys: {sorted(extra_new)}")
            if diffs:
                lines.append(f"- tool `{name}` differs: " + "; ".join(diffs))
    return lines


def _diff_messages(old_msgs: List[Dict], new_msgs: List[Dict]) -> List[str]:
    lines = []
    if len(old_msgs) != len(new_msgs):
        lines.append(f"- **message count differs**: old={len(old_msgs)} new={len(new_msgs)}")
    for i, (om, nm) in enumerate(zip(old_msgs, new_msgs)):
        o_role, n_role = om.get("role"), nm.get("role")
        o_content, n_content = om.get("content"), nm.get("content")
        if o_role != n_role:
            lines.append(f"- message[{i}] role differs: old={o_role!r} new={n_role!r}")
        if o_content != n_content:
            lines.append(
                f"- message[{i}] content differs (old={len(o_content or '')} chars, "
                f"new={len(n_content or '')} chars)"
            )
        extra_old = set(om.keys()) - set(nm.keys()) - {"role", "content"}
        extra_new = set(nm.keys()) - set(om.keys()) - {"role", "content"}
        if extra_old:
            lines.append(f"- message[{i}] old-only keys: {sorted(extra_old)}")
        if extra_new:
            lines.append(f"- message[{i}] new-only keys: {sorted(extra_new)}")
    if not lines:
        lines.append(f"- messages: IDENTICAL ({len(old_msgs)} messages)")
    return lines


def _diff_top_level(old: Dict, new: Dict) -> List[str]:
    lines = []
    skip = {"messages", "tools", "_spike_meta"}
    old_keys = set(old.keys()) - skip
    new_keys = set(new.keys()) - skip
    only_old = sorted(old_keys - new_keys)
    only_new = sorted(new_keys - old_keys)
    if only_old:
        lines.append(f"- **keys only in OLD**: {only_old}")
    if only_new:
        lines.append(f"- **keys only in NEW** (LangChain-added): {only_new}")
    for k in sorted(old_keys & new_keys):
        if old[k] != new[k]:
            lines.append(f"- `{k}` differs: old={old[k]!r} new={new[k]!r}")
        else:
            lines.append(f"- `{k}`: IDENTICAL ({old[k]!r})")
    return lines


def build_diff_report(old: Dict, new: Dict) -> str:
    lines = ["# Phase 0 request-body diff: OLD (BaseKBQAAgent) vs NEW (LangGraph)\n"]
    lines.append(f"qtype used for tool filtering: `{old.get('_spike_meta', {}).get('qtype')}`\n")

    lines.append("## Top-level params\n")
    lines.extend(_diff_top_level(old, new))

    lines.append("\n## Tools\n")
    lines.extend(_diff_tools(old.get("tools", []), new.get("tools", [])))

    lines.append("\n## Messages\n")
    lines.extend(_diff_messages(old.get("messages", []), new.get("messages", [])))

    return "\n".join(lines) + "\n"


async def main() -> None:
    question = _load_question(QUESTION_ID)
    print(f"Question ({QUESTION_ID}): {question}")

    print("\n=== Capturing OLD path request ===")
    old_request = await capture_old_request(question)
    qtype = old_request["_spike_meta"]["qtype"]
    print(f"OLD path qtype: {qtype}")
    print(f"OLD path tools: {[t['function']['name'] for t in old_request.get('tools', [])]}")

    with open(SPIKE_DIR / "old_request.json", "w") as f:
        json.dump(old_request, f, indent=2)

    tool_choice = old_request.get("tool_choice", "required")

    print("\n=== Capturing NEW (LangGraph) path request ===")
    new_request = await capture_new_request(
        old_request["messages"],
        qtype,
        tool_choice,
    )
    print(f"NEW path tools: {[t['function']['name'] for t in new_request.get('tools', [])]}")

    with open(SPIKE_DIR / "new_request.json", "w") as f:
        json.dump(new_request, f, indent=2)

    report = build_diff_report(old_request, new_request)
    with open(SPIKE_DIR / "request_diff.md", "w") as f:
        f.write(report)

    print("\n" + report)
    print(f"\nWrote: {SPIKE_DIR / 'old_request.json'}")
    print(f"Wrote: {SPIKE_DIR / 'new_request.json'}")
    print(f"Wrote: {SPIKE_DIR / 'request_diff.md'}")


if __name__ == "__main__":
    asyncio.run(main())
