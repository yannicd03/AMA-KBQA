"""One-question smoke test for the demo frontend image, one agent per process.

Run inside the demo frontend container, e.g. on the Hetzner box:

    docker exec -i frontend_amakbqa_next python -u - "<agent>" [model] < scripts/demo_smoke_ask.py

<agent> is a picker entry: "Orchestrator (Router)", "Orchestrator (Federated)",
"KQAPro" or "SciQA". The model defaults to the demo's preferred KIT model and is
applied the way the sidebar picker applies it (in-memory config plus env vars
for the MCP subprocesses); the agent then answers one question end to end
(KIT LLM, Qdrant, Virtuoso, MCP servers). Exit code 0 means a non-empty answer.

One agent per process on purpose: several agents in one asyncio.run() trip over
MCP stdio_client teardown across tasks. The demo itself runs each question in
its own thread and event loop.
"""

import asyncio
import sys
import time

from ama_kbqa.frontend.utils.agent_factory import create_agent
from ama_kbqa.frontend.utils.chat_controls import DEFAULT_MODEL_PREFERENCE, apply_chat_settings

QUESTIONS = {
    "Orchestrator (Router)": "In which city was Albert Einstein born?",
    # Cross-graph on purpose, so federated dispatch fans out to both
    # specialists: breast cancer is a KQAPro (Wikidata) disease entity with
    # `notable_people_with_this_condition` links AND an ORKG research problem
    # with real contributions behind it, so both halves of the fused answer
    # are grounded. Budget ~3 minutes: the slower specialist gates the run.
    "Orchestrator (Federated)": (
        "Which notable people have had breast cancer, and what research "
        "contributions address breast cancer?"
    ),
    "KQAPro": "Who is the director of Inception?",
    "SciQA": "What research contributions address COVID-19 detection?",
}
TIMEOUT_S = 300


async def ask(agent_name: str, question: str) -> str:
    agent = create_agent(agent_name)
    try:
        return await asyncio.wait_for(agent.ask(question), timeout=TIMEOUT_S)
    finally:
        # Specialists keep their MCP connection open after ask(); close it in the
        # task that opened it, so asyncio.run() shutdown doesn't tear it down from
        # a different task. The Orchestrator closes its own inside ask() and has
        # no close().
        close = getattr(agent, "close", None)
        if close is not None:
            await close()


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in QUESTIONS:
        print(f"usage: demo_smoke_ask.py <agent> [model]; agent is one of {list(QUESTIONS)}")
        return 2
    agent_name = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL_PREFERENCE[0]
    apply_chat_settings(model, 1.0)

    question = QUESTIONS[agent_name]
    t0 = time.monotonic()
    try:
        answer = asyncio.run(ask(agent_name, question))
        ok = bool(answer and answer.strip())
    except Exception as exc:  # noqa: BLE001
        answer, ok = f"EXCEPTION {type(exc).__name__}: {exc}", False
    flat = " ".join(str(answer).split())
    print(f"[{'OK' if ok else 'FAIL'}] {agent_name} ({time.monotonic() - t0:.0f}s) {model}")
    print(f"    Q: {question}")
    print(f"    A: {flat[:400]}")
    # Full, unflattened answer: the conversational answer contract spans several
    # sections (including a fenced ```sparql block) that the 400-char preview
    # above truncates away.
    print("--- FULL ANSWER BEGIN ---")
    print(answer)
    print("--- FULL ANSWER END ---")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
