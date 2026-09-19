"""Every ``_persist_completed_run`` call in the chat page names the provider.

``_persist_completed_run`` has taken a required keyword-only ``provider`` since
the booth commit f35b2a7, which passed it only from the simplified-view call.
A run completing in the default view therefore raised TypeError inside the
live fragment. That path needs a running agent, which AppTest cannot provide,
so the call sites are pinned statically instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

CHAT_PY = Path(__file__).resolve().parents[2] / "ama_kbqa" / "frontend" / "chat.py"


def test_every_persist_call_names_the_provider():
    tree = ast.parse(CHAT_PY.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_persist_completed_run"
    ]
    assert len(calls) >= 2
    for call in calls:
        assert "provider" in {kw.arg for kw in call.keywords}, ast.unparse(call)
