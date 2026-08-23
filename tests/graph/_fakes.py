"""Shared fakes for tests/graph/*: a scripted LangChain chat model and a
``BaseKBQAAgent`` test-double base class.

Not a pytest module itself (no ``test_`` prefix) — imported by the test
modules that need them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from ama_kbqa.framework.base_agent import BaseKBQAAgent
from ama_kbqa.framework.trace import TraceRecorder


class ScriptedChatModel(BaseChatModel):
    """Returns one scripted ``AIMessage`` per call, in order.

    ``bind_tools`` is overridden (the ``BaseChatModel`` default raises
    ``NotImplementedError``) to record every ``tool_choice`` it is called
    with, so a test can assert the tool_choice schedule
    (``"required"`` for iteration <= 3, ``"auto"`` after) without needing a
    real provider.
    """

    _responses: List[AIMessage] = PrivateAttr(default_factory=list)
    _index: int = PrivateAttr(default=0)
    _tool_choice_calls: List[Optional[str]] = PrivateAttr(default_factory=list)

    def __init__(self, responses: List[AIMessage], **kwargs: Any):
        super().__init__(**kwargs)
        self._responses = list(responses)
        self._index = 0
        self._tool_choice_calls = []

    @property
    def tool_choice_calls(self) -> List[Optional[str]]:
        return self._tool_choice_calls

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("ScriptedChatModel is async-only; use ainvoke.")

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self._responses[self._index]
        self._index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, *, tool_choice: Optional[str] = None, **kwargs: Any):
        self._tool_choice_calls.append(tool_choice)
        return self.bind(tool_choice=tool_choice, **kwargs)


class GraphAgentDouble(BaseKBQAAgent):
    """Shared ``BaseKBQAAgent`` test double for tests/graph/*.

    Phase 2's ``execute_tools``/``call_model`` nodes reuse several
    ``BaseKBQAAgent`` methods directly (``_detect_loops``,
    ``_handle_loop_detected``, ``_manage_context_window``,
    ``_inject_journal_refresh``, ``_maybe_inject_raw_sparql_distress``,
    ``_get_journal_summary_answer_prompt``) — see ``ama_kbqa.graph.guards``
    and ``ama_kbqa.graph.context`` module docstrings. Those methods read
    agent-level attributes that ``BaseKBQAAgent.__init__`` normally sets up
    (via a real ``get_config()``/MCP connection this test double bypasses),
    so this base class sets sane defaults for all of them, matching the
    attribute set ``tests/graph/test_legacy_graph_parity.py::_ParityAgent``
    already established. Subclasses override only what they specifically
    need to script or assert on (tool execution, known tools, mcp, etc).
    """

    def __init__(self, known_tools: tuple = ()):
        self.name = "graph_test_agent"
        self.model = "test-model"
        self.recorder = TraceRecorder()
        self._parent_span_id_override = None
        from chatkit import TransientRetry

        self._retry = TransientRetry()
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.tool_call_counts: Dict[str, int] = {}
        self.tool_call_durations: list = []
        self.tool_call_history: list = []
        self.tool_sequence: list = []
        self.journal_snapshots: list = []
        self.last_journal_state = None
        self._known_tool_names = set(known_tools)
        self._context_limit = 100000
        self._next_trim_trigger = 0.5
        self._find_resource_cap = 8
        self._sparql_cap = 10
        self._raw_sparql_intervention_done = False
        self._text_tool_call_mode = False
        self.mcp = None
        self._messages: List[Dict[str, Any]] = []

    def get_config(self):  # pragma: no cover - unused
        raise NotImplementedError

    def get_mcp_server_path(self) -> str:  # pragma: no cover - unused
        raise NotImplementedError

    def _trace(self, message: str, color: str = "") -> None:
        pass

    async def _execute_single_tool(self, func_name: str, func_args: dict) -> str:
        self.tool_call_counts[func_name] = self.tool_call_counts.get(func_name, 0) + 1
        return f"result:{func_name}"
