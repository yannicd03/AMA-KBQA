"""Shared fake LangChain chat model for tests/graph/*.

Not a pytest module itself (no ``test_`` prefix) — imported by the test
modules that need a scripted, network-free ``BaseChatModel``.
"""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


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
