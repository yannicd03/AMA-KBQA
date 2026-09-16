"""
Shared MCP client for KBQA agents.

Provides a reusable client for communicating with MCP servers
that host knowledge graph tools.
"""

from __future__ import annotations
import os
import sys
import asyncio
from pathlib import Path
from contextlib import AsyncExitStack
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from mcp.types import Tool as McpTool
from asyncio.exceptions import CancelledError


# Terminal colors for tracing
COLOR_BLUE = '\033[94m'
COLOR_GREEN = '\033[92m'
COLOR_RED = '\033[91m'
COLOR_YELLOW = '\033[93m'
COLOR_CYAN = '\033[96m'
COLOR_END = '\033[0m'

# Bounded wait for a single MCP tool call. Without one, `call_tool` awaits a
# response forever — the only genuinely unbounded wait in the system, and the
# reason no run could promise a time limit (the SPARQL timeout one layer down
# bounds the query, not a wedged or silent server subprocess).
#
# Resolution order mirrors framework.adapters.resolve: env var, then
# config.toml, then this default. A value <= 0 means "no timeout" and restores
# the old unbounded behaviour for a deliberately slow workload.
DEFAULT_TOOL_TIMEOUT_SECONDS = 180.0
TOOL_TIMEOUT_ENV_VAR = "AMA_KBQA_MCP_TOOL_TIMEOUT_SECONDS"
TOOL_TIMEOUT_CONFIG_KEY = ("agent", "mcp_tool_timeout_seconds")

# JSON-RPC error code the MCP SDK raises when a request read times out
# (httpx.codes.REQUEST_TIMEOUT); matched by value so this module does not
# import httpx just for a constant.
_REQUEST_TIMEOUT_CODE = 408


class McpToolTimeout(RuntimeError):
    """An MCP tool call exceeded its configured timeout."""


def is_timeout_error(exc: BaseException) -> bool:
    """True when `exc` is an MCP read timeout (or a raw transport timeout)."""
    if isinstance(exc, TimeoutError):
        return True
    return (
        isinstance(exc, McpError)
        and getattr(exc.error, "code", None) == _REQUEST_TIMEOUT_CODE
    )


def tool_timeout_error(
    agent_name: str, tool_name: str, timeout: Optional[float]
) -> McpToolTimeout:
    """Build (and trace) the error for a tool call that outran its budget.

    Shared by this client and the Orchestrator's own MCP client so both report
    a stalled tool the same way.
    """
    budget = f"{timeout:.0f}s" if timeout else "the configured timeout"
    trace(
        agent_name,
        f"{COLOR_YELLOW}Tool '{tool_name}' timed out after {budget}{COLOR_END}",
        COLOR_YELLOW,
    )
    return McpToolTimeout(
        f"Tool '{tool_name}' timed out after {budget}. The tool server did not "
        f"respond; try a narrower request or a different tool."
    )


def resolve_tool_timeout(timeout_seconds: Optional[float] = None) -> Optional[float]:
    """Resolve the per-call MCP tool timeout in seconds.

    Returns None when the timeout is disabled (a configured value <= 0), which
    callers pass straight through as "wait indefinitely".
    """
    value: Optional[float] = None

    if timeout_seconds is not None:
        value = float(timeout_seconds)
    else:
        raw = os.environ.get(TOOL_TIMEOUT_ENV_VAR)
        if raw is not None:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = None  # fall through to config.toml / default
        if value is None:
            try:
                from ama_kbqa.config import load_config
                section, key = TOOL_TIMEOUT_CONFIG_KEY
                configured = load_config().get(section, {}).get(key)
                if configured is not None:
                    value = float(configured)
            except Exception:
                # config.toml is optional here: the client must stay
                # constructible without one (tests, stand-alone scripts).
                value = None
        if value is None:
            value = DEFAULT_TOOL_TIMEOUT_SECONDS

    return value if value > 0 else None


def trace(agent_name: str, msg: str, color: str = COLOR_BLUE) -> None:
    """Standardized tracing with timestamp and agent prefix."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    lines = msg.split('\n')
    header = f"[{color}{timestamp}{COLOR_END}] {color}[{agent_name}]{COLOR_END} -> {lines[0]}"
    print(header)
    for line in lines[1:]:
        print(f"{' ' * (len(timestamp) + 2 + len(agent_name) + 6)} {color}{line}{COLOR_END}")


class MCPClient:
    """
    Client for communicating with MCP servers.

    Manages the lifecycle of an MCP server connection, providing
    methods to list tools and call them.
    """

    def __init__(
        self,
        server_path: str,
        agent_name: str,
        tool_timeout_seconds: Optional[float] = None,
    ):
        """
        Initialize the MCP client.

        Args:
            server_path: Path to the MCP server Python file
            agent_name: Name of the agent for tracing
            tool_timeout_seconds: Per-call timeout for `call_tool`. Defaults to
                the resolved configuration (see `resolve_tool_timeout`); pass a
                value <= 0 to wait indefinitely.
        """
        self.server_path = Path(server_path)
        self.agent_name = agent_name
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self._connected = False
        self.tool_timeout_seconds = resolve_tool_timeout(tool_timeout_seconds)

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected."""
        return self._connected

    async def start(self, max_attempts: int = 3, backoff_seconds: float = 1.0) -> None:
        """
        Start the MCP server and establish connection.

        Retries on transient startup failures (e.g. subprocess stdio race,
        slow first-time import). Raises only after `max_attempts` failures.

        Args:
            max_attempts: Total number of start attempts (>=1).
            backoff_seconds: Initial delay between attempts; doubles each retry.
        """
        if self._connected:
            return

        if not self.server_path.exists():
            trace(
                self.agent_name,
                f"{COLOR_RED}MCP server not found: {self.server_path}{COLOR_END}",
                COLOR_RED
            )
            raise FileNotFoundError(f"MCP server not found: {self.server_path}")

        last_err: Optional[BaseException] = None
        for attempt in range(1, max_attempts + 1):
            try:
                # env=None would make MCP SDK use a minimal default env (PATH only),
                # stripping API keys from the server subprocess. Pass parent env.
                client_gen = stdio_client(
                    StdioServerParameters(
                        command=sys.executable,
                        args=[str(self.server_path)],
                        env=os.environ.copy(),
                    )
                )

                read, write = await self.exit_stack.enter_async_context(client_gen)
                self.session = await self.exit_stack.enter_async_context(
                    ClientSession(read, write)
                )
                await self.session.initialize()
                self._connected = True
                if attempt > 1:
                    trace(
                        self.agent_name,
                        f"{COLOR_GREEN}MCP server started on attempt {attempt}/{max_attempts}{COLOR_END}",
                        COLOR_GREEN,
                    )
                return

            except Exception as e:
                last_err = e
                # Clean up partially entered async contexts to avoid orphaned anyio tasks
                try:
                    await self.exit_stack.aclose()
                except Exception:
                    pass
                self.exit_stack = AsyncExitStack()
                self.session = None

                if attempt < max_attempts:
                    trace(
                        self.agent_name,
                        f"{COLOR_YELLOW}MCP start attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {backoff_seconds:.1f}s...{COLOR_END}",
                        COLOR_YELLOW,
                    )
                    await asyncio.sleep(backoff_seconds)
                    backoff_seconds *= 2

        trace(
            self.agent_name,
            f"{COLOR_RED}Failed to start MCP server after {max_attempts} attempts: {last_err}{COLOR_END}",
            COLOR_RED,
        )
        raise RuntimeError(
            f"Failed to start MCP server after {max_attempts} attempts: {last_err}"
        )

    async def list_tools(self) -> List[McpTool]:
        """
        List available tools from the MCP server.

        Returns:
            List of available MCP tools

        Raises:
            RuntimeError: If not connected
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        result = await self.session.list_tools()
        return result.tools

    async def call_tool(self, name: str, args: Dict) -> str:
        """
        Call a tool on the MCP server, bounded by the configured timeout.

        The SDK's own `read_timeout_seconds` is used rather than an
        `asyncio.wait_for` wrapper: it cancels the pending request inside the
        session (so the response stream is cleaned up on the session's own
        task) instead of cancelling an await from outside it.

        Args:
            name: Name of the tool to call
            args: Arguments to pass to the tool

        Returns:
            Tool result as a string

        Raises:
            RuntimeError: If not connected
            McpToolTimeout: If the server did not answer within the timeout.
                Callers (`BaseKBQAAgent._execute_single_tool`) already turn a
                tool exception into a normal "Error executing ..." tool result,
                so the agent can react to it like any other tool failure.
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        timeout = self.tool_timeout_seconds
        read_timeout = timedelta(seconds=timeout) if timeout else None

        try:
            result = await self.session.call_tool(
                name, arguments=args, read_timeout_seconds=read_timeout
            )
        except Exception as e:
            if is_timeout_error(e):
                raise tool_timeout_error(self.agent_name, name, timeout) from e
            raise

        if hasattr(result, "content") and result.content:
            return result.content[0].text

        return str(result)

    async def close(self) -> None:
        """
        Close the MCP server connection.

        Handles cleanup gracefully, ignoring expected errors during shutdown.
        """
        if not self._connected:
            return

        try:
            await self.exit_stack.aclose()
            # Brief yield for subprocess cleanup and stdio buffer flushing.
            # Trimmed from 0.5s: on stdio, aclose() already tears down the
            # transport; the long sleep was pure dead time paid on every close
            # (notably the orchestrator, which closes its probe server on every
            # question). A short yield is enough to let the child reap.
            await asyncio.sleep(0.05)

        except (CancelledError, RuntimeError) as e:
            trace(
                self.agent_name,
                f"{COLOR_YELLOW}WARNING: MCP-Close error on shutdown ignored "
                f"({type(e).__name__}).{COLOR_END}",
                COLOR_YELLOW
            )

        except Exception as e:
            trace(
                self.agent_name,
                f"{COLOR_RED}Error closing MCP client: {e}{COLOR_END}",
                COLOR_RED
            )

        finally:
            self._connected = False
            self.session = None
            self.exit_stack = AsyncExitStack()

    @staticmethod
    def _compress_description(description: str, max_chars: int = 300) -> str:
        """
        Compress a tool description to save tokens.
        Keeps the first line (summary) and truncates the rest.

        Args:
            description: Original tool description
            max_chars: Maximum characters to keep

        Returns:
            Compressed description
        """
        if not description or len(description) <= max_chars:
            return description or ""

        # Keep first meaningful paragraph
        lines = description.strip().split('\n')
        result = []
        char_count = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if char_count + len(stripped) > max_chars:
                break
            result.append(stripped)
            char_count += len(stripped)

        return ' '.join(result) if result else description[:max_chars]

    def get_tool_schema(self, mcp_tool: McpTool, compress: bool = True) -> Dict:
        """
        Convert an MCP tool to OpenAI function calling format.

        Args:
            mcp_tool: The MCP tool to convert
            compress: Whether to compress descriptions for token savings

        Returns:
            Tool schema in OpenAI format
        """
        description = mcp_tool.description or ""
        if compress:
            description = self._compress_description(description)

        return {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": description,
                "parameters": mcp_tool.inputSchema
            }
        }

    def convert_tools_to_openai_format(self, mcp_tools: List[McpTool], compress: bool = True) -> List[Dict]:
        """
        Convert a list of MCP tools to OpenAI function calling format.

        Args:
            mcp_tools: List of MCP tools
            compress: Whether to compress descriptions

        Returns:
            List of tools in OpenAI format
        """
        return [self.get_tool_schema(t, compress=compress) for t in mcp_tools]
