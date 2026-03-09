"""
Shared MCP client for KBQA agents.

Provides a reusable client for communicating with MCP servers
that host knowledge graph tools.
"""

from __future__ import annotations
import sys
import asyncio
from pathlib import Path
from contextlib import AsyncExitStack
from typing import Dict, List, Optional
from datetime import datetime

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as McpTool
from asyncio.exceptions import CancelledError


# Terminal colors for tracing
COLOR_BLUE = '\033[94m'
COLOR_GREEN = '\033[92m'
COLOR_RED = '\033[91m'
COLOR_YELLOW = '\033[93m'
COLOR_CYAN = '\033[96m'
COLOR_END = '\033[0m'


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

    def __init__(self, server_path: str, agent_name: str):
        """
        Initialize the MCP client.

        Args:
            server_path: Path to the MCP server Python file
            agent_name: Name of the agent for tracing
        """
        self.server_path = Path(server_path)
        self.agent_name = agent_name
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected."""
        return self._connected

    async def start(self) -> None:
        """
        Start the MCP server and establish connection.

        Raises:
            FileNotFoundError: If the server file doesn't exist
            RuntimeError: If connection fails
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

        try:
            client_gen = stdio_client(
                StdioServerParameters(
                    command=sys.executable,
                    args=[str(self.server_path)],
                    env=None
                )
            )

            read, write = await self.exit_stack.enter_async_context(client_gen)
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await self.session.initialize()
            self._connected = True

        except Exception as e:
            # Clean up partially entered async contexts to avoid orphaned anyio tasks
            try:
                await self.exit_stack.aclose()
            except Exception:
                pass
            self.exit_stack = AsyncExitStack()
            self.session = None
            trace(
                self.agent_name,
                f"{COLOR_RED}Failed to start MCP server: {e}{COLOR_END}",
                COLOR_RED
            )
            raise RuntimeError(f"Failed to start MCP server: {e}")

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
        Call a tool on the MCP server.

        Args:
            name: Name of the tool to call
            args: Arguments to pass to the tool

        Returns:
            Tool result as a string

        Raises:
            RuntimeError: If not connected
        """
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        result = await self.session.call_tool(name, arguments=args)

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
            # Allow time for subprocess cleanup and stdio buffer flushing
            await asyncio.sleep(0.5)

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
            # Additional cleanup delay to ensure resources are fully released
            await asyncio.sleep(0.2)

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
