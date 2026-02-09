"""Async utilities for running coroutines inside Streamlit."""

import asyncio


def run_async(coro):
    """Run an async coroutine from synchronous Streamlit code.

    Creates a fresh event loop each time to avoid stale anyio state
    (cancel scopes, task groups) leaking between invocations, which
    causes 'cancel scope in a different task' errors with MCP clients.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
