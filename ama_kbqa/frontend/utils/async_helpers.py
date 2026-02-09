"""Async utilities for running coroutines inside Streamlit."""

import asyncio


def run_async(coro):
    """Run an async coroutine from synchronous Streamlit code.

    Handles the common pattern of getting or creating an event loop.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)
