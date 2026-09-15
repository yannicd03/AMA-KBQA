"""FastAPI backend for the React demo frontend (``web/``).

Runs the same in-process agents as the Streamlit chat page, through the same
helpers (``ama_kbqa.frontend.utils.*``), and streams each run's live state to
the browser over Server-Sent Events. The API contract lives in the React
frontend spec; ``app.py`` is the entry point (``ama-kbqa-api``).

All state (runs, sessions, persisted multiturn agents) is held in process
memory, so the server must run with exactly one uvicorn worker.
"""

import logging


class _NoRuntimeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "No runtime found" not in record.getMessage()


def _quiet_streamlit_cache_warning() -> None:
    """Drop Streamlit's "No runtime found" line in this process.

    ``chat_controls`` wraps its model fetchers in ``st.cache_data``. Outside a
    Streamlit runtime (the API never has one) each wrapped function logs
    "No runtime found, using MemoryCacheStorageManager" once when it is
    defined and once on its first call; after that the in-memory cache works
    as intended (TTL honoured, exceptions not cached). The line is noise here,
    so only that message is filtered, nothing else from Streamlit.
    """
    logging.getLogger("streamlit.runtime.caching.cache_data_api").addFilter(_NoRuntimeFilter())


_quiet_streamlit_cache_warning()
