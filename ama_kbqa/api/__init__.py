"""FastAPI backend for the React demo frontend (``web/``).

Runs the same in-process agents as the Streamlit chat page, through the same
helpers (``ama_kbqa.frontend.utils.*``), and streams each run's live state to
the browser over Server-Sent Events. The API contract lives in the React
frontend spec; ``app.py`` is the entry point (``ama-kbqa-api``).

All state (runs, sessions, persisted multiturn agents) is held in process
memory, so the server must run with exactly one uvicorn worker.
"""
