# Benchmark Persistence Refactor

**Status:** 📋 Planned
**Priority:** Medium
**Identified:** 2026-04-20

## Problem

Long-running benchmark / batch runs kicked off from the Streamlit frontend do
not survive browser or SSH tunnel disconnection. The current implementation
in `ama_kbqa/frontend/pages/2_Batch_Processing.py` spawns the benchmark via
`subprocess.Popen` and pipes stdout into a daemon thread that writes progress
into `st.session_state`. When the WebSocket drops:

1. Streamlit eventually garbage-collects the disconnected session (seconds to
   a few minutes, depending on config).
2. The daemon thread dies with the parent Streamlit script run.
3. The subprocess has no process-group isolation and typically dies too.
4. Partial results are lost — `benchmark_agents.py` writes
   `summary.json` / `results.json` / `console_output.txt` only at run end,
   not incrementally.
5. There is no job ID + reconnect mechanism, so even on a brief reconnect the
   UI resets.

## Comparison: Orca has the inverse pattern

The sibling Orca project runs batches as decoupled `asyncio` tasks spawned
by a FastAPI background worker. Results persist to disk after each item;
jobs are addressable by `task_id`; clients can reconnect within a 5-minute
window to replay events. See Orca's `batch_queue.py` for reference.

## Proposed Direction

1. **Decouple run lifecycle from the Streamlit session.** Run benchmarks in a
   process that isn't a child of the Streamlit script — options:
   - Dedicated FastAPI sidecar with a background-task queue (mirrors Orca).
   - systemd-style worker daemon that watches a job spool directory.
   - Container-level job runner (separate service in
     `docker-compose.yml`) consuming a shared queue.
2. **Write progress incrementally.** Update `summary.json` (or a
   `progress.json`) after each question, not only on completion. Any crash
   or disconnect leaves a meaningful partial artifact.
3. **Address runs by `job_id`.** Frontend submits and gets a job_id back,
   can query status / tail console / cancel. Closing the browser does not
   affect the run.
4. **Reconnect support.** On page load, the frontend lists recent jobs
   (from disk) and can re-attach to in-progress ones by streaming their
   progress file or subscribing to SSE.

## Out of Scope for this TODO

- Drag-to-reorder queue semantics (separate concern).
- Multi-tenant isolation.

## Interim Workaround (documented in SOP)

Run the benchmark CLI in a detached `tmux` / `screen` session:
```bash
ssh hetzner
tmux new -s bench
uv run ama-kbqa benchmark -s kqapro -n 100 --seed 42
# Ctrl-b d to detach; reattach later with tmux attach -t bench
```
This keeps the run alive across disconnects without any code changes.

## Links

- `ama_kbqa/frontend/pages/2_Batch_Processing.py:190-208` — current subprocess+thread pattern
- `ama_kbqa/frontend/pages/2_Batch_Processing.py:35-52` — session-state-only storage
- `ama_kbqa/benchmark_agents.py:895-898` — end-of-run disk write
