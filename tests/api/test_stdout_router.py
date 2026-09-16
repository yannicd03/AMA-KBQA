"""The thread/context-routing stdout proxy and the per-run log buffer."""

from __future__ import annotations

import asyncio
import io
import sys
import threading

from fastapi.testclient import TestClient

from ama_kbqa.api import app as app_module
from ama_kbqa.api import stdout_router
from ama_kbqa.api.stdout_router import RoutingStdout, RunLog, route_current_context


def _in_thread(fn) -> None:
    t = threading.Thread(target=fn)
    t.start()
    t.join(5)
    assert not t.is_alive()


def test_unbound_writes_reach_the_real_stream():
    real = io.StringIO()
    proxy = RoutingStdout(real)
    proxy.write("hello\n")
    proxy.writelines(["a", "b"])
    proxy.flush()
    assert real.getvalue() == "hello\nab"


def test_bound_thread_writes_into_its_log_only():
    real = io.StringIO()
    proxy = RoutingStdout(real)
    log_a, log_b = RunLog(), RunLog()

    def worker(log, text):
        def run():
            route_current_context(log)
            assert proxy.isatty() is False
            for _ in range(50):
                proxy.write(text)
        return run

    ta = threading.Thread(target=worker(log_a, "A"))
    tb = threading.Thread(target=worker(log_b, "B"))
    ta.start()
    tb.start()
    proxy.write("main")
    ta.join(5)
    tb.join(5)

    assert log_a.text() == "A" * 50
    assert log_b.text() == "B" * 50
    assert real.getvalue() == "main"
    assert log_a.version == 50


def test_binding_follows_the_event_loop_and_to_thread():
    real = io.StringIO()
    proxy = RoutingStdout(real)
    log = RunLog()

    async def main():
        async def child():
            proxy.write("task ")
        await asyncio.create_task(child())
        await asyncio.to_thread(proxy.write, "to_thread")

    def run():
        route_current_context(log)
        asyncio.run(main())

    _in_thread(run)
    assert log.text() == "task to_thread"
    assert real.getvalue() == ""


def test_binding_does_not_leak_into_the_calling_thread():
    log = RunLog()
    _in_thread(lambda: route_current_context(log))
    assert stdout_router.current_log() is None


def test_install_routes_real_prints(capsys):
    proxy = stdout_router.install()
    try:
        assert stdout_router.install() is proxy  # idempotent
        log = RunLog()

        def run():
            route_current_context(log)
            print("from the run")

        _in_thread(run)
        print("from main")
    finally:
        stdout_router.uninstall(proxy)
    assert not isinstance(sys.stdout, RoutingStdout)
    assert log.text() == "from the run\n"
    out = capsys.readouterr().out
    assert "from main" in out and "from the run" not in out


def test_install_wraps_a_stream_that_replaced_the_proxy():
    proxy = stdout_router.install()
    replacement = io.StringIO()
    sys.stdout = replacement
    try:
        again = stdout_router.install()
        assert again is not proxy and again.real is replacement
        print("unbound")
        assert replacement.getvalue() == "unbound\n"
    finally:
        sys.stdout = proxy.real


def test_uninstall_leaves_a_replaced_stdout_alone():
    proxy = stdout_router.install()
    replacement = io.StringIO()
    sys.stdout = replacement
    try:
        stdout_router.uninstall(proxy)
        assert sys.stdout is replacement
    finally:
        sys.stdout = proxy.real


def test_run_log_drops_the_head_past_its_cap():
    log = RunLog(max_chars=100)
    for i in range(30):
        log.write(f"{i:04d}\n")  # 150 chars total
    text = log.text()
    assert log.truncated is True
    assert len(text) <= 100
    assert text.endswith("0029\n")
    assert log.write("") == 0


def test_app_lifespan_installs_and_removes_the_proxy(monkeypatch):
    monkeypatch.setattr(app_module.meta, "default_temperature", lambda: 1.0)
    before = sys.stdout
    with TestClient(app_module.create_app()):
        assert isinstance(sys.stdout, RoutingStdout)
        assert sys.stdout.real is before
    assert sys.stdout is before
