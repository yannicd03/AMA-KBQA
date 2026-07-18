"""Provider-agnostic transient-error retry with stepped, self-resetting backoff.

One implementation shared by every surface (raw OpenAI SDK, LangChain ``ChatKIT``,
this repo's ``BaseKBQAAgent``) so the resilience behaviour is identical everywhere.

The backoff level lives on the :class:`TransientRetry` instance: it climbs on each
consecutive transient failure (across calls, not just within one) and resets to 0 on
the first success. So a flaky endpoint is waited out with ever-longer pauses rather
than abandoned, and a recovered endpoint is served promptly again.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Sequence, TypeVar

T = TypeVar("T")

# Substrings marking a RETRYABLE provider hiccup. Includes KIT's "Open WebUI: Server
# Connection Error", which arrives as a 200/400-with-error-body rather than a clean
# 5xx, so the OpenAI SDK's built-in retries miss it — the reason this layer exists.
TRANSIENT_MARKERS: tuple[str, ...] = (
    "server connection error", "open webui", "connection error", "connection reset",
    "connection aborted", "remote end closed", "timeout", "timed out", "read timed out",
    "500", "internal server error", "502", "503", "504", "bad gateway",
    "gateway timeout", "overloaded", "temporarily unavailable", "429", "rate limit",
)
# Substrings marking a DETERMINISTIC failure — retrying is futile, re-raise at once.
NON_TRANSIENT_MARKERS: tuple[str, ...] = (
    "401", "403", "invalid api key", "unauthorized", "permission",
)

DEFAULT_BACKOFF_STEPS: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 300.0)
DEFAULT_MAX_ATTEMPTS = 10


def is_transient_error(exc: BaseException) -> bool:
    """True for provider hiccups worth retrying, False for deterministic failures.

    Auth/permission errors short-circuit to False even if the message also happens to
    contain a transient-looking token.
    """
    m = str(exc).lower()
    if any(s in m for s in NON_TRANSIENT_MARKERS):
        return False
    return any(s in m for s in TRANSIENT_MARKERS)


class TransientRetry:
    """Stepped-backoff retry wrapper with a persistent, self-resetting level.

    Provider-agnostic: :meth:`run` wraps any zero-arg callable performing one attempt.
    """

    def __init__(
        self,
        steps: Sequence[float] = DEFAULT_BACKOFF_STEPS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        is_transient: Callable[[BaseException], bool] = is_transient_error,
        sleep: Optional[Callable[[float], None]] = None,
    ):
        self.steps = tuple(steps)
        self.max_attempts = max_attempts
        self._is_transient = is_transient
        # Resolve time.sleep lazily at call time (None) so tests can monkeypatch it;
        # pass an explicit callable to inject a fake clock.
        self._sleep = sleep
        self.level = 0  # persists across run() calls; resets on any success

    def run(
        self,
        fn: Callable[[], T],
        on_retry: Optional[Callable[[BaseException, int, float], None]] = None,
    ) -> T:
        """Call ``fn`` with retry. ``on_retry(exc, attempt, wait)`` fires before each sleep."""
        sleep = self._sleep or time.sleep
        attempts = 0
        while True:
            try:
                out = fn()
                self.level = 0  # endpoint alive: reset the ramp
                return out
            except Exception as exc:
                attempts += 1
                if not self._is_transient(exc) or attempts >= self.max_attempts:
                    raise
                wait = self.steps[min(self.level, len(self.steps) - 1)]
                self.level += 1
                if on_retry is not None:
                    on_retry(exc, attempts, wait)
                sleep(wait)


def create_with_retry(
    client,
    retry: Optional[TransientRetry] = None,
    on_retry: Optional[Callable[[BaseException, int, float], None]] = None,
    **create_kwargs,
):
    """``client.chat.completions.create(**kwargs)`` wrapped in stepped-backoff retry.

    Convenience for the raw OpenAI-SDK surface. Pass a shared :class:`TransientRetry`
    to keep the backoff level persistent across many calls.
    """
    retry = retry or TransientRetry()
    return retry.run(lambda: client.chat.completions.create(**create_kwargs), on_retry=on_retry)
