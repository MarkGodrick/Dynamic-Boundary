"""Two-stage Ctrl-C handling for entry points that own a vLLM subprocess.

Why this exists: SGBS / adaptive-eval main loops park on ThreadPoolExecutor +
blocking vLLM HTTP calls. A naive `sys.exit(1)` from a signal handler waits
for every in-flight worker to drain (minutes), and `os._exit()` would leave
the vLLM subprocess group as an orphan. This module installs handlers that:

  1st signal: set a `stop_event`. The host loop is expected to check it at a
              safe point (after a snapshot/flush) and break gracefully.
  2nd signal: SIGKILL the active model's vLLM process group (no wait), then
              os._exit(130). Guarantees no orphan vllm; sub-second response.

The host registers the currently-active model with `set_active_model(m)` /
`clear_active_model()` so the handler can reach it. Any object exposing a
`force_kill()` method works (VLLMModel.force_kill).
"""

from __future__ import annotations

import os
import signal
import sys
import threading
from typing import Protocol


class _Killable(Protocol):
    def force_kill(self) -> None: ...


_STOP = threading.Event()
_ACTIVE: _Killable | None = None
_ACTIVE_LOCK = threading.Lock()


def set_active_model(m: _Killable | None) -> None:
    global _ACTIVE
    with _ACTIVE_LOCK:
        _ACTIVE = m


def clear_active_model() -> None:
    set_active_model(None)


def install_two_stage_handlers() -> threading.Event:
    """Install SIGINT/SIGTERM handlers and return the shared stop event.

    Idempotent: re-installing replaces the previous handlers and clears the
    stop_event so a second run in the same process starts fresh (only matters
    in tests; entry points install once and exit).
    """
    _STOP.clear()

    def _on_sig(sig: int, _frame) -> None:
        if _STOP.is_set():
            # Hard exit: SIGKILL vllm pgroup (no wait) → os._exit(130).
            # Cannot use Python logging here; stdio may be in inconsistent
            # state mid-handler. Print is best-effort.
            try:
                print(
                    f"\nSecond signal {sig} → hard exit "
                    f"(SIGKILL vllm pgroup, os._exit(130))",
                    file=sys.stderr, flush=True,
                )
            except Exception:
                pass
            with _ACTIVE_LOCK:
                m = _ACTIVE
            if m is not None:
                try:
                    m.force_kill()
                except Exception:
                    pass
            os._exit(130)

        try:
            print(
                f"\nCaught signal {sig}; finishing current batch then saving "
                f"(Ctrl-C again to force exit)",
                file=sys.stderr, flush=True,
            )
        except Exception:
            pass
        _STOP.set()

    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)
    return _STOP


def stop_event() -> threading.Event:
    """Return the shared stop event (for callers that didn't call install)."""
    return _STOP
