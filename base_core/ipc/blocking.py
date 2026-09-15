"""Turning the asynchronous worker-handle API into blocking calls.

``BaseWorkerHandle._request(msg, on_reply, on_error)`` has always taken both callbacks;
for a long time nothing used them — every handle's ``_on_reply`` was ``pass``, discarding
the result *and* the error. That is fine for a UI, which wants its thread back, and wrong
for a routine, which is a sequence: step two must not begin until step one finished, and
"finished" has to include "or failed, and here is why".

**Where these may be called.** On a routine's own ``TaskRunner`` thread, which is what
they are for. Replies arrive on the IPC reader thread, so the wait cannot deadlock.
**Never** from an EventBus handler — those run on the publisher's thread, which may be
the reader thread itself, and waiting there deadlocks against the reply you are waiting
for.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

from base_core.ipc.worker_handle import BaseWorkerHandle, WorkerStatus

#: Poll interval for :func:`wait_for_status`.
DEFAULT_POLL_S = 0.05


class WorkerCallError(RuntimeError):
    """A worker request failed or did not answer in time."""


def blocking_request(
    submit: Callable[[Callable[[], None], Callable[[str], None]], None],
    *,
    timeout_s: float,
    what: str,
) -> None:
    """Issue one request and block until it is answered, correlated on the reply.

    ``submit`` receives an ``on_done()`` and an ``on_error(message)`` and is expected to
    hand them to a handle method, e.g.::

        blocking_request(lambda ok, err: stage.move_to(12.5, on_done=ok, on_error=err),
                         timeout_s=130.0, what="probe move to 12.5 mm")

    The correlation is the whole point, and it is why this is not built on a completion
    event: an event on the bus carries neither a request id nor a target, so it cannot be
    matched to the request that caused it, and it races any other subscriber issuing
    requests of its own — a device panel's live move subscription, for instance.

    ``what`` names the operation in the error message; it is often the only thing in the
    log that says which axis stopped answering.
    """
    done = threading.Event()
    error: list[str] = []

    def on_ok() -> None:
        done.set()

    def on_err(message: str) -> None:
        error.append(message)
        done.set()

    submit(on_ok, on_err)

    if not done.wait(timeout_s):
        raise WorkerCallError(f"{what}: no reply within {timeout_s:.0f}s")
    if error:
        raise WorkerCallError(f"{what}: {error[0]}")


def wait_for_status(
    handle: BaseWorkerHandle,
    status: WorkerStatus = WorkerStatus.RUNNING,
    *,
    timeout_s: float,
    poll_s: float = DEFAULT_POLL_S,
) -> bool:
    """Poll until the handle reports ``status``, or the timeout expires.

    Polling rather than event-driven on purpose. ``WorkerState`` does publish a no-arg
    event on every transition, but a ``_start()`` that *raises* in the subprocess sends no
    reply at all — ``BaseWorker._on_start_cmd`` calls ``_start()`` before ``_reply_ok``
    with no try/except, and the exception is swallowed by the worker's ``TaskRunner``. So
    the failure case produces no transition and no error reply, and there would be
    nothing to wake on. The timeout is the only thing that distinguishes it from a slow
    start.

    Note that reaching ``RUNNING`` is a worker-lifecycle fact, not proof that the
    hardware link works: a device whose connection failure is non-fatal registers and
    reports RUNNING regardless. The first real command is what exercises the link.
    """
    clock = threading.Event()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if handle.state == status:
            return True
        clock.wait(poll_s)
    return handle.state == status
