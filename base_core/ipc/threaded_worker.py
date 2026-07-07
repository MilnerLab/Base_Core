"""
ThreadedWorker and ProducingThreadedWorker — per-worker serial dispatch.

In a subprocess with multiple workers all sharing a single EventBus, the connector
poll thread calls bus.publish() synchronously, so a slow handler in one worker blocks
every other worker.  These classes fix that by dispatching each callback onto the
worker's own TaskRunner, so the poll thread returns immediately and workers process
commands serially but independently.

Usage
-----
Command-only workers:

    class RotatorWorker(ThreadedWorker):
        @worker_thread
        def _on_rotate(self, msg: Rotate) -> None:
            self._rotator.rotate(msg.angle)
            self._reply_ok(msg)

Continuous-acquisition workers (spectrometer, oscilloscope):

    class SpectrometerWorker(ProducingThreadedWorker):
        def _start(self) -> None:
            ...
            self._start_producing(self._acquire_producer, on_item=self._on_acquired)

        def _pause(self) -> None:
            handle = self._stop_producing()
            if handle:
                handle.wait(timeout=5.0)   # drain before closing hardware
            ...

        def _resume(self) -> None:
            self._start_producing(self._acquire_producer, on_item=self._on_acquired)
"""
from __future__ import annotations

import logging
from functools import wraps
from typing import Callable, Iterable, TypeVar

from base_core.framework.concurrency.models import StreamHandle
from base_core.framework.concurrency.task_runner import TaskRunner
from base_core.framework.events.event_bus import EventBus
from base_core.ipc.subprocess_connector import SubprocessPipelineConnector
from base_core.ipc.worker import BaseWorker
from base_core.ipc.worker_messages import PauseWorker, ResumeWorker, StartWorker, StopWorker

log = logging.getLogger(__name__)

T = TypeVar("T")


def worker_thread(fn: Callable) -> Callable:
    """Dispatch a ThreadedWorker method onto the worker's own serial TaskRunner.

    The calling thread (connector poll loop) returns immediately; the method body
    runs on the worker's dedicated thread.
    """
    @wraps(fn)
    def wrapper(self: ThreadedWorker, *args, **kwargs) -> None:
        self._runner.run(
            lambda: fn(self, *args, **kwargs),
            on_error=lambda e: log.exception(
                "Unhandled error in %s.%s", type(self).__name__, fn.__name__
            ),
        )
    return wrapper


class ThreadedWorker(BaseWorker):
    """
    BaseWorker that dispatches all event callbacks onto its own serial TaskRunner.

    Commands for this worker are processed one-at-a-time on a dedicated thread in
    arrival order, without blocking other workers that share the subprocess.

    Concrete subclasses decorate domain handlers with @worker_thread.
    """

    def __init__(
        self,
        worker_id: str,
        bus: EventBus,
        connector: SubprocessPipelineConnector,
    ) -> None:
        super().__init__(worker_id, bus, connector)
        self._runner = TaskRunner(name=f"worker.{worker_id}")

    def deactivate(self) -> None:
        super().deactivate()  # unsubscribes all handlers first
        self._runner.shutdown(wait=False)

    @worker_thread
    def _on_start_cmd(self, msg: StartWorker) -> None:
        super()._on_start_cmd(msg)

    @worker_thread
    def _on_pause_cmd(self, msg: PauseWorker) -> None:
        super()._on_pause_cmd(msg)

    @worker_thread
    def _on_resume_cmd(self, msg: ResumeWorker) -> None:
        super()._on_resume_cmd(msg)

    @worker_thread
    def _on_stop_cmd(self, msg: StopWorker) -> None:
        super()._on_stop_cmd(msg)


class ProducingThreadedWorker(ThreadedWorker):
    """
    ThreadedWorker with a second independent TaskRunner for a continuous production loop.

    Provides _start_producing() / _stop_producing() so the acquisition stream and
    the command stream never block each other.  Use inside _start() / _stop():

        def _start(self) -> None:
            self._device.open()
            self._start_producing(self._my_producer, on_item=self._on_item)

        def _pause(self) -> None:
            handle = self._stop_producing()
            if handle:
                handle.wait(timeout=5.0)   # drain before closing hardware
            self._device.close()

        def _resume(self) -> None:
            self._device.open()
            self._start_producing(self._my_producer, on_item=self._on_item)
    """

    def __init__(
        self,
        worker_id: str,
        bus: EventBus,
        connector: SubprocessPipelineConnector,
    ) -> None:
        super().__init__(worker_id, bus, connector)
        self._prod_runner = TaskRunner(name=f"worker.{worker_id}.prod")
        self._prod_handle: StreamHandle | None = None

    def _start_producing(
        self,
        producer: Callable[[any], Iterable[T]],
        *,
        on_item: Callable[[T], None],
        on_error: Callable[[BaseException], None] | None = None,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        """Launch a lossless production stream on the dedicated production runner."""
        self._prod_handle = self._prod_runner.stream(
            producer,
            on_item=on_item,
            on_error=on_error,
            on_complete=on_complete,
        )

    def _stop_producing(self) -> StreamHandle | None:
        """Signal the production stream to stop. Returns the handle to wait on."""
        handle = self._prod_handle
        if handle is not None:
            handle.stop()
            self._prod_handle = None
        return handle

    def deactivate(self) -> None:
        self._stop_producing()
        super().deactivate()
        self._prod_runner.shutdown(wait=False)
