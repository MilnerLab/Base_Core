from __future__ import annotations

import threading
from typing import Optional

from base_core.framework.subprocess.messages import Kind, MessageRegistry
from base_core.framework.subprocess.worker import Worker
from base_core.framework.subprocess.worker_protocol import (
    StartWorker,
    StopWorker,
    WorkerError,
)
from json_process_base import JsonlStdioAppBase


class SubprocessApp(JsonlStdioAppBase):
    """
    Subprocess entry point that hosts multiple independent Workers.

    Workers are registered before run() via add_worker(). Each worker:
      - Starts in STOPPED state; requires an explicit StartWorker command.
      - Has its own thread and _worker_stop event — can be restarted independently.
      - Is addressed by name via the "target" field on incoming commands.

    SubprocessApp handles StartWorker and StopWorker itself (no target needed).
    Commands without a target fall through to self.on() handlers for app-level use.

    Usage::

        app = SubprocessApp(registry=base_registry())
        app.add_worker(RotatorWorker())
        app.add_worker(PressureWorker())
        app.run()
    """

    def __init__(
        self,
        registry: MessageRegistry,
        *,
        source: Optional[str] = None,
    ) -> None:
        # Always include worker lifecycle messages in the registry.
        combined = registry.extend(StartWorker, StopWorker, WorkerError)
        super().__init__(combined, source=source)
        self._workers: dict[str, Worker] = {}
        self.on(StartWorker, self._on_start_worker)
        self.on(StopWorker, self._on_stop_worker)

    # ------------------------------------------------------------------
    # Worker registration (call before run())
    # ------------------------------------------------------------------

    def add_worker(self, worker: Worker) -> None:
        """Register a worker. Must be called before run()."""
        if worker.messages:
            self._registry.register(*worker.messages)
        worker._inject_stop(self._stop)
        worker._emit_fn = self.emit
        worker._reply_ok_fn = self.reply_ok
        worker._reply_error_fn = self.reply_error
        self._workers[worker.name] = worker

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        # Workers start STOPPED; they wait for StartWorker commands.
        # Inject stop event for workers added before run() — already done in add_worker.
        try:
            super().run()  # starts stdin thread, calls main(), sets _stop on exit
        finally:
            # On subprocess shutdown, stop all running workers.
            for worker in self._workers.values():
                worker._join()

    def main(self, stop_event: threading.Event) -> None:
        """All work happens in worker threads; wait for shutdown signal."""
        stop_event.wait()

    # ------------------------------------------------------------------
    # Lifecycle message handlers (app-level, no target)
    # ------------------------------------------------------------------

    def _on_start_worker(self, msg: StartWorker, request_id: Optional[str]) -> None:
        worker = self._workers.get(msg.worker_name)
        if worker is None:
            self.reply_error(request_id, f"Unknown worker: {msg.worker_name!r}")
            return
        if worker._thread is not None and worker._thread.is_alive():
            self.reply_error(request_id, f"Worker {msg.worker_name!r} is already running.")
            return
        worker._start()
        self.reply_ok(request_id)

    def _on_stop_worker(self, msg: StopWorker, request_id: Optional[str]) -> None:
        worker = self._workers.get(msg.worker_name)
        if worker is None:
            self.reply_error(request_id, f"Unknown worker: {msg.worker_name!r}")
            return
        worker._stop_worker()
        self.reply_ok(request_id)

    # ------------------------------------------------------------------
    # Command dispatch (routes by "target" field)
    # ------------------------------------------------------------------

    def _dispatch(self, envelope: dict) -> None:
        if envelope.get("kind") != Kind.COMMAND:
            return

        target = envelope.get("target")
        if not isinstance(target, str):
            # No target: app-level dispatch (StartWorker, StopWorker, self.on() handlers)
            super()._dispatch(envelope)
            return

        raw_id = envelope.get("id")
        request_id: Optional[str] = raw_id if isinstance(raw_id, str) else None

        worker = self._workers.get(target)
        if worker is None:
            self.reply_error(request_id, f"Unknown worker target: {target!r}")
            return

        msg = self._registry.decode(envelope)
        if msg is None:
            self.reply_error(request_id, f"Unknown command: {envelope.get('name')!r}")
            return

        try:
            worker._receive(msg, request_id)
        except Exception as exc:
            self.reply_error(request_id, str(exc))
