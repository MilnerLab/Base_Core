from __future__ import annotations

import threading
from typing import Optional

from json_process_base import JsonlStdioAppBase
from messages import Kind, MessageRegistry
from base_core.framework.subprocess.worker import Worker


class SubprocessApp(JsonlStdioAppBase):
    """
    Subprocess entry point that hosts multiple Workers.

    Workers are registered before run() via add_worker(). Each worker gets its
    own thread; the SubprocessApp owns the stdin/stdout loop and routes each
    incoming command to the worker named in the envelope's "target" field.

    Commands without a "target" fall through to handlers registered via the
    inherited self.on() mechanism (unchanged from JsonlStdioAppBase), so
    app-level commands (e.g. a global ping/stop) still work as before.

    The main() implementation simply waits for the stop event — all real work
    happens in worker threads.

    Usage::

        app = SubprocessApp(registry=base_registry().extend(MoveRotator, ...))
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
        super().__init__(registry, source=source)
        self._workers: dict[str, Worker] = {}

    # ------------------------------------------------------------------
    # Worker registration (call before run())
    # ------------------------------------------------------------------

    def add_worker(self, worker: Worker) -> None:
        """Register a worker. Must be called before run()."""
        worker._emit_fn = self.emit
        worker._reply_ok_fn = self.reply_ok
        worker._reply_error_fn = self.reply_error
        self._workers[worker.name] = worker

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        for worker in self._workers.values():
            worker._start(self._stop)
        try:
            super().run()  # starts stdin thread, calls main(), sets _stop on exit
        finally:
            for worker in self._workers.values():
                worker._join()

    def main(self, stop_event: threading.Event) -> None:
        """All work happens in worker threads; just wait for shutdown signal."""
        stop_event.wait()

    # ------------------------------------------------------------------
    # Command dispatch (routes by "target" field)
    # ------------------------------------------------------------------

    def _dispatch(self, envelope: dict) -> None:
        if envelope.get("kind") != Kind.COMMAND:
            return

        target = envelope.get("target")
        if not isinstance(target, str):
            # No target: fall through to app-level self.on() handlers.
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
            worker.handle(msg, request_id)
        except Exception as exc:
            self.reply_error(request_id, str(exc))
