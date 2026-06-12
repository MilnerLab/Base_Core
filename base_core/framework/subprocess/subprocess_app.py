from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from base_core.framework.events.event_bus import EventBus
from base_core.framework.subprocess.json_process_base import JsonlStdioAppBase
from base_core.framework.subprocess.messages import Kind, MessageRegistry
from base_core.framework.subprocess.shared_memory.shared_memory_base_messages import ConfigureBuffer
from base_core.framework.subprocess.worker import Worker
from base_core.framework.subprocess.worker_protocol import (
    StartWorker,
    StopWorker,
    WorkerError,
)


class SubprocessApp(JsonlStdioAppBase):
    """
    Subprocess entry point that hosts multiple independent Workers.

    Every decoded stdin command is published to the internal EventBus.
    Workers subscribe to the message types they care about via `bus_messages`;
    the bus is the single dispatch mechanism — there is no separate target routing.

    App-level lifecycle messages (StartWorker, StopWorker, ConfigureBuffer) are
    handled via bus subscriptions set up in __init__.

    Usage::

        app = SubprocessApp(registry=base_registry())
        app.add_worker(RotatorWorker())
        app.add_worker(SpectrometerWorker())
        app.run()
    """

    def __init__(
        self,
        registry: MessageRegistry,
        *,
        source: Optional[str] = None,
    ) -> None:
        combined = registry.extend(StartWorker, StopWorker, WorkerError)
        super().__init__(combined, source=source)
        self._workers: dict[str, Worker] = {}
        self._buffers: dict[str, Any] = {}
        self._buffer_factories: dict[str, Callable[..., Any]] = {}
        self._internal_bus = EventBus()

        # App-level lifecycle via bus subscriptions (not self.on())
        self._internal_bus.subscribe(StartWorker, self._on_start_worker)
        self._internal_bus.subscribe(StopWorker, self._on_stop_worker)
        self._internal_bus.subscribe(ConfigureBuffer, self._on_configure_buffer)

    # ------------------------------------------------------------------
    # Worker registration
    # ------------------------------------------------------------------

    def add_worker(self, worker: Worker) -> None:
        """Register a worker. Must be called before run()."""
        if worker.bus_messages:
            self._registry.register(*worker.bus_messages)
        worker._inject_stop(self._stop)
        worker._emit_fn = self.emit
        worker._reply_fn = self.reply
        worker._reply_ok_fn = self.reply_ok
        worker._reply_error_fn = self.reply_error
        worker._process_buffers = self._buffers
        worker._internal_bus = self._internal_bus
        self._workers[worker.name] = worker

        # Auto-register buffer factories from worker ClassVar declarations.
        if worker.write_buffer_cls is not None:
            bid = getattr(worker, 'buffer_id', worker.name)
            self._buffer_factories.setdefault(bid, worker.write_buffer_cls.attach)

        for bid, cls in zip(
            getattr(worker, 'input_buffer_ids', []),
            worker.read_buffer_cls,
        ):
            self._buffer_factories.setdefault(bid, cls.attach)

        worker._setup_bus_subscriptions()

    def add_buffer(self, buffer_id: str, factory: Callable[..., Any]) -> None:
        """Explicitly register a buffer attachment factory. Must be called before run()."""
        self._buffer_factories[buffer_id] = factory

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            super().run()
        finally:
            for worker in self._workers.values():
                worker._join()

    def main(self, stop_event: threading.Event) -> None:
        stop_event.wait()

    # ------------------------------------------------------------------
    # Dispatch — all stdin commands go to the internal bus
    # ------------------------------------------------------------------

    def _dispatch(self, envelope: dict) -> None:
        if envelope.get("kind") != Kind.COMMAND:
            return

        raw_id = envelope.get("id")
        request_id: Optional[str] = raw_id if isinstance(raw_id, str) else None

        msg = self._registry.decode(envelope)
        if msg is None:
            if request_id is not None:
                self.reply_error(request_id, f"Unknown command: {envelope.get('name')!r}")
            return

        # Every decoded command is published to the internal bus.
        # Workers and SubprocessApp itself react via their bus subscriptions.
        self._internal_bus.publish(msg)

    # ------------------------------------------------------------------
    # App-level bus handlers
    # ------------------------------------------------------------------

    def _on_start_worker(self, msg: StartWorker) -> None:
        worker = self._workers.get(msg.worker_name)
        if worker is None:
            self.reply_error(msg.request_id, f"Unknown worker: {msg.worker_name!r}")
            return
        if worker._thread is not None and worker._thread.is_alive():
            self.reply_error(msg.request_id, f"Worker {msg.worker_name!r} is already running.")
            return
        worker._start()
        self.reply_ok(msg.request_id)

    def _on_stop_worker(self, msg: StopWorker) -> None:
        worker = self._workers.get(msg.worker_name)
        if worker is None:
            self.reply_error(msg.request_id, f"Unknown worker: {msg.worker_name!r}")
            return
        worker._stop_worker()
        self.reply_ok(msg.request_id)

    def _on_configure_buffer(self, msg: ConfigureBuffer) -> None:
        if msg.buffer_id not in self._buffers:
            factory = self._buffer_factories.get(msg.buffer_id)
            if factory is None:
                self.reply_error(msg.request_id, f"No factory registered for buffer {msg.buffer_id!r}")
                return
            self._buffers[msg.buffer_id] = factory(msg.spec)
        self.reply_ok(msg.request_id)
