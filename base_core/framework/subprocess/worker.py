from __future__ import annotations

import collections
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Optional

from base_core.framework.events.event_bus import EventBus
from base_core.framework.subprocess.shared_memory.shared_memory_base_messages import (
    ItemAck,
)
from messages import Message


def _not_attached(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("Worker has not been added to a SubprocessApp yet.")


class Worker(ABC):
    """
    Atomic unit of logic running in its own thread within a SubprocessApp.

    All messages arrive via the subprocess internal EventBus: SubprocessApp
    publishes every decoded stdin command to the bus, and each worker subscribes
    to the message types it declares in `bus_messages`.

    Thread-bridge rule
    ------------------
    `_on_bus_message` is called on the stdin thread (synchronously by
    EventBus.publish). It must never block — it enqueues immediately and
    returns. The worker thread dequeues and calls `handle(msg)`, which may
    block freely (hardware I/O, slow computation).

    When the worker is busy the queue absorbs incoming messages. The ring
    buffer's fixed slot count provides natural backpressure for data messages.

    Lifecycle
    ---------
    Workers start in STOPPED state. An explicit StartWorker command (handled
    by SubprocessApp) is required before the worker does anything. StopWorker
    stops just this worker; the subprocess-level _stop event stops everything.

    Buffer helpers
    --------------
    Declare buffer types via `write_buffer_cls` / `read_buffer_cls`. SubprocessApp
    attaches live buffers (populated by ConfigureBuffer messages) via
    `_process_buffers`. Access them in handle() via `self._process_buffers[buffer_id]`.

    Convenience helpers `ack()` and `reply()` / `reply_ok()` / `reply_error()` are
    provided so subclasses rarely need to call emit() directly.
    """

    name: str
    bus_messages: ClassVar[tuple[type[Message], ...]] = ()
    write_buffer_cls: ClassVar[Optional[type]] = None
    read_buffer_cls: ClassVar[tuple[type, ...]] = ()

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._worker_stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process_buffers: dict[str, Any] = {}
        self._internal_bus: Optional[EventBus] = None
        self._queue: collections.deque[Message] = collections.deque()
        self._queue_event = threading.Event()
        # Injected by SubprocessApp.add_worker()
        self._emit_fn: Callable[[Message], None] = _not_attached  # type: ignore[assignment]
        self._reply_fn: Callable[..., None] = _not_attached       # type: ignore[assignment]
        self._reply_ok_fn: Callable[..., None] = _not_attached    # type: ignore[assignment]
        self._reply_error_fn: Callable[..., None] = _not_attached # type: ignore[assignment]

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def emit(self, msg: Message) -> None:
        """Send an event to the main process. Thread-safe."""
        self._emit_fn(msg)

    def reply(self, request_id: Optional[str], reply_msg: Message) -> None:
        """Send any typed reply message. Thread-safe."""
        self._reply_fn(request_id, reply_msg)

    def reply_ok(self, request_id: Optional[str]) -> None:
        """Send an OKMessage reply. Thread-safe."""
        self._reply_ok_fn(request_id)

    def reply_error(self, request_id: Optional[str], error: str) -> None:
        """Send an ErrorMessage reply. Thread-safe."""
        self._reply_error_fn(request_id, error)

    def ack(self, slot: int, item_id: int, buffer_id: str) -> None:
        """Emit an ItemAck for a consumed slot. Thread-safe."""
        self.emit(ItemAck(slot=slot, item_id=item_id, consumer_id=self.name, buffer_id=buffer_id))

    # ------------------------------------------------------------------
    # Bus subscription
    # ------------------------------------------------------------------

    def _setup_bus_subscriptions(self) -> None:
        """Called by SubprocessApp.add_worker() after injection."""
        if self._internal_bus is None:
            return
        for msg_cls in self.bus_messages:
            self._internal_bus.subscribe(msg_cls, self._on_bus_message)

    def _on_bus_message(self, msg: Message) -> None:
        """
        Called on the stdin thread by EventBus.publish(). Must not block.
        Filters by consumer_id, then enqueues for the worker thread.
        """
        if msg.consumer_id is not None and msg.consumer_id != self.name:
            return
        self._queue.append(msg)
        self._queue_event.set()

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    @property
    def _should_stop(self) -> bool:
        return self._stop.is_set() or self._worker_stop.is_set()

    def run(self) -> None:
        """
        Default run loop: drain queue, call handle() for each message.
        Override for workers that need a continuous acquisition loop.
        """
        while not self._should_stop:
            if not self._queue_event.wait(timeout=0.1):
                continue
            while self._queue:
                msg = self._queue.popleft()
                if not self._queue:
                    self._queue_event.clear()
                try:
                    self.handle(msg)
                except Exception as exc:
                    self.reply_error(msg.request_id, str(exc))

    def handle(self, msg: Message) -> None:
        """Process an incoming message. Override to dispatch by type."""

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        """Reset transient state before (re)starting. Called before each start()."""
        self._queue.clear()
        self._queue_event.clear()

    def start(self) -> None:
        """Called after _reset(), before the thread begins. Override to open hardware."""

    def close(self) -> None:
        """Called after the thread exits. Override to close hardware."""

    # ------------------------------------------------------------------
    # Lifecycle (called by SubprocessApp — not by user code)
    # ------------------------------------------------------------------

    def _inject_stop(self, subprocess_stop: threading.Event) -> None:
        self._stop = subprocess_stop

    def _start(self) -> None:
        self._worker_stop = threading.Event()
        self._reset()
        self.start()
        self._thread = threading.Thread(
            target=self._run_safe, name=f"worker.{self.name}", daemon=True
        )
        self._thread.start()

    def _stop_worker(self, timeout: float = 5.0) -> None:
        self._worker_stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self.close()

    def _join(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run_safe(self) -> None:
        try:
            self.run()
        except Exception as exc:
            try:
                from base_core.framework.subprocess.worker_protocol import WorkerError
                self.emit(WorkerError(worker_name=self.name, error=str(exc)))
            except Exception:
                pass
