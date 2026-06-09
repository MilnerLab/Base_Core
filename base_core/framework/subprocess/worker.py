from __future__ import annotations

import collections
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Generic, Optional, TypeVar

from base_core.framework.subprocess.shared_memory.base_protocol import (
    ConfigureBuffer,
    ItemWritten,
    SlotGranted,
)
from base_core.framework.subprocess.shared_memory.models import (
    ItemDescriptor,
    SharedRingBufferSpec,
)
from messages import Message


TBuffer = TypeVar("TBuffer")
TData = TypeVar("TData")


def _not_attached(*_args, **_kwargs) -> None:
    raise RuntimeError("Worker has not been added to a SubprocessApp yet.")


class Worker(ABC):
    """
    Atomic unit of logic running in its own thread within a SubprocessApp.

    Workers are registered with SubprocessApp.add_worker() before the app
    starts. The app injects I/O callbacks (emit / reply_ok / reply_error),
    starts the worker thread, and routes incoming commands to handle().

    Subclasses override:
      - handle(msg, request_id): called from the stdin thread; must NOT block.
      - run(): runs in the worker's own thread; loop until self._stop is set.
    """

    name: str

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Injected by SubprocessApp.add_worker() before the first start().
        self._emit_fn: Callable[[Message], None] = _not_attached  # type: ignore[assignment]
        self._reply_ok_fn: Callable[[Optional[str], Optional[dict]], None] = _not_attached  # type: ignore[assignment]
        self._reply_error_fn: Callable[[Optional[str], str], None] = _not_attached  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # I/O helpers available inside run() and handle()
    # ------------------------------------------------------------------

    def emit(self, msg: Message) -> None:
        """Send an event to the main process. Thread-safe."""
        self._emit_fn(msg)

    def reply_ok(self, request_id: Optional[str], payload: Optional[dict] = None) -> None:
        """Reply to a command with success. Thread-safe."""
        self._reply_ok_fn(request_id, payload)

    def reply_error(self, request_id: Optional[str], error: str) -> None:
        """Reply to a command with an error. Thread-safe."""
        self._reply_error_fn(request_id, error)

    # ------------------------------------------------------------------
    # Dispatch hook (called from stdin thread — must not block)
    # ------------------------------------------------------------------

    def handle(self, msg: Message, request_id: Optional[str]) -> None:
        """Receive an incoming command. Default implementation is a no-op."""

    # ------------------------------------------------------------------
    # Worker thread entry point
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self) -> None:
        """Run the worker loop. Called in the worker's own thread.
        Poll self._stop.is_set() (or .wait()) to know when to exit."""

    # ------------------------------------------------------------------
    # Lifecycle (called by SubprocessApp, not by subclasses)
    # ------------------------------------------------------------------

    def _start(self, stop_event: threading.Event) -> None:
        self._stop = stop_event
        self._thread = threading.Thread(
            target=self._run_safe, name=f"worker.{self.name}", daemon=True
        )
        self._thread.start()

    def _join(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run_safe(self) -> None:
        try:
            self.run()
        except Exception:
            pass  # worker crash must not take down the subprocess


# ---------------------------------------------------------------------------
# CommandWorker
# ---------------------------------------------------------------------------

class CommandWorker(Worker, ABC):
    """
    Worker that processes commands from a queue in its own thread.

    handle() enqueues incoming commands immediately (non-blocking). run()
    drains the queue and calls _process() for each item, so hardware I/O
    or blocking calls belong inside _process(), not handle().
    """

    def __init__(self) -> None:
        super().__init__()
        self._queue: collections.deque[tuple[Message, Optional[str]]] = collections.deque()
        self._queue_event = threading.Event()

    def handle(self, msg: Message, request_id: Optional[str]) -> None:
        self._queue.append((msg, request_id))
        self._queue_event.set()

    def run(self) -> None:
        while not self._stop.is_set():
            if not self._queue_event.wait(timeout=0.1):
                continue
            while self._queue:
                msg, request_id = self._queue.popleft()
                if not self._queue:
                    self._queue_event.clear()
                try:
                    self._process(msg, request_id)
                except Exception as exc:
                    self.reply_error(request_id, str(exc))

    @abstractmethod
    def _process(self, msg: Message, request_id: Optional[str]) -> None:
        """Handle one command. Call self.reply_ok / self.reply_error as needed."""


# ---------------------------------------------------------------------------
# ProducerWorker
# ---------------------------------------------------------------------------

class ProducerWorker(Worker, ABC, Generic[TBuffer, TData]):
    """
    Worker that continuously acquires data and writes it into a shared ring buffer.

    Replicates the GrantedSlotWriterProcessBase grant-loop logic as a Worker,
    so multiple ProducerWorkers (and other Worker types) can coexist in one
    SubprocessApp without interfering with each other.

    Each ProducerWorker must have a unique buffer_id that matches the value
    the main process uses in ConfigureBuffer / SlotGranted / ItemWritten.

    Incoming handle() calls for ConfigureBuffer and SlotGranted are processed
    synchronously on the stdin thread (they are quick lock/event operations).
    The acquisition loop runs entirely in the worker's own thread.
    """

    def __init__(self, *, buffer_id: str = "") -> None:
        super().__init__()
        self._buffer_id = buffer_id
        self._buffer: Optional[TBuffer] = None
        self._configured = threading.Event()
        self._grants_lock = threading.Lock()
        self._granted_slots: collections.deque[ItemDescriptor] = collections.deque()
        self._grants_available = threading.Event()
        self._live_slots: set[int] = set()

    @property
    def buffer(self) -> TBuffer:
        if self._buffer is None:
            raise RuntimeError(f"Worker '{self.name}': buffer not configured yet.")
        return self._buffer

    # ------------------------------------------------------------------
    # Abstract hooks (subclass implements)
    # ------------------------------------------------------------------

    @abstractmethod
    def attach_buffer(self, spec: SharedRingBufferSpec) -> TBuffer:
        """Attach the domain-specific shared buffer by name/spec."""

    @abstractmethod
    def acquire(self) -> Optional[TData]:
        """Acquire one data item, or None if not ready (will retry)."""

    @abstractmethod
    def write_to_slot(self, *, data: TData, item: ItemDescriptor) -> None:
        """Write one acquired item into the granted slot."""

    def on_buffer_configured(self) -> None:
        """Optional hook called after the buffer is successfully attached."""

    # ------------------------------------------------------------------
    # Command dispatch (called from stdin thread)
    # ------------------------------------------------------------------

    def handle(self, msg: Message, request_id: Optional[str]) -> None:
        if isinstance(msg, ConfigureBuffer):
            if msg.buffer_id == self._buffer_id:
                self._buffer = self.attach_buffer(msg.spec)
                self._configured.set()
                self.on_buffer_configured()
                self.reply_ok(request_id)
        elif isinstance(msg, SlotGranted):
            if msg.buffer_id == self._buffer_id:
                item = ItemDescriptor(slot=msg.slot, item_id=msg.item_id, timestamp_ns=0)
                with self._grants_lock:
                    if msg.slot in self._live_slots:
                        self.reply_error(request_id, f"Slot {msg.slot} granted while still live.")
                        return
                    self._live_slots.add(msg.slot)
                    self._granted_slots.append(item)
                    self._grants_available.set()
                self.reply_ok(request_id)

    # ------------------------------------------------------------------
    # Acquisition loop (runs in worker thread)
    # ------------------------------------------------------------------

    def run(self) -> None:
        while not self._stop.is_set():
            if not self._configured.wait(timeout=0.05):
                continue

            granted = self._next_grant(timeout=0.05)
            if granted is None:
                continue
            if self._stop.is_set():
                break

            data = self.acquire()
            if data is None:
                time.sleep(0.001)
                continue

            item = ItemDescriptor(
                slot=granted.slot,
                item_id=granted.item_id,
                timestamp_ns=time.time_ns(),
            )
            self.write_to_slot(data=data, item=item)
            self._retire_grant(item.slot)

            self.emit(
                ItemWritten(
                    slot=item.slot,
                    item_id=item.item_id,
                    timestamp_ns=item.timestamp_ns,
                    buffer_id=self._buffer_id,
                )
            )

    # ------------------------------------------------------------------
    # Grant queue helpers (mirror of GrantedSlotWriterProcessBase)
    # ------------------------------------------------------------------

    def _next_grant(self, timeout: float) -> Optional[ItemDescriptor]:
        if not self._grants_available.wait(timeout=timeout):
            return None
        with self._grants_lock:
            if not self._granted_slots:
                self._grants_available.clear()
                return None
            return self._granted_slots[0]

    def _retire_grant(self, slot: int) -> None:
        with self._grants_lock:
            if self._granted_slots and self._granted_slots[0].slot == slot:
                self._granted_slots.popleft()
            self._live_slots.discard(slot)
            if not self._granted_slots:
                self._grants_available.clear()
