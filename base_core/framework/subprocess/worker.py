from __future__ import annotations

import collections
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Generic, Optional, TypeVar

from base_core.framework.subprocess.shared_memory.base_protocol import (
    ConfigureBuffer,
    ItemAvailable,
    ItemAck,
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


def _not_attached(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("Worker has not been added to a SubprocessApp yet.")


# ---------------------------------------------------------------------------
# Worker — the base class
# ---------------------------------------------------------------------------

class Worker(ABC):
    """
    Atomic unit of logic running in its own thread within a SubprocessApp.

    Lifecycle (controlled by SubprocessApp via StartWorker / StopWorker commands):
      - Workers are registered in STOPPED state.
      - An explicit StartWorker command is required before the worker does anything.
      - StopWorker stops just this worker without touching the rest of the subprocess.
      - When the subprocess itself shuts down, all workers are stopped via the shared _stop event.

    I/O helpers (emit / reply_ok / reply_error) are injected by SubprocessApp.add_worker()
    and are thread-safe — safe to call from the worker thread.
    """

    name: str
    messages: ClassVar[list[type[Message]]] = []

    def __init__(self) -> None:
        self._stop = threading.Event()           # subprocess-level stop (injected)
        self._worker_stop = threading.Event()    # per-worker stop (set by StopWorker)
        self._thread: Optional[threading.Thread] = None
        # Injected by SubprocessApp.add_worker()
        self._emit_fn: Callable[[Message], None] = _not_attached  # type: ignore[assignment]
        self._reply_ok_fn: Callable[..., None] = _not_attached    # type: ignore[assignment]
        self._reply_error_fn: Callable[..., None] = _not_attached  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # I/O helpers (available inside run() and handle())
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

    @property
    def _should_stop(self) -> bool:
        """True when either the subprocess or this specific worker should stop."""
        return self._stop.is_set() or self._worker_stop.is_set()

    # ------------------------------------------------------------------
    # Dispatch hook (called from stdin thread — must not block)
    # ------------------------------------------------------------------

    def handle(self, msg: Message, request_id: Optional[str]) -> None:
        """Receive an incoming targeted command. Default is a no-op."""

    # ------------------------------------------------------------------
    # Worker thread entry point (override in subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self) -> None:
        """Worker loop. Poll _should_stop periodically."""

    # ------------------------------------------------------------------
    # Resettable state (override in subclasses that have internal state)
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        """Reset internal state before (re)starting. Called before each _start()."""

    # ------------------------------------------------------------------
    # Lifecycle (called by SubprocessApp — not by user code)
    # ------------------------------------------------------------------

    def _inject_stop(self, subprocess_stop: threading.Event) -> None:
        """Called once when the worker is registered with a SubprocessApp."""
        self._stop = subprocess_stop

    def _start(self) -> None:
        """Start the worker thread. Called on each StartWorker command."""
        self._worker_stop = threading.Event()
        self._reset()
        self._thread = threading.Thread(
            target=self._run_safe, name=f"worker.{self.name}", daemon=True
        )
        self._thread.start()

    def _stop_worker(self, timeout: float = 5.0) -> None:
        """Stop only this worker. Called on StopWorker command."""
        self._worker_stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _join(self, timeout: float = 5.0) -> None:
        """Join thread on subprocess shutdown (worker exits via _stop)."""
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


# ---------------------------------------------------------------------------
# CommandWorker
# ---------------------------------------------------------------------------

class CommandWorker(Worker, ABC):
    """
    Worker that processes commands from a queue in its own thread.

    handle() enqueues incoming commands immediately (non-blocking from stdin thread).
    _process() runs in the worker thread — safe for blocking hardware I/O.
    """

    def __init__(self) -> None:
        super().__init__()
        self._queue: collections.deque[tuple[Message, Optional[str]]] = collections.deque()
        self._queue_event = threading.Event()

    def handle(self, msg: Message, request_id: Optional[str]) -> None:
        self._queue.append((msg, request_id))
        self._queue_event.set()

    def run(self) -> None:
        while not self._should_stop:
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

    def _reset(self) -> None:
        self._queue.clear()
        self._queue_event.clear()

    @abstractmethod
    def _process(self, msg: Message, request_id: Optional[str]) -> None:
        """Handle one command. Call self.reply_ok / self.reply_error as needed."""


# ---------------------------------------------------------------------------
# ProducerWorker
# ---------------------------------------------------------------------------

class ProducerWorker(Worker, ABC, Generic[TBuffer, TData]):
    """
    Worker that continuously acquires data and writes to a shared ring buffer.

    Handles ConfigureBuffer and SlotGranted (sent by OutputBufferHandle on startup).
    After receiving the first SlotGranted, enters the acquire → write → emit loop.
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

    def _reset(self) -> None:
        self._buffer = None
        self._configured.clear()
        with self._grants_lock:
            self._granted_slots.clear()
            self._live_slots.clear()
            self._grants_available.clear()

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def attach_buffer(self, spec: SharedRingBufferSpec) -> TBuffer: ...

    @abstractmethod
    def acquire(self) -> Optional[TData]:
        """Return the next data item, or None if not ready yet (will retry)."""

    @abstractmethod
    def write_to_slot(self, *, data: TData, item: ItemDescriptor) -> None: ...

    def on_buffer_configured(self) -> None:
        """Optional hook after the buffer is attached."""

    # ------------------------------------------------------------------
    # Command dispatch (called from stdin thread — non-blocking)
    # ------------------------------------------------------------------

    def handle(self, msg: Message, request_id: Optional[str]) -> None:
        if isinstance(msg, ConfigureBuffer) and msg.buffer_id == self._buffer_id:
            self._buffer = self.attach_buffer(msg.spec)
            self._configured.set()
            self.on_buffer_configured()
            self.reply_ok(request_id)
        elif isinstance(msg, SlotGranted) and msg.buffer_id == self._buffer_id:
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
    # Acquisition loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        while not self._should_stop:
            if not self._configured.wait(timeout=0.05):
                continue
            granted = self._next_grant(timeout=0.05)
            if granted is None:
                continue
            if self._should_stop:
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
            self.emit(ItemWritten(
                slot=item.slot,
                item_id=item.item_id,
                timestamp_ns=item.timestamp_ns,
                buffer_id=self._buffer_id,
            ))

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


# ---------------------------------------------------------------------------
# ConsumerWorker
# ---------------------------------------------------------------------------

class ConsumerWorker(Worker, ABC, Generic[TBuffer]):
    """
    Worker that reads from a single input shared-memory buffer.

    Handles ConfigureBuffer (to attach the input buffer) and ItemAvailable
    (forwarded by InputBufferHandle on the main process side).
    Subclass implements on_item() to do computation and must call ack() when done.

    For a worker that reads multiple input buffers, see ProcessorWorker.
    """

    def __init__(self, *, buffer_id: str = "") -> None:
        super().__init__()
        self._buffer_id = buffer_id
        self._buffer: Optional[TBuffer] = None
        self._item_queue: collections.deque[ItemAvailable] = collections.deque()
        self._item_event = threading.Event()

    @property
    def buffer(self) -> TBuffer:
        if self._buffer is None:
            raise RuntimeError(f"Worker '{self.name}': buffer not configured yet.")
        return self._buffer

    def _reset(self) -> None:
        self._buffer = None
        self._item_queue.clear()
        self._item_event.clear()

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def attach_buffer(self, spec: SharedRingBufferSpec) -> TBuffer: ...

    @abstractmethod
    def on_item(self, slot: int, item_id: int, timestamp_ns: int) -> None:
        """Process one input item. Must call self.ack(slot, item_id) when done."""

    # ------------------------------------------------------------------
    # Ack helper
    # ------------------------------------------------------------------

    def ack(self, slot: int, item_id: int) -> None:
        """Signal to the main process that this slot has been consumed."""
        self.emit(ItemAck(
            slot=slot,
            item_id=item_id,
            consumer_id=self.name,
            buffer_id=self._buffer_id,
        ))

    # ------------------------------------------------------------------
    # Command dispatch (stdin thread)
    # ------------------------------------------------------------------

    def handle(self, msg: Message, request_id: Optional[str]) -> None:
        if isinstance(msg, ConfigureBuffer) and msg.buffer_id == self._buffer_id:
            self._buffer = self.attach_buffer(msg.spec)
            self.reply_ok(request_id)
        elif isinstance(msg, ItemAvailable) and msg.buffer_id == self._buffer_id:
            self._item_queue.append(msg)
            self._item_event.set()

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        while not self._should_stop:
            if not self._item_event.wait(timeout=0.1):
                continue
            while self._item_queue:
                item = self._item_queue.popleft()
                if not self._item_queue:
                    self._item_event.clear()
                self.on_item(item.slot, item.item_id, item.timestamp_ns)


# ---------------------------------------------------------------------------
# ProcessorWorker
# ---------------------------------------------------------------------------

class ProcessorWorker(Worker, ABC):
    """
    Worker that reads from multiple input shared-memory buffers and writes to
    one output buffer.

    Combines ConsumerWorker (multiple inputs) + ProducerWorker (one output) in
    one thread. The subclass decides when all required inputs are present
    by overriding _inputs_ready() and performs computation in process().

    Typical subclass pattern:
      input_buffer_ids = ["spectrometer", "camera"]
      output_buffer_id = "ion_result"

      def _inputs_ready(self): return all(bid in self._pending for bid in self.input_buffer_ids)
      def process(self, pending): ...  # read, compute, write, ack all
    """

    input_buffer_ids: list[str] = []
    output_buffer_id: str = ""

    def __init__(self) -> None:
        super().__init__()
        self._input_buffers: dict[str, Any] = {}
        self._output_buffer: Optional[Any] = None
        self._pending: dict[str, ItemAvailable] = {}
        self._item_event = threading.Event()
        self._grants_lock = threading.Lock()
        self._granted_slots: collections.deque[ItemDescriptor] = collections.deque()
        self._grants_available = threading.Event()
        self._live_slots: set[int] = set()

    def _reset(self) -> None:
        self._input_buffers.clear()
        self._output_buffer = None
        self._pending.clear()
        self._item_event.clear()
        with self._grants_lock:
            self._granted_slots.clear()
            self._live_slots.clear()
            self._grants_available.clear()

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def attach_input_buffer(self, buffer_id: str, spec: SharedRingBufferSpec) -> Any: ...

    @abstractmethod
    def attach_output_buffer(self, spec: SharedRingBufferSpec) -> Any: ...

    @abstractmethod
    def write_to_slot(self, *, data: Any, item: ItemDescriptor) -> None: ...

    def _inputs_ready(self) -> bool:
        """Return True when the worker has enough pending inputs to process.
        Default: all declared input_buffer_ids have a pending item."""
        return all(bid in self._pending for bid in self.input_buffer_ids)

    @abstractmethod
    def process(self, pending: dict[str, ItemAvailable]) -> Any:
        """
        Perform computation. Return output data to write to the output slot,
        or None to skip writing. Must call self.ack_input() for each consumed slot.
        """

    # ------------------------------------------------------------------
    # Ack helper
    # ------------------------------------------------------------------

    def ack_input(self, buffer_id: str, slot: int, item_id: int) -> None:
        self.emit(ItemAck(
            slot=slot,
            item_id=item_id,
            consumer_id=self.name,
            buffer_id=buffer_id,
        ))

    # ------------------------------------------------------------------
    # Command dispatch (stdin thread)
    # ------------------------------------------------------------------

    def handle(self, msg: Message, request_id: Optional[str]) -> None:
        # Input buffer configuration
        if isinstance(msg, ConfigureBuffer) and msg.buffer_id in self.input_buffer_ids:
            self._input_buffers[msg.buffer_id] = self.attach_input_buffer(msg.buffer_id, msg.spec)
            self.reply_ok(request_id)
        # Output buffer configuration
        elif isinstance(msg, ConfigureBuffer) and msg.buffer_id == self.output_buffer_id:
            self._output_buffer = self.attach_output_buffer(msg.spec)
            self.reply_ok(request_id)
        # Slot granted for output
        elif isinstance(msg, SlotGranted) and msg.buffer_id == self.output_buffer_id:
            item = ItemDescriptor(slot=msg.slot, item_id=msg.item_id, timestamp_ns=0)
            with self._grants_lock:
                if msg.slot in self._live_slots:
                    self.reply_error(request_id, f"Slot {msg.slot} granted while still live.")
                    return
                self._live_slots.add(msg.slot)
                self._granted_slots.append(item)
                self._grants_available.set()
            self.reply_ok(request_id)
        # Input item available
        elif isinstance(msg, ItemAvailable) and msg.buffer_id in self.input_buffer_ids:
            self._pending[msg.buffer_id] = msg
            self._item_event.set()

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        while not self._should_stop:
            if not self._item_event.wait(timeout=0.1):
                continue
            self._item_event.clear()
            if not self._inputs_ready():
                continue

            snapshot = dict(self._pending)
            self._pending.clear()

            # If this worker has an output buffer, wait for a slot grant
            if self.output_buffer_id:
                granted = self._next_grant(timeout=1.0)
                if granted is None or self._should_stop:
                    # Return pending items to queue (re-set event so we retry)
                    self._pending.update(snapshot)
                    self._item_event.set()
                    continue
                item = ItemDescriptor(
                    slot=granted.slot,
                    item_id=granted.item_id,
                    timestamp_ns=time.time_ns(),
                )
            else:
                item = None  # type: ignore[assignment]

            data = self.process(snapshot)

            if item is not None and data is not None:
                self.write_to_slot(data=data, item=item)
                self._retire_grant(item.slot)
                self.emit(ItemWritten(
                    slot=item.slot,
                    item_id=item.item_id,
                    timestamp_ns=item.timestamp_ns,
                    buffer_id=self.output_buffer_id,
                ))

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
