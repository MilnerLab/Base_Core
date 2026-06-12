from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Optional

from base_core.framework.app.app_message import AppMessage, MessageLevel
from base_core.framework.app.service_status import ServiceStatus
from base_core.framework.concurrency.models import StreamHandle
from base_core.framework.concurrency.task_runner import TaskRunner
from base_core.framework.events.event_bus import EventBus
from base_core.framework.subprocess.json_endpoint import JsonlSubprocessEndpoint
from base_core.framework.subprocess.messages import ErrorMessage, Message
from base_core.framework.subprocess.shared_memory.shared_memory_base_messages import (
    ConfigureBuffer,
    ItemAck,
    ItemWritten,
    SlotGranted,
)
from base_core.framework.subprocess.worker_handle import WorkerHandle
from base_core.framework.subprocess.worker_protocol import WorkerError


# ---------------------------------------------------------------------------
# Buffer config records
# ---------------------------------------------------------------------------

@dataclass
class _WriteBufferConfig:
    worker_name: str
    buffer_id: str
    spec: Any  # SharedRingBufferSpec
    output: Any  # BufferOutput


@dataclass
class _ReadBufferConfig:
    buffer_id: str
    spec: Any  # SharedRingBufferSpec
    upstream_output: Any  # BufferOutput


# ---------------------------------------------------------------------------
# SubprocessService
# ---------------------------------------------------------------------------

class SubprocessService:
    """
    Main-process handle to any JSONL subprocess.

    Owns the transport lifecycle, pumps decoded inbound messages onto the
    shared EventBus, and exposes typed send / request helpers.

    Buffer setup
    ------------
    Call set_write_buffer() and add_read_buffer() in the subclass __init__ to
    register shared-memory buffers. SubprocessService handles ConfigureBuffer
    messages, ItemWritten events, ItemAck routing, and initial slot grants —
    WorkerHandle is a pure command proxy and knows nothing about buffers.

    Usage::

        self.set_write_buffer(
            worker_name="spectrometer",
            buffer_id="spectrometer",
            buffer=buffer,          # must have .spec
            output=buffer_output,
        )
        self.add_read_buffer(
            buffer_id="spectrometer",
            buffer=spec_buf,
            upstream_output=spec_output,
        )

    Use make_send_grant(worker_name, buffer_id) to get the send_grant callable
    needed by BufferOutput before calling set_write_buffer.

    Routing of inbound messages is the EventBus's job: subscribe to message
    types (optionally scoped by source) rather than registering callbacks here.
    """

    service_name: ClassVar[str] = ""

    def __init__(
        self,
        io: TaskRunner,
        endpoint: JsonlSubprocessEndpoint,
        bus: EventBus,
    ) -> None:
        self._io = io
        self._endpoint = endpoint
        self._bus = bus
        self._internal_bus = EventBus()
        self._handle: Optional[StreamHandle] = None
        self._lock = threading.RLock()
        self._handles: dict[str, WorkerHandle] = {}
        self._worker_error_sub: Optional[Callable] = None

        # Buffer configs — populated by set_write_buffer / add_read_buffer
        self._write_buffer: Optional[_WriteBufferConfig] = None
        self._read_buffers: dict[str, _ReadBufferConfig] = {}

        # Bus subscriptions for buffer events (set in start, cleared in stop)
        self._item_written_sub: Optional[Callable] = None
        self._item_ack_sub: Optional[Callable] = None

    @property
    def internal_bus(self) -> EventBus:
        return self._internal_bus

    # ---------- buffer registration ----------

    def make_send_grant(self, worker_name: str, buffer_id: str) -> Callable[[dict], None]:
        """
        Return a Callable[[dict], None] that sends a SlotGranted to the named
        worker. Pass this as send_grant= when constructing BufferOutput, then
        call set_write_buffer() with the finished output object.
        """
        def _send(grant: dict) -> None:
            self.send(SlotGranted(
                slot=grant["slot"],
                item_id=grant["item_id"],
                buffer_id=buffer_id,
                consumer_id=worker_name,
            ))
        return _send

    def set_write_buffer(
        self,
        worker_name: str,
        buffer_id: str,
        buffer: Any,
        output: Any,
    ) -> None:
        """Register the subprocess write buffer. Must be called before start()."""
        self._write_buffer = _WriteBufferConfig(
            worker_name=worker_name,
            buffer_id=buffer_id,
            spec=buffer.spec,
            output=output,
        )

    def add_read_buffer(
        self,
        buffer_id: str,
        buffer: Any,
        upstream_output: Any,
    ) -> None:
        """Register a read buffer from an upstream service. Must be called before start()."""
        self._read_buffers[buffer_id] = _ReadBufferConfig(
            buffer_id=buffer_id,
            spec=buffer.spec,
            upstream_output=upstream_output,
        )

    # ---------- status ----------

    def _publish_status(self, running: bool, detail: str = "") -> None:
        if self.service_name:
            self._bus.publish(ServiceStatus(self.service_name, running, detail))

    # ---------- lifecycle ----------

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._handle is not None

    def start(self) -> None:
        with self._lock:
            if self._handle is not None:
                return
            self._endpoint.start()
            self._handle = self._io.stream(
                self._endpoint.produce,
                on_item=self._internal_bus.publish,
                on_error=self._on_stream_error,
                on_complete=self._on_stream_complete,
                key="subprocess.stream",
                cancel_previous=True,
                drop_outdated=True,
                coalesce=False,
            )
            self._worker_error_sub = self._internal_bus.subscribe(
                WorkerError, self._on_worker_error
            )
            if self._write_buffer is not None:
                self._item_written_sub = self._internal_bus.subscribe(
                    ItemWritten, self._on_item_written
                )
            if self._read_buffers:
                self._item_ack_sub = self._internal_bus.subscribe(
                    ItemAck, self._on_item_ack
                )
        self._configure_buffers()

    def stop(self) -> None:
        with self._lock:
            handle = self._handle
            self._handle = None
            self._endpoint.stop()
            if handle is not None:
                handle.stop()
            for attr in ("_worker_error_sub", "_item_written_sub", "_item_ack_sub"):
                unsub = getattr(self, attr, None)
                if unsub is not None:
                    unsub()
                    setattr(self, attr, None)

    # ---------- buffer configuration ----------

    def _configure_buffers(self) -> None:
        """
        Send ConfigureBuffer for every registered write and read buffer.
        Called from start() after the subprocess stream is running.
        """
        if self._write_buffer is not None:
            cfg = self._write_buffer
            reply = self.request_typed(ConfigureBuffer(
                spec=cfg.spec, buffer_id=cfg.buffer_id, buffer_type="write"
            ))
            if isinstance(reply, ErrorMessage):
                raise RuntimeError(
                    f"ConfigureBuffer(write, {cfg.buffer_id!r}) failed: {reply.error}"
                )
        for cfg in self._read_buffers.values():
            reply = self.request_typed(ConfigureBuffer(
                spec=cfg.spec, buffer_id=cfg.buffer_id, buffer_type="read"
            ))
            if isinstance(reply, ErrorMessage):
                raise RuntimeError(
                    f"ConfigureBuffer(read, {cfg.buffer_id!r}) failed: {reply.error}"
                )

    def _send_initial_grants(self) -> None:
        """
        Reset the write-buffer coordinator and send initial SlotGranted messages.
        Pass as post_start= in WorkerHandle.start_async() for producer workers.
        """
        if self._write_buffer is None:
            return
        cfg = self._write_buffer
        cfg.output.coordinator.reset()
        for grant in cfg.output.coordinator.grant_initial_slots():
            cfg.output.send_grant(grant)

    # ---------- buffer event handlers (internal bus) ----------

    def _on_item_written(self, msg: ItemWritten) -> None:
        cfg = self._write_buffer
        if cfg is None or msg.buffer_id != cfg.buffer_id:
            return
        cfg.output.coordinator.on_item_written(slot=msg.slot, item_id=msg.item_id)
        cfg.output.notify_available(msg.slot, msg.item_id, msg.timestamp_ns)

    def _on_item_ack(self, msg: ItemAck) -> None:
        cfg = self._read_buffers.get(msg.buffer_id)
        if cfg is None:
            return
        cfg.upstream_output.ack_slot(msg.slot, msg.item_id, msg.consumer_id)

    # ---------- control API ----------

    def send(self, message: Message) -> None:
        """Fire-and-forget command/event to the subprocess. Silent no-op if not running."""
        with self._lock:
            if self._handle is None or not self._endpoint.is_running():
                return
            self._endpoint.send(message)

    def request_sync(self, message: Message, *, timeout_s: float = 2.0) -> dict:
        with self._lock:
            self._ensure_running()
        return self._endpoint.raw_request(message, timeout_s=timeout_s)

    def request_typed(self, message: Message, *, timeout_s: float = 2.0) -> Optional[Message]:
        with self._lock:
            self._ensure_running()
        return self._endpoint.request(message, timeout_s=timeout_s)

    def request_async(
        self,
        message: Message,
        *,
        timeout_s: float = 2.0,
        key: str = "subprocess.control.request",
        cancel_previous: bool = False,
        drop_outdated: bool = True,
        on_success: Optional[Callable[[Optional[Message]], None]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> Future:
        with self._lock:
            self._ensure_running()
        return self._io.run(
            lambda: self._endpoint.request(message, timeout_s=timeout_s),
            on_success=on_success,
            on_error=on_error,
            key=key,
            cancel_previous=cancel_previous,
            drop_outdated=drop_outdated,
        )

    def run_async(
        self,
        fn: Callable[[], Any],
        *,
        key: str = "subprocess.control.run",
        cancel_previous: bool = False,
        drop_outdated: bool = True,
        on_success: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> Future:
        with self._lock:
            self._ensure_running()
        return self._io.run(
            fn,
            on_success=on_success,
            on_error=on_error,
            key=key,
            cancel_previous=cancel_previous,
            drop_outdated=drop_outdated,
        )

    # ---------- worker routing ----------

    def worker(self, name: str) -> WorkerHandle:
        if name not in self._handles:
            self._handles[name] = WorkerHandle(service=self, name=name)
        return self._handles[name]

    def _register_handle(self, name: str, handle: WorkerHandle) -> None:
        self._handles[name] = handle

    # ---------- stream callbacks ----------

    def _on_worker_error(self, event: WorkerError) -> None:
        detail = f"worker crashed ({event.worker_name}): {event.error}"
        if self.service_name:
            self._bus.publish(AppMessage(f"{self.service_name}: {detail}", MessageLevel.ERROR))
            self._bus.publish(ServiceStatus(self.service_name, False, detail))

    def _on_stream_error(self, e: BaseException) -> None:
        self.stop()

    def _on_stream_complete(self) -> None:
        self.stop()

    # ---------- helpers ----------

    def _ensure_running(self) -> None:
        if self._handle is None or not self._endpoint.is_running():
            raise RuntimeError("SubprocessService is not running. Call start() first.")
