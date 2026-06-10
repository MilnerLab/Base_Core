from __future__ import annotations

import threading
from concurrent.futures import Future
from typing import Any, Callable, ClassVar, Optional

from base_core.framework.app.service_status import ServiceStatus
from base_core.framework.concurrency.models import StreamHandle
from base_core.framework.concurrency.task_runner import TaskRunner
from base_core.framework.events.event_bus import EventBus
from base_core.framework.subprocess.json_endpoint import JsonlSubprocessEndpoint
from base_core.framework.subprocess.messages import Message
from base_core.framework.subprocess.worker_handle import WorkerHandle


class SubprocessService:
    """
    Main-process handle to any JSONL subprocess (the general base).

    Owns the transport lifecycle, pumps decoded inbound messages onto the
    shared EventBus, and exposes typed send / request helpers.

    Routing of inbound messages is the EventBus's job: subscribe to message
    types (optionally scoped by source) rather than registering callbacks here.

    For hardware device subprocesses use DeviceService(SubprocessService).
    For analysis / calculation subprocesses use this directly.
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
        self._handle: Optional[StreamHandle] = None
        self._lock = threading.RLock()
        self._handles: dict[str, WorkerHandle] = {}

    # ---------- status ----------

    def _publish_status(self, running: bool, detail: str = "") -> None:
        """Publish a ServiceStatus event if service_name is declared."""
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
                on_item=self._bus.publish,
                on_error=self._on_stream_error,
                on_complete=self._on_stream_complete,
                key="subprocess.stream",
                cancel_previous=True,
                drop_outdated=True,
                coalesce=False,
            )

    def stop(self) -> None:
        with self._lock:
            handle = self._handle
            self._handle = None
            self._endpoint.stop()
            if handle is not None:
                handle.stop()

    # ---------- control API ----------

    def send(self, message: Message) -> None:
        """Fire-and-forget command/event to the subprocess. Silent no-op if not running."""
        with self._lock:
            if self._handle is None or not self._endpoint.is_running():
                return
            self._endpoint.send(message)

    def request_sync(self, message: Message, *, timeout_s: float = 2.0) -> dict:
        """
        Synchronous blocking request/reply, returning the raw envelope dict.
        For startup/setup sequences that need confirmation before proceeding.
        Must NOT be called from the EventBus / TaskRunner stream thread.
        """
        with self._lock:
            self._ensure_running()
        return self._endpoint.raw_request(message, timeout_s=timeout_s)

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
        """Send a command and resolve with the decoded reply Message (or None)."""
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
        """Submit an arbitrary callable to the TaskRunner pool thread."""
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
        """Return the registered handle for a named worker, creating a plain WorkerHandle if none was registered."""
        if name not in self._handles:
            self._handles[name] = WorkerHandle(service=self, name=name, bus=self._bus)
        return self._handles[name]

    def _register_handle(self, name: str, handle: WorkerHandle) -> None:
        """Register a pre-built handle for worker(name) to return."""
        self._handles[name] = handle

    # ---------- stream callbacks ----------

    def _on_stream_error(self, e: BaseException) -> None:
        self.stop()

    def _on_stream_complete(self) -> None:
        self.stop()

    # ---------- helpers ----------

    def _ensure_running(self) -> None:
        if self._handle is None or not self._endpoint.is_running():
            raise RuntimeError("SubprocessService is not running. Call start() first.")
