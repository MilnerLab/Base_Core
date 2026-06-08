from __future__ import annotations

import threading
from concurrent.futures import Future
from typing import Callable, Optional

from base_core.framework.concurrency.models import StreamHandle
from base_core.framework.concurrency.task_runner import TaskRunner
from json_endpoint import JsonlSubprocessEndpoint
from messages import Message
from events.event_bus import EventBus


class DeviceService:
    """
    Main-process wrapper around one subprocess endpoint.

    Owns the transport lifecycle, pumps decoded inbound messages onto the
    shared EventBus, and exposes typed send / request helpers. Routing of
    inbound messages is the bus's job: subscribe to message types (optionally
    scoped by source) rather than registering name-keyed callbacks here.

    Knows nothing about how large data is stored; messages carry only handles
    and metadata.
    """

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

    # ---------- lifecycle ----------

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
                key="device.stream",
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
        """Fire-and-forget command/event to the subprocess."""
        with self._lock:
            self._ensure_running()
            self._endpoint.send(message)

    def request_async(
        self,
        message: Message,
        *,
        timeout_s: float = 2.0,
        key: str = "device.control.request",
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

    # ---------- stream callbacks ----------

    def _on_stream_error(self, e: BaseException) -> None:
        self.stop()

    def _on_stream_complete(self) -> None:
        self.stop()

    # ---------- helpers ----------

    def _ensure_running(self) -> None:
        if self._handle is None or not self._endpoint.is_running():
            raise RuntimeError("DeviceService is not running. Call start() first.")