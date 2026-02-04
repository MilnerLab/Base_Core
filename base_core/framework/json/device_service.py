from __future__ import annotations

import threading
from concurrent.futures import Future
from typing import Any, Callable, Dict, Optional

from base_core.framework.concurrency.interfaces import ITaskRunner
from base_core.framework.concurrency.models import StreamHandle
from base_core.framework.json.json_endpoint import JsonlSubprocessEndpoint


MessageHandler = Callable[[dict], None]


class DeviceService:
    def __init__(self, io: ITaskRunner, endpoint: JsonlSubprocessEndpoint) -> None:
        self._io = io
        self._endpoint = endpoint
        self._handle: Optional[StreamHandle] = None
        self._lock = threading.RLock()

        # message routing
        self._handlers: Dict[str, MessageHandler] = {}

    # ---------- lifecycle ----------

    def start(self) -> None:
        with self._lock:
            if self._handle is not None:
                return

            self._endpoint.start()

            def producer(stop: threading.Event):
                # runs in TaskRunner.stream worker
                yield from self._endpoint.produce(stop)

            self._handle = self._io.stream(
                producer,
                on_item=self._on_stream_message,
                on_error=self._on_stream_error,
                on_complete=self._on_stream_complete,
                key="device.stream",
                cancel_previous=True,
                drop_outdated=True,
            )

    def stop(self) -> None:
        with self._lock:
            handle = self._handle
            self._handle = None

            # Best practice for Option A: stop endpoint first to unblock stdout reading.
            self._endpoint.stop()

            if handle is not None:
                handle.stop()

    # ---------- message routing ----------

    def register_handler(self, msg_type: str, handler: MessageHandler) -> None:
        """
        Register a handler for messages where msg.get("type") == msg_type.

        NOTE: Handlers are called in the stream worker thread.
        Keep them fast (store data + signal/UI dispatch), don't block.
        """
        with self._lock:
            self._handlers[msg_type] = handler

    def unregister_handler(self, msg_type: str) -> None:
        with self._lock:
            self._handlers.pop(msg_type, None)

    def clear_handlers(self) -> None:
        with self._lock:
            self._handlers.clear()

    # ---------- Control API ----------

    def send(self, msg: dict) -> None:
        with self._lock:
            self._ensure_running()
            self._endpoint.send(msg)

    def request_async(
        self,
        msg: dict,
        *,
        timeout_s: float = 2.0,
        key: str = "device.control.request",
        cancel_previous: bool = False,
        drop_outdated: bool = True,
        on_success: Optional[Callable[[dict], None]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> Future[dict]:
        with self._lock:
            self._ensure_running()

        return self._io.run(
            lambda: self._endpoint.request(msg, timeout_s=timeout_s),
            on_success=on_success,
            on_error=on_error,
            key=key,
            cancel_previous=cancel_previous,
            drop_outdated=drop_outdated,
        )

    # ---------- Stream callbacks ----------

    def _on_stream_message(self, msg: dict) -> None:
        msg_type = msg.get("type")
        if isinstance(msg_type, str):
            # Copy handler reference under lock, then call without lock
            with self._lock:
                handler = self._handlers.get(msg_type)
            if handler is not None:
                handler(msg)
                return

    def _on_stream_error(self, e: BaseException) -> None:
        self.stop()
        print("Stream error:", e)

    def _on_stream_complete(self) -> None:
        self.stop()
        print("Stream complete")

    # ---------- helpers ----------

    def _ensure_running(self) -> None:
        if self._handle is None or not self._endpoint.is_running():
            raise RuntimeError("DeviceService is not running. Call start() first.")
