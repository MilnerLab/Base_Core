from __future__ import annotations

import threading
from concurrent.futures import Future
from typing import Any, Callable, Dict, Mapping, Optional

from base_core.framework.concurrency.interfaces import ITaskRunner
from base_core.framework.concurrency.models import StreamHandle
from base_core.framework.json.json_endpoint import JsonlSubprocessEndpoint


EventHandler = Callable[[dict], None]


class DeviceService:
    """
    Main-process wrapper around a subprocess endpoint.

    Responsibilities:
      - own the JSON transport lifecycle
      - dispatch incoming protocol events to local handlers
      - expose command / request helpers

    This class intentionally does NOT know how large data is stored.
    Event payloads should only carry handles / metadata.
    """

    def __init__(self, io: ITaskRunner, endpoint: JsonlSubprocessEndpoint) -> None:
        self._io = io
        self._endpoint = endpoint
        self._handle: Optional[StreamHandle] = None
        self._lock = threading.RLock()

        # event routing by event name
        self._event_handlers: Dict[str, EventHandler] = {}

    # ---------- lifecycle ----------

    def start(self) -> None:
        with self._lock:
            if self._handle is not None:
                return

            self._endpoint.start()

            def producer(stop: threading.Event):
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
            self._endpoint.stop()
            if handle is not None:
                handle.stop()

    # ---------- event routing ----------

    def register_event_handler(self, event_name: str, handler: EventHandler) -> None:
        """
        Register a handler for messages where:
          msg.get("kind") == "event" and msg.get("name") == event_name

        NOTE: Handlers are called in the stream worker thread.
        Keep them fast (store data + signal/UI dispatch), do not block.
        """
        with self._lock:
            self._event_handlers[event_name] = handler

    def unregister_event_handler(self, event_name: str) -> None:
        with self._lock:
            self._event_handlers.pop(event_name, None)

    def clear_event_handlers(self) -> None:
        with self._lock:
            self._event_handlers.clear()

    # Backward-compatible convenience names
    def register_handler(self, msg_type: str, handler: EventHandler) -> None:
        self.register_event_handler(msg_type, handler)

    def unregister_handler(self, msg_type: str) -> None:
        self.unregister_event_handler(msg_type)

    def clear_handlers(self) -> None:
        self.clear_event_handlers()

    # ---------- Control API ----------

    def send(self, msg: dict) -> None:
        with self._lock:
            self._ensure_running()
            self._endpoint.send(msg)

    def send_command(
        self,
        name: str,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        target: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._ensure_running()
            self._endpoint.send_command(name, payload, target=target)

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

    def request_command_async(
        self,
        name: str,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        target: Optional[str] = None,
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
            lambda: self._endpoint.request_command(
                name,
                payload,
                target=target,
                timeout_s=timeout_s,
            ),
            on_success=on_success,
            on_error=on_error,
            key=key,
            cancel_previous=cancel_previous,
            drop_outdated=drop_outdated,
        )

    # ---------- Stream callbacks ----------

    def _on_stream_message(self, msg: dict) -> None:
        kind = msg.get("kind")
        name = msg.get("name")
        if kind != "event" or not isinstance(name, str):
            return

        with self._lock:
            handler = self._event_handlers.get(name)

        if handler is not None:
            handler(msg)

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
