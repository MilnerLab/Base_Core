from __future__ import annotations

import logging
import threading
from multiprocessing.connection import Connection
from typing import Callable, Iterator, TypeVar

from base_core.framework.concurrency.models import StreamHandle
from base_core.framework.concurrency.task_runner import TaskRunner
from base_core.framework.events.event_bus import EventBus
from base_core.ipc.codec import decode, encode
from base_core.ipc.message import ErrorReply, Message, Reply, Request

log = logging.getLogger(__name__)

TReply = TypeVar("TReply", bound=Reply)

_POLL_TIMEOUT = 0.05


class ServicePipelineConnector:
    """
    Main-process end of a multiprocessing.Pipe() connection to a subprocess.

    Threading model:
    - The read loop runs on a TaskRunner thread (io.stream with coalesce=False).
    - send() / emit() may be called from any thread; a write lock serializes them.
    - on_reply callbacks are called on the background reader thread. Consumers that
      need to dispatch to a UI thread must do so themselves (e.g. via QtDispatcher).
    """

    def __init__(self, conn: Connection, bus: EventBus, io: TaskRunner) -> None:
        self._conn = conn
        self._bus = bus
        self._io = io
        self._pending: dict[str, tuple[Callable[[Reply], None], Callable[[ErrorReply], None] | None]] = {}
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._handle: StreamHandle | None = None

    def start(self) -> StreamHandle:
        """Begin the background read loop. Returns a StreamHandle to stop it."""
        self._handle = self._io.stream(
            self._read_loop,
            on_item=self._dispatch,
            coalesce=False,
        )
        return self._handle

    def stop(self) -> None:
        """Stop the read loop and block until the reader thread exits."""
        if self._handle is not None:
            self._handle.stop_event.set()
            self._handle.future.result()
            self._handle = None

    def send(
        self,
        msg: Request[TReply],
        on_reply: Callable[[TReply], None],
        on_error: Callable[[ErrorReply], None] | None = None,
    ) -> None:
        """
        Send a Request to the subprocess and register callbacks for the response.
        Non-blocking. Callbacks are invoked on the background reader thread.
        on_error fires when the subprocess sends ErrorReply; if None, errors are silently dropped.
        """
        with self._pending_lock:
            self._pending[msg.id] = (on_reply, on_error)  # type: ignore[assignment]
        with self._write_lock:
            self._conn.send_bytes(encode(msg))

    def emit(self, msg: Message) -> None:
        """Send a plain message to the subprocess (no reply expected)."""
        with self._write_lock:
            self._conn.send_bytes(encode(msg))

    def _read_loop(self, stop: threading.Event) -> Iterator[Message]:
        while not stop.is_set():
            try:
                if self._conn.poll(_POLL_TIMEOUT):
                    data = self._conn.recv_bytes()
                    try:
                        yield decode(data)
                    except Exception:
                        log.exception("ServicePipelineConnector: decode error")
            except EOFError:
                log.debug("ServicePipelineConnector: connection closed by subprocess")
                return
            except OSError:
                log.exception("ServicePipelineConnector: connection error")
                return

    def _dispatch(self, msg: Message) -> None:
        if isinstance(msg, Reply) and msg.request_id:
            with self._pending_lock:
                entry = self._pending.pop(msg.request_id, None)
            if entry is not None:
                on_reply, on_error = entry
                if isinstance(msg, ErrorReply) and on_error is not None:
                    on_error(msg)
                else:
                    on_reply(msg)  # type: ignore[arg-type]
                return
        self._bus.publish(msg)
