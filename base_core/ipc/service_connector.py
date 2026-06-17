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

    Symmetric with SubprocessPipelineConnector on the worker side:
    - send(msg)             — fire-and-forget write to pipe
    - request(msg, ...)     — write to pipe + register reply callbacks
    - Incoming bytes are read on a background thread and dispatched:
        reply messages  → matched against pending callbacks
        other messages  → published to service_bus for handle subscribers

    Threading model:
    - The read loop runs on a TaskRunner thread (io.stream with coalesce=False).
    - send()/request() acquire a write lock so concurrent callers don't interleave bytes.
    - Reply callbacks are invoked on the reader thread. Consumers that need to dispatch
      to the UI thread must do so explicitly.
    """

    def __init__(
        self,
        conn: Connection,
        service_bus: EventBus,
        io: TaskRunner,
    ) -> None:
        self._conn = conn
        self._service_bus = service_bus
        self._io = io
        self._pending: dict[str, tuple[Callable[[Reply], None], Callable[[ErrorReply], None] | None]] = {}
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._handle: StreamHandle | None = None

    def start(self) -> StreamHandle:
        """Begin the background read loop."""
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

    def send(self, msg: Message) -> None:
        """Fire-and-forget: encode and write msg to the pipe."""
        with self._write_lock:
            self._conn.send_bytes(encode(msg))

    def request(
        self,
        msg: Request[TReply],
        on_reply: Callable[[TReply], None],
        on_error: Callable[[ErrorReply], None] | None = None,
    ) -> None:
        """Send a Request and register reply callbacks keyed by msg.id."""
        with self._pending_lock:
            self._pending[msg.id] = (on_reply, on_error)  # type: ignore[assignment]
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
        self._service_bus.publish(msg)
