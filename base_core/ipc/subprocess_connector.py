from __future__ import annotations

import logging
import threading
from multiprocessing.connection import Connection

from base_core.framework.events.event_bus import EventBus
from base_core.ipc.codec import decode, encode
from base_core.ipc.message import Message

log = logging.getLogger(__name__)

_POLL_TIMEOUT = 0.05


class SubprocessPipelineConnector:
    """
    Subprocess end of a multiprocessing.Pipe() connection.

    run() is a blocking loop; call it from the subprocess entry point after setup.
    Workers subscribe to the EventBus to receive inbound messages.

    Usage pattern in a worker:
        def _on_set_config(self, msg: SetStabilizationConfig) -> None:
            self._apply(msg.config)
            self._connector.send(OKReply(request_id=msg.id))
    """

    def __init__(self, conn: Connection, bus: EventBus) -> None:
        self._conn = conn
        self._bus = bus
        self._write_lock = threading.Lock()

    def run(self, stop: threading.Event) -> None:
        """Block until stop is set or the connection closes. Each inbound message is published on bus."""
        while not stop.is_set():
            try:
                if self._conn.poll(_POLL_TIMEOUT):
                    data = self._conn.recv_bytes()
                    try:
                        msg = decode(data)
                        self._bus.publish(msg)
                    except Exception:
                        log.exception("SubprocessPipelineConnector: decode/dispatch error")
            except EOFError:
                log.debug("SubprocessPipelineConnector: connection closed by main process")
                return
            except OSError:
                log.exception("SubprocessPipelineConnector: connection error")
                return

    def send(self, msg: Message) -> None:
        """Send a message to the main process (event or reply). Thread-safe."""
        with self._write_lock:
            self._conn.send_bytes(encode(msg))
