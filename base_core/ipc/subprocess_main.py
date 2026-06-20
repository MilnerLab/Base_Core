from __future__ import annotations

import logging
import signal
import sys
import threading
from abc import ABC, abstractmethod
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, TypeVar

from base_core.framework.events.event_bus import EventBus
from base_core.ipc.message import ErrorReply, OKReply
from base_core.ipc.subprocess_connector import SubprocessPipelineConnector

if TYPE_CHECKING:
    from base_core.framework.shm.buffer import SharedMemoryBuffer
    from base_core.ipc.worker import BaseWorker

log = logging.getLogger(__name__)

TBuffer = TypeVar("TBuffer", bound="SharedMemoryBuffer")


class BaseSubprocessMain(ABC):
    """
    Abstract base for subprocess entry-point classes.

    Buffer attachment: call register_buffer_class(SomeBuffer) in setup(). The base
    run() subscribes to AttachBuffer messages before setup(), so any message that
    arrives during or after setup() will create the corresponding buffer instance
    (via SomeBuffer.attach(spec)) and make it available via get_buffer(SomeBuffer).

    Usage:
        class PhaseControlProcess(BaseSubprocessMain):
            def setup(self) -> None:
                self.register_buffer_class(SpectrumBuffer)
                worker = PhaseTrackingWorker(self.connector, ...)
                self.bus.subscribe(SetStabilizationConfig, worker.on_set_config)

        if __name__ == "__main__":
            PhaseControlProcess.main()
    """

    def __init__(self, conn: Connection) -> None:
        self.bus = EventBus()
        self.connector = SubprocessPipelineConnector(conn, self.bus)
        self._buffer_classes: dict[str, type[SharedMemoryBuffer]] = {}
        self._buffers: dict[str, SharedMemoryBuffer] = {}
        self._workers: dict[str, BaseWorker] = {}

    def register_buffer_class(self, cls: type[TBuffer]) -> None:
        """Declare a buffer class available for attachment. Call in setup()."""
        self._buffer_classes[cls.__name__] = cls  # type: ignore[assignment]

    def register_worker(self, worker: BaseWorker) -> None:
        """Register and activate a worker. Call in setup()."""
        self._workers[worker.worker_id] = worker
        worker.activate()

    def get_buffer(self, cls: type[TBuffer]) -> TBuffer:
        """Return the attached buffer for cls. Raises KeyError if not yet attached."""
        return self._buffers[cls.__name__]  # type: ignore[return-value]

    @abstractmethod
    def setup(self) -> None:
        """Subscribe workers / handlers to self.bus. Called once before the read loop."""
        ...

    def run(self) -> None:
        """Install signal handlers, subscribe AttachBuffer, call setup(), run read loop."""
        stop_event = threading.Event()

        def _handle_signal(signum, frame):
            log.debug("BaseSubprocessMain: received signal %s, stopping", signum)
            stop_event.set()

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

        # Subscribe before setup() so AttachBuffer messages arriving during setup are handled
        from base_core.framework.shm.messages import AttachBuffer
        self.bus.subscribe(AttachBuffer, self._on_attach_buffer)

        try:
            self.setup()
        except Exception:
            log.exception("BaseSubprocessMain.setup() failed")
            return

        self.connector.run(stop_event)
        self._teardown()

    def _teardown(self) -> None:
        """Called after the read loop exits (SIGTERM or pipe close).
        Calls _shutdown() on every registered worker so hardware is released cleanly."""
        for worker in self._workers.values():
            try:
                worker._shutdown()
            except Exception:
                log.exception("Error in %r._shutdown()", worker.worker_id)
        for worker in self._workers.values():
            worker.deactivate()

    def _on_attach_buffer(self, msg: AttachBuffer) -> None:  # type: ignore[name-defined]
        cls = self._buffer_classes.get(msg.buffer_class_name)
        if cls is None:
            log.error("AttachBuffer: class %r not registered — call register_buffer_class() in setup()", msg.buffer_class_name)
            self.connector.send(ErrorReply(request_id=msg.id, error=f"{msg.buffer_class_name!r} not registered"))
            return
        try:
            self._buffers[msg.buffer_class_name] = cls.attach(msg.spec)
            self.connector.send(OKReply(request_id=msg.id))
            log.debug("Attached %s to shared memory %r", msg.buffer_class_name, msg.spec.name)
        except Exception as exc:
            log.exception("AttachBuffer: failed to attach %r", msg.buffer_class_name)
            self.connector.send(ErrorReply(request_id=msg.id, error=str(exc)))

    @classmethod
    def main(cls) -> None:
        """
        Entry point called from `if __name__ == "__main__"`.
        Reads the inherited pipe FD from sys.argv[1], reconstructs the Connection,
        then creates and runs an instance of this class.
        """
        if len(sys.argv) < 2:
            log.error("%s.main(): expected port as first argument", cls.__name__)
            sys.exit(1)

        import socket as _socket

        port = int(sys.argv[1])
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        sock.connect(('127.0.0.1', port))
        conn = Connection(sock.detach())
        cls(conn).run()
