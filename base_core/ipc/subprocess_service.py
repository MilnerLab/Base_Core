from __future__ import annotations

import logging
import socket as _socket
import subprocess
import sys
from abc import ABC, abstractmethod
from multiprocessing.connection import Connection as _MpConnection
from typing import TYPE_CHECKING

from base_core.framework.events.event_bus import EventBus
from base_core.framework.shm.writer_worker_handle import WriterWorkerHandle
from base_core.ipc.service_connector import ServicePipelineConnector

if TYPE_CHECKING:
    from base_core.framework.shm.buffer import SharedMemoryBuffer
    from base_core.framework.shm.spec import MemorySpec
    from base_core.ipc.worker_handle import BaseWorkerHandle

log = logging.getLogger(__name__)

_JOIN_TIMEOUT = 5.0


class SubprocessService(ABC):
    """
    Abstract base for services that run domain logic in a child process.

    The child is launched via subprocess.Popen so it can use a different Python
    executable (e.g. a 32-bit interpreter for hardware drivers).

    Symmetric with the subprocess side:
    - Outgoing: call connector.send(msg) or connector.request(msg, on_reply) directly.
    - Incoming: the connector dispatches non-reply IPC messages to _service_bus;
      handles subscribe there via _subscribe_service().

    The global bus (self._bus) is used only for domain events.

    Buffer attachments: call add_buffer(BufferClass, spec) before start(). On start(),
    AttachBuffer requests are sent to the subprocess so it attaches shared memory before
    workers receive any slot grants.

    Lifecycle: start() → [connector.send/request] → stop()
    Re-starting after stop() is supported.
    """

    def __init__(
        self,
        bus: EventBus,
        python_exe: str | None = None,
    ) -> None:
        self._bus = bus
        self._python_exe = python_exe or sys.executable
        self._service_bus = EventBus()
        self._connector: ServicePipelineConnector | None = None
        self._process: subprocess.Popen | None = None  # type: ignore[type-arg]
        self._attach_buffers: list[tuple[type[SharedMemoryBuffer], MemorySpec]] = []
        self._worker_handles: list[BaseWorkerHandle] = []

    @property
    def connector(self) -> ServicePipelineConnector | None:
        """The active pipe connector, or None before start() / after stop()."""
        return self._connector

    @property
    def service_bus(self) -> EventBus:
        """Per-service bus for incoming IPC events from the subprocess."""
        return self._service_bus

    def add_handle(self, handle: BaseWorkerHandle) -> None:
        """Register a worker handle. subscribe()/cleanup are called on start/stop."""
        self._worker_handles.append(handle)

    def add_buffer(
        self,
        buffer_cls: type[SharedMemoryBuffer],
        spec: MemorySpec,
    ) -> None:
        """
        Register a shared memory buffer to attach in the subprocess on next start().
        The subprocess must call register_buffer_class(buffer_cls) in its setup().
        """
        self._attach_buffers.append((buffer_cls, spec))

    @property
    @abstractmethod
    def _entry_module(self) -> str:
        """
        Dotted module path executed as `python -m <entry_module> <fd>`.
        Example: "app_apps.analysis.phase_control.subprocess.phase_control_process"
        """
        ...

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        """Create the subprocess and begin the main-process read loop."""
        if self.is_running:
            log.warning("%s.start() called while already running", type(self).__name__)
            return

        srv = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        srv.bind(('127.0.0.1', 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        srv.settimeout(10.0)

        self._process = subprocess.Popen(
            [self._python_exe, "-m", self._entry_module, str(port)],
        )

        try:
            client_sock, _ = srv.accept()
        finally:
            srv.close()

        handles = list(self._worker_handles)

        def _on_disconnect() -> None:
            for handle in handles:
                handle._on_disconnect()

        self._connector = ServicePipelineConnector(
            _MpConnection(client_sock.detach()),
            service_bus=self._service_bus,
            on_disconnect=_on_disconnect,
        )
        self._connector.start()

        for handle in self._worker_handles:
            handle._bind(self._connector, self._service_bus)

        # AttachBuffer for every registered buffer (FIFO ordering with slot grants guaranteed)
        if self._attach_buffers:
            from base_core.framework.shm.messages import AttachBuffer
            for cls, spec in self._attach_buffers:
                self._connector.request(
                    AttachBuffer(buffer_class_name=cls.__name__, spec=spec),
                    on_reply=lambda r: None,
                    on_error=lambda r, name=cls.__name__: log.error(
                        "AttachBuffer failed for %r: %s", name, r.error
                    ),
                )

        for handle in self._worker_handles:
            if isinstance(handle, WriterWorkerHandle):
                handle._on_attached()

    def stop(self) -> None:
        """Stop the read loop, terminate and join the subprocess."""
        for handle in self._worker_handles:
            handle._unbind()

        if self._connector is not None:
            self._connector.stop()
            self._connector = None

        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=_JOIN_TIMEOUT)
                except subprocess.TimeoutExpired:
                    log.warning(
                        "%s: subprocess did not exit after SIGTERM, killing",
                        type(self).__name__,
                    )
                    self._process.kill()
                    self._process.wait()
            self._process = None
