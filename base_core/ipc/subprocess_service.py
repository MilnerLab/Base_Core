from __future__ import annotations

import logging
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from multiprocessing import Pipe
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from base_core.framework.concurrency.task_runner import TaskRunner
from base_core.framework.events.event_bus import EventBus
from base_core.ipc.message import ErrorReply, Message, Reply, Request
from base_core.ipc.service_connector import ServicePipelineConnector

if TYPE_CHECKING:
    from base_core.framework.shm.buffer import SharedMemoryBuffer
    from base_core.framework.shm.spec import MemorySpec
    from base_core.ipc.worker_handle import BaseWorkerHandle

log = logging.getLogger(__name__)

TReply = TypeVar("TReply", bound=Reply)

_JOIN_TIMEOUT = 5.0


class SubprocessService(ABC):
    """
    Abstract base for services that run domain logic in a child process.

    The child is launched via subprocess.Popen so it can use a different Python
    executable (e.g. a 32-bit interpreter for hardware drivers).

    Buffer attachments: call add_buffer(BufferClass, spec) before start(). On start(),
    the service sends AttachBuffer messages to the subprocess for each registered buffer
    so the subprocess can attach to shared memory before its workers begin.

    Subclass responsibilities:
    1. Implement _entry_module — the dotted module path run with `python -m`.
    2. Call add_buffer() for any shared memory buffers the subprocess should read.
    3. Call send() / emit() to communicate with the subprocess after start().

    Lifecycle: start() -> [send/emit] -> stop()
    Re-starting after stop() is supported.
    """

    def __init__(
        self,
        bus: EventBus,
        io: TaskRunner,
        python_exe: str | None = None,
    ) -> None:
        self._bus = bus
        self._io = io
        self._python_exe = python_exe or sys.executable
        self._connector: ServicePipelineConnector | None = None
        self._process: subprocess.Popen | None = None  # type: ignore[type-arg]
        self._attach_buffers: list[tuple[type[SharedMemoryBuffer], MemorySpec]] = []
        self._worker_handles: list[BaseWorkerHandle] = []
        self._translations: dict[type, Callable[[Any], Any]] = {}
        self._translation_unsubs: list[Callable[[], None]] = []

    def add_translation(self, msg_type: type, to_event: Callable[[Any], Any]) -> None:
        """
        Declare a subprocess→bus translation.

        When a Message of msg_type arrives from the subprocess (dispatched onto the main
        bus by the connector), call to_event(msg) and publish the returned event.
        Subscriptions are created on start() and removed on stop().
        """
        self._translations[msg_type] = to_event

    def add_handle(self, handle: BaseWorkerHandle) -> None:
        """Register a worker handle. _on_attached/_on_detached are called on start/stop."""
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

        parent_conn, child_conn = Pipe()
        child_fd = child_conn.fileno()
        os.set_inheritable(child_fd, True)

        self._connector = ServicePipelineConnector(parent_conn, self._bus, self._io)
        self._process = subprocess.Popen(
            [self._python_exe, "-m", self._entry_module, str(child_fd)],
            close_fds=False,
        )
        child_conn.close()
        self._connector.start()

        # Send AttachBuffer for every registered buffer (FIFO — arrives before SlotGrants)
        if self._attach_buffers:
            from base_core.framework.shm.messages import AttachBuffer
            for cls, spec in self._attach_buffers:
                self.send(
                    AttachBuffer(buffer_class_name=cls.__name__, spec=spec),
                    on_reply=lambda r: None,
                    on_error=lambda r, name=cls.__name__: log.error(
                        "AttachBuffer failed for %r: %s", name, r.error
                    ),
                )

        for handle in self._worker_handles:
            handle._on_attached()

        for msg_type, factory in self._translations.items():
            def _handler(msg: Any, f: Callable[[Any], Any] = factory) -> None:
                event = f(msg)
                if event is not None:
                    self._bus.publish(event)
            self._translation_unsubs.append(self._bus.subscribe(msg_type, _handler))

    def stop(self) -> None:
        """Stop the read loop, terminate and join the subprocess."""
        for unsub in self._translation_unsubs:
            unsub()
        self._translation_unsubs.clear()

        for handle in self._worker_handles:
            handle._on_detached()

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

    def send(
        self,
        msg: Request[TReply],
        on_reply: Callable[[TReply], None],
        on_error: Callable[[ErrorReply], None] | None = None,
    ) -> None:
        """Delegate to connector. No-op if not started."""
        if self._connector is not None:
            self._connector.send(msg, on_reply, on_error)

    def emit(self, msg: Message) -> None:
        """Send a plain message (no reply expected). No-op if not started."""
        if self._connector is not None:
            self._connector.emit(msg)
