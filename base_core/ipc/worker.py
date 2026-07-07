from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable

from base_core.framework.events.event_bus import EventBus
from base_core.ipc.message import ErrorReply, Message, OKReply, Reply
from base_core.ipc.subprocess_connector import SubprocessPipelineConnector
from base_core.ipc.worker_messages import PauseWorker, ResumeWorker, StartWorker, StopWorker

log = logging.getLogger(__name__)


class BaseWorker(ABC):
    """
    Abstract base for workers that run inside a subprocess.

    Lifecycle:
        BaseSubprocessMain.register_worker(worker)  ← calls activate()
            activate() subscribes StartWorker/PauseWorker/ResumeWorker/StopWorker/ShutdownWorker + calls _setup()
        [receive StartWorker]    → _start()    → _reply_ok()
        [receive PauseWorker]    → _pause()    → _reply_ok()
        [receive ResumeWorker]   → _resume()   → _reply_ok()
        [receive StopWorker]     → _stop()     → _reply_ok()
        [receive ShutdownWorker] → _shutdown() → _reply_ok()  (then subprocess exits)
        [subprocess SIGTERM]     → _shutdown() called by BaseSubprocessMain._teardown()

    Concrete workers:
    - Subscribe domain messages in _setup() via self._bus.subscribe(), appending
      the returned unsub callable to self._unsubs for clean teardown.
    - Override _shutdown() to close hardware connections / flush buffers on exit.
    - Use _reply(reply) / _reply_ok(msg) / _reply_error(msg, err) to respond to requests.
    - Use _notify(msg) to send spontaneous events to the main process.
    """

    def __init__(
        self,
        worker_id: str,
        bus: EventBus,
        connector: SubprocessPipelineConnector,
    ) -> None:
        self._worker_id = worker_id
        self._bus = bus
        self._connector = connector
        self._unsubs: list[Callable[[], None]] = []

    @property
    def worker_id(self) -> str:
        return self._worker_id

    def activate(self) -> None:
        """Wire subscriptions. Called once by BaseSubprocessMain.register_worker()."""
        self._unsubs.append(self._bus.subscribe(StartWorker, self._on_start_cmd))
        self._unsubs.append(self._bus.subscribe(PauseWorker, self._on_pause_cmd))
        self._unsubs.append(self._bus.subscribe(ResumeWorker, self._on_resume_cmd))
        self._unsubs.append(self._bus.subscribe(StopWorker, self._on_stop_cmd))
        self._setup()

    def deactivate(self) -> None:
        """Unsubscribe all handlers."""
        for unsub in self._unsubs:
            unsub()
        self._unsubs.clear()

    @abstractmethod
    def _setup(self) -> None:
        """Subscribe domain-specific messages on self._bus.
        Append each returned unsub callable to self._unsubs."""
        ...

    @abstractmethod
    def _start(self) -> None: ...

    @abstractmethod
    def _pause(self) -> None: ...

    @abstractmethod
    def _resume(self) -> None: ...

    @abstractmethod
    def _stop(self) -> None: ...

    def _shutdown(self) -> None:
        """Close hardware connections and release resources.
        Called by BaseSubprocessMain._teardown() when the subprocess exits.
        Override in concrete workers; default is a no-op."""
        self.deactivate()
        pass

    # --- reply / notify helpers ----------------------------------------

    def _reply(self, reply: Reply) -> None:
        """Send any Reply back to the main process."""
        self._connector.send(reply)

    def _reply_ok(self, request: Message) -> None:
        self._reply(OKReply(request_id=request.id))

    def _reply_error(self, request: Message, error: str) -> None:
        self._reply(ErrorReply(request_id=request.id, error=error))

    def _notify(self, msg: Message) -> None:
        """Send a spontaneous event (not a reply) to the main process."""
        self._connector.send(msg)

    # --- internal lifecycle handlers -----------------------------------

    def _on_start_cmd(self, msg: StartWorker) -> None:
        if msg.worker_id == self._worker_id:
            self._start()
            self._reply_ok(msg)

    def _on_pause_cmd(self, msg: PauseWorker) -> None:
        if msg.worker_id == self._worker_id:
            self._pause()
            self._reply_ok(msg)

    def _on_resume_cmd(self, msg: ResumeWorker) -> None:
        if msg.worker_id == self._worker_id:
            self._resume()
            self._reply_ok(msg)

    def _on_stop_cmd(self, msg: StopWorker) -> None:
        if msg.worker_id == self._worker_id:
            self._stop()
            self._reply_ok(msg)
