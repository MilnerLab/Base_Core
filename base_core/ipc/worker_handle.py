from __future__ import annotations

import logging
import threading
from typing import Callable, TypeVar

from base_core.framework.events.event_bus import EventBus
from base_core.ipc.message import ErrorReply, Message, OKReply, Reply, Request
from base_core.ipc.service_connector import ServicePipelineConnector
from base_core.ipc.worker_messages import ResetWorker, StartWorker, StopWorker

log = logging.getLogger(__name__)

TReply = TypeVar("TReply", bound=Reply)
T = TypeVar("T")


class BaseWorkerHandle:
    """
    Main-process counterpart to BaseWorker.

    Symmetric with the subprocess side: holds the connector directly and calls
    connector.send() / connector.request() for outgoing traffic. Spontaneous IPC
    messages from the subprocess are published to service_bus by the connector and
    received via _subscribe_service().

    The service owns the handles and injects the connector + service_bus at start
    time via _bind() / _unbind(). Handles never hold a reference back to the service.

    Lifecycle hooks (called by SubprocessService):
      _bind(connector, service_bus)  — before _on_pre_attach
      _on_pre_attach()               — create shared memory (no send needed)
      [AttachBuffer sent]
      _on_attached()                 — subscribe events, send initial messages
      _unbind()                      — calls _on_detached(), clears connector

    Concrete handles:
    - Override _on_attached() and call _subscribe_service(EventType, handler) to
      receive spontaneous subprocess events (e.g. CorrectionAvailable).
    - Implement typed wrapper methods that call _emit() or _request().
    """

    def __init__(self, worker_id: str, bus: EventBus) -> None:
        self._worker_id = worker_id
        self._bus = bus
        self._connector: ServicePipelineConnector | None = None
        self._service_bus: EventBus | None = None
        self._unsubs: list[Callable[[], None]] = []
        self._pending_count: int = 0
        self._pending_lock = threading.Lock()

    @property
    def busy(self) -> bool:
        """True while any request sent via _request() is awaiting a reply."""
        with self._pending_lock:
            return self._pending_count > 0

    # --- lifecycle commands (Request[OKReply]) --------------------------

    def start(self) -> None:
        self._request(StartWorker(worker_id=self._worker_id), self._on_start_reply)

    def stop(self) -> None:
        self._request(StopWorker(worker_id=self._worker_id), self._on_stop_reply)

    def reset(self) -> None:
        self._request(ResetWorker(worker_id=self._worker_id), self._on_reset_reply)

    def _on_start_reply(self, reply: OKReply) -> None:
        pass

    def _on_stop_reply(self, reply: OKReply) -> None:
        pass

    def _on_reset_reply(self, reply: OKReply) -> None:
        pass

    # --- helpers for concrete subclasses --------------------------------

    def _emit(self, msg: Message) -> None:
        """Fire-and-forget message to the subprocess."""
        self._connector.send(msg)  # type: ignore[union-attr]

    def _request(
        self,
        msg: Request[TReply],
        on_reply: Callable[[TReply], None],
        on_error: Callable[[ErrorReply], None] | None = None,
    ) -> None:
        """Send a Request. on_reply fires on success; on_error fires on ErrorReply."""
        with self._pending_lock:
            self._pending_count += 1
        effective_error = on_error or self._on_error

        def _on_done(reply: TReply) -> None:
            with self._pending_lock:
                self._pending_count -= 1
            on_reply(reply)

        def _on_err(err: ErrorReply) -> None:
            with self._pending_lock:
                self._pending_count -= 1
            effective_error(err)

        self._connector.request(msg, _on_done, _on_err)  # type: ignore[union-attr]

    def _subscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        """Subscribe to domain events on the global bus. Cleaned up in _on_detached."""
        self._unsubs.append(self._bus.subscribe(event_type, handler))

    def _subscribe_service(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        """Subscribe to spontaneous IPC events from the subprocess on the service bus."""
        self._unsubs.append(self._service_bus.subscribe(event_type, handler))  # type: ignore[union-attr]

    def _on_error(self, reply: ErrorReply) -> None:
        """Default error handler. Override for custom behaviour."""
        log.error("%s[%s]: error reply: %s", type(self).__name__, self._worker_id, reply.error)

    # --- called by SubprocessService -----------------------------------

    def _bind(self, connector: ServicePipelineConnector, service_bus: EventBus) -> None:
        """Inject connector and service bus. Called by service before _on_pre_attach."""
        self._connector = connector
        self._service_bus = service_bus

    def _unbind(self) -> None:
        """Tear down subscriptions and release the connector. Called by service on stop."""
        self._on_detached()
        self._connector = None

    def _on_pre_attach(self) -> None:
        """Called after _bind(), before AttachBuffer is sent. Create shared memory here."""
        pass

    def _on_attached(self) -> None:
        """Called after AttachBuffer is sent. Subscribe events and send initial messages."""
        pass

    def _on_detached(self) -> None:
        """Clean up all bus subscriptions. Called by _unbind()."""
        for unsub in self._unsubs:
            unsub()
        self._unsubs.clear()
