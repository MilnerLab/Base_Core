from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, TypeVar

from base_core.framework.events.event_bus import EventBus
from base_core.ipc.message import ErrorReply, Message, OKReply, Reply, Request
from base_core.ipc.worker_messages import ResetWorker, StartWorker, StopWorker

if TYPE_CHECKING:
    from base_core.ipc.subprocess_service import SubprocessService

log = logging.getLogger(__name__)

TReply = TypeVar("TReply", bound=Reply)
T = TypeVar("T")


class BaseWorkerHandle:
    """
    Main-process counterpart to BaseWorker.

    Wraps pipeline communication: fire-and-forget via _emit(), request/reply via
    _request(). The optional on_error parameter fires when the subprocess sends
    ErrorReply; defaults to logging the error.

    Lifecycle hooks _on_attached / _on_detached are called by SubprocessService
    on start() / stop() so bus subscriptions are properly scoped to service lifetime.

    Concrete handles:
    - Override _on_attached() and call _subscribe(EventType, handler) to receive
      spontaneous events from the subprocess (e.g. CorrectionAvailable).
    - Implement typed wrapper methods that call _emit() or _request().
    - Define reply handlers as methods (named _on_<something>_reply) for clarity.
    """

    def __init__(
        self,
        worker_id: str,
        service: SubprocessService,
        bus: EventBus,
    ) -> None:
        self._worker_id = worker_id
        self._service = service
        self._bus = bus
        self._unsubs: list[Callable[[], None]] = []

    # --- lifecycle (Request[OKReply]) ----------------------------------

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

    # --- helpers for concrete subclasses -------------------------------

    def _emit(self, msg: Message) -> None:
        """Fire-and-forget message to the subprocess."""
        self._service.emit(msg)

    def _request(
        self,
        msg: Request[TReply],
        on_reply: Callable[[TReply], None],
        on_error: Callable[[ErrorReply], None] | None = None,
    ) -> None:
        """Send a Request. on_reply fires on success; on_error fires on ErrorReply."""
        self._service.send(msg, on_reply, on_error or self._on_error)

    def _subscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        """Subscribe to spontaneous events on the main bus. Cleaned up in _on_detached."""
        self._unsubs.append(self._bus.subscribe(event_type, handler))

    def _on_error(self, reply: ErrorReply) -> None:
        """Default error handler. Override for custom behaviour."""
        log.error("%s[%s]: error reply: %s", type(self).__name__, self._worker_id, reply.error)

    # --- lifecycle hooks called by SubprocessService -------------------

    def _on_attached(self) -> None:
        """Called by service.start(). Subscribe main-bus events here via _subscribe()."""
        pass

    def _on_detached(self) -> None:
        """Called by service.stop(). Cleans up all bus subscriptions."""
        for unsub in self._unsubs:
            unsub()
        self._unsubs.clear()
