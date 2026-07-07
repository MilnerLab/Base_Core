from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Callable, Generic, TypeVar

from base_core.framework.events.event_bus import EventBus
from base_core.ipc.message import ErrorReply, Message, OKReply, Reply, Request
from base_core.ipc.service_connector import ServicePipelineConnector
from base_core.ipc.worker_messages import PauseWorker, ResumeWorker, StartWorker, StopWorker

log = logging.getLogger(__name__)

TReply = TypeVar("TReply", bound=Reply)
T = TypeVar("T")
TEvent = TypeVar("TEvent")


class WorkerStatus(Enum):
    NEW     = "new"      # initial state; also after stop()
    RUNNING = "running"  # after start() or resume() confirmed
    PAUSED  = "paused"   # after pause() confirmed
    BUSY    = "busy"     # IPC request in-flight; restores to previous status on reply


class WorkerState(Generic[TEvent]):
    """
    Tracks the current WorkerStatus and optionally notifies via a typed event.

    Concrete handles pass a no-arg event class as make_event so that VMs can
    subscribe to state-change notifications without polling. The VM then reads
    handle.state directly to get the current WorkerStatus.

    Example:
        WorkerState(bus, SpectrometerWorkerStateChanged)
        # publishes SpectrometerWorkerStateChanged() on every status transition
    """

    def __init__(
        self,
        bus: EventBus,
        make_event: Callable[[], TEvent] | None = None,
    ) -> None:
        self._bus = bus
        self._make_event = make_event
        self._status = WorkerStatus.NEW

    @property
    def value(self) -> WorkerStatus:
        return self._status

    def _set(self, status: WorkerStatus) -> None:
        if status == self._status:
            return
        self._status = status
        if self._make_event is not None:
            self._bus.publish(self._make_event())


class BaseWorkerHandle(Generic[TEvent]):
    """
    Main-process counterpart to BaseWorker.

    Generic over TEvent — the type of the no-arg domain event published on every
    WorkerStatus transition. Pass state_event=YourEventClass to opt in; omit it
    (or leave it None) for handles that don't need state-change notifications.

    Lifecycle hooks (called by SubprocessService):
      _bind(connector, service_bus)  — injects connector, calls subscribe()
      [AttachBuffer sent for buffer-backed handles (WriterWorkerHandle only)]
      _unbind()                      — clears subscriptions and connector

    Concrete handles:
    - Override subscribe() and call _subscribe() / _subscribe_service() to register
      event handlers. All subscriptions are cleared automatically on unbind.
    - Implement typed wrapper methods that call _emit() or _request().
    """

    def __init__(
        self,
        worker_id: str,
        bus: EventBus,
        *,
        state_event: Callable[[], TEvent] | None = None,
    ) -> None:
        self._worker_id = worker_id
        self._bus = bus
        self._connector: ServicePipelineConnector | None = None
        self._service_bus: EventBus | None = None
        self._unsubs: list[Callable[[], None]] = []
        self._pending_count: int = 0
        self._pending_lock = threading.Lock()
        self._worker_state: WorkerState[TEvent] = WorkerState(bus, state_event)
        self._pre_busy_status: WorkerStatus = WorkerStatus.NEW

    @property
    def state(self) -> WorkerStatus:
        return self._worker_state.value

    @property
    def busy(self) -> bool:
        return self._worker_state.value == WorkerStatus.BUSY

    # --- lifecycle commands (Request[OKReply]) --------------------------

    def start(self) -> None:
        self._request(StartWorker(worker_id=self._worker_id), self._on_start_reply)

    def pause(self) -> None:
        self.unsubscribe()
        self._request(PauseWorker(worker_id=self._worker_id), self._on_pause_reply)

    def resume(self) -> None:
        self._request(ResumeWorker(worker_id=self._worker_id), self._on_resume_reply)

    def stop(self) -> None:
        self._request(StopWorker(worker_id=self._worker_id), self._on_stop_reply)

    def _on_start_reply(self, reply: OKReply) -> None:
        self.subscribe()
        self._worker_state._set(WorkerStatus.RUNNING)

    def _on_pause_reply(self, reply: OKReply) -> None:
        self._worker_state._set(WorkerStatus.PAUSED)

    def _on_resume_reply(self, reply: OKReply) -> None:
        self.subscribe()
        self._worker_state._set(WorkerStatus.RUNNING)

    def _on_stop_reply(self, reply: OKReply) -> None:
        self._worker_state._set(WorkerStatus.NEW)

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
        """Send a Request. Status → BUSY while in-flight.

        Lifecycle handlers (_on_start_reply etc.) set the final status themselves.
        Non-lifecycle handlers leave status as BUSY, so the restore fires after them.
        Either way exactly one state-change event is published per transition.
        """
        with self._pending_lock:
            if self._pending_count == 0:
                self._pre_busy_status = self._worker_state.value
                self._worker_state._set(WorkerStatus.BUSY)
            self._pending_count += 1
        effective_error = on_error or self._on_error

        def _on_done(reply: TReply) -> None:
            with self._pending_lock:
                self._pending_count -= 1
                last = self._pending_count == 0
            on_reply(reply)  # lifecycle handlers set status (RUNNING/PAUSED/NEW) here
            if last and self._worker_state.value == WorkerStatus.BUSY:
                # Non-lifecycle reply: handler didn't change status, restore pre-busy
                self._worker_state._set(self._pre_busy_status)

        def _on_err(err: ErrorReply) -> None:
            with self._pending_lock:
                self._pending_count -= 1
                last = self._pending_count == 0
            if last:
                self._worker_state._set(self._pre_busy_status)
            effective_error(err)

        self._connector.request(msg, _on_done, _on_err)  # type: ignore[union-attr]
    
    def subscribe(self) -> None:
        """Override to register event subscriptions via _subscribe() / _subscribe_service().
        All subscriptions are cleared automatically on unbind."""
        pass
    
    def unsubscribe(self) -> None:
        for unsub in self._unsubs:
            unsub()
        self._unsubs.clear()
    
    def _subscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        """Subscribe to domain events on the global bus. Cleared automatically on stop."""
        self._unsubs.append(self._bus.subscribe(event_type, handler))

    def _subscribe_service(self, message_type: type[T], handler: Callable[[T], None]) -> None:
        """Subscribe to spontaneous IPC events from the subprocess. Cleared automatically on stop."""
        self._unsubs.append(self._service_bus.subscribe(message_type, handler))  # type: ignore[union-attr]

    def _on_error(self, reply: ErrorReply) -> None:
        """Default error handler. Override for custom behaviour."""
        log.error("%s[%s]: error reply: %s", type(self).__name__, self._worker_id, reply.error)

    # --- called by SubprocessService -----------------------------------

    def _bind(self, connector: ServicePipelineConnector, service_bus: EventBus) -> None:
        """Inject connector and service bus, then call subscribe().
        WriterWorkerHandle overrides this to defer subscribe() until _on_attached()."""
        self._connector = connector
        self._service_bus = service_bus

    def _unbind(self) -> None:
        """Clear subscriptions and release the connector. Called by SubprocessService."""
        self.unsubscribe()
        self._connector = None
        self._service_bus = None

    def _on_disconnect(self) -> None:
        """Called when the subprocess connection is lost unexpectedly.

        Clears pending requests (no replies will ever arrive) and resets visible
        state to NEW so the UI unlocks. Also unsubscribes domain handlers so a
        stale config-change event cannot try to send on a dead connector.
        """
        self.unsubscribe()
        with self._pending_lock:
            self._pending_count = 0
        self._worker_state._set(WorkerStatus.NEW)
