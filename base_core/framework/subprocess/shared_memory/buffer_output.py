from __future__ import annotations

from typing import Callable, Generic, Optional, TypeVar

from base_core.framework.events.event_bus import EventBus
from base_core.framework.subprocess.shared_memory.shared_buffer_coordinator import (
    SharedBufferCoordinator,
)

AvailableT = TypeVar("AvailableT")
AckT = TypeVar("AckT")


class BufferOutput(Generic[AvailableT, AckT]):
    """
    Pairs a SharedBufferCoordinator with the bus events that represent item
    availability (AvailableT) and acknowledgement (AckT).

    Publishes a single broadcast AvailableT event per frame (no consumer_id).
    All registered consumers receive every frame and ack independently.

    Call start() when the service starts and stop() when it stops.
    """

    def __init__(
        self,
        coordinator: SharedBufferCoordinator,
        send_grant: Callable[[dict], None],
        bus: EventBus,
        available_cls: type[AvailableT],
        ack_cls: type[AckT],
        buffer_id: str = "",
    ) -> None:
        self._coordinator = coordinator
        self._send_grant = send_grant
        self._bus = bus
        self._available_cls = available_cls
        self._ack_cls = ack_cls
        self._buffer_id = buffer_id
        self._ack_unsub: Optional[Callable[[], None]] = None

    @property
    def coordinator(self) -> SharedBufferCoordinator:
        return self._coordinator

    @property
    def buffer_id(self) -> str:
        return self._buffer_id

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._ack_unsub = self._bus.subscribe(self._ack_cls, self._on_ack)

    def stop(self) -> None:
        if self._ack_unsub is not None:
            self._ack_unsub()
            self._ack_unsub = None

    # ------------------------------------------------------------------
    # Consumer registration
    # ------------------------------------------------------------------

    def register_consumer(self, consumer_id: str) -> None:
        for grant in self._coordinator.register_consumer(consumer_id):
            self._send_grant(grant)

    def unregister_consumer(self, consumer_id: str) -> None:
        for grant in self._coordinator.unregister_consumer(consumer_id):
            self._send_grant(grant)

    # ------------------------------------------------------------------
    # Notification
    # ------------------------------------------------------------------

    def notify_available(self, slot: int, item_id: int, timestamp_ns: int) -> None:
        self._bus.publish(self._available_cls(
            slot=slot, item_id=item_id, timestamp_ns=timestamp_ns,
        ))

    def subscribe_available(
        self,
        bus: EventBus,
        handler: Callable[[AvailableT], None],
    ) -> Callable[[], None]:
        return bus.subscribe(self._available_cls, handler)

    # ------------------------------------------------------------------
    # Ack
    # ------------------------------------------------------------------

    def send_grant(self, grant: dict) -> None:
        """Send a single slot grant (e.g. from initial grants or catch-up after registration)."""
        self._send_grant(grant)

    def ack_slot(self, slot: int, item_id: int, consumer_id: str) -> None:
        self._commit_ack(slot, item_id, consumer_id)

    def _commit_ack(self, slot: int, item_id: int, consumer_id: str) -> None:
        result = self._coordinator.on_item_ack(slot=slot, item_id=item_id, consumer_id=consumer_id)
        if result.outcome == "completed" and result.grant is not None:
            self._send_grant(result.grant)

    def _on_ack(self, event: AckT) -> None:
        self._commit_ack(
            event.slot,        # type: ignore[union-attr]
            event.item_id,     # type: ignore[union-attr]
            event.consumer_id, # type: ignore[union-attr]
        )
