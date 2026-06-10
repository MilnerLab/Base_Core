from __future__ import annotations

from typing import Callable, Generic, Optional, TypeVar

from base_core.framework.events.event_bus import EventBus
from base_core.framework.subprocess.shared_memory.shared_buffer_coordinator import (
    SharedBufferCoordinator,
)

_SlotCallback = Callable[[str, int, int, int], None]  # (consumer_id, slot, item_id, timestamp_ns)

AvailableT = TypeVar("AvailableT")
AckT = TypeVar("AckT")


class BufferOutput(Generic[AvailableT, AckT]):
    """
    Pairs a SharedBufferCoordinator with the bus events that represent item
    availability (AvailableT) and acknowledgement (AckT).

    Call start() when the service starts and stop() when it stops.

    Tracks in-flight notifications per consumer so that unregister_consumer()
    can force-ack any outstanding slots, preventing ring-buffer stalls when a
    UI consumer closes mid-stream.
    """

    def __init__(
        self,
        coordinator: SharedBufferCoordinator,
        send_grant: Callable[[dict], None],
        bus: EventBus,
        available_cls: type[AvailableT],
        ack_cls: type[AckT],
    ) -> None:
        self._coordinator = coordinator
        self._send_grant = send_grant
        self._bus = bus
        self._available_cls = available_cls
        self._ack_cls = ack_cls
        self._listeners: list[_SlotCallback] = []
        self._pending: dict[str, list[tuple[int, int]]] = {}  # consumer_id -> [(slot, item_id)]
        self._ack_unsub: Optional[Callable[[], None]] = None
        self._available_unsub: Optional[Callable[[], None]] = None

    @property
    def coordinator(self) -> SharedBufferCoordinator:
        return self._coordinator

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._available_unsub = self.add_item_listener(self._publish_available)
        self._ack_unsub = self._bus.subscribe(self._ack_cls, self._on_ack)

    def stop(self) -> None:
        if self._ack_unsub is not None:
            self._ack_unsub()
            self._ack_unsub = None
        if self._available_unsub is not None:
            self._available_unsub()
            self._available_unsub = None

    # ------------------------------------------------------------------
    # Consumer registration
    # ------------------------------------------------------------------

    def register_consumer(self, consumer_id: str) -> None:
        for grant in self._coordinator.register_consumer(consumer_id):
            self._send_grant(grant)

    def unregister_consumer(self, consumer_id: str) -> None:
        # Force-ack any notified-but-unacked slots so the ring buffer never stalls.
        for slot, item_id in self._pending.pop(consumer_id, []):
            self._commit_ack(slot, item_id, consumer_id)
        for grant in self._coordinator.unregister_consumer(consumer_id):
            self._send_grant(grant)

    # ------------------------------------------------------------------
    # Item listener API (used by WorkerHandle for subprocess slot forwarding)
    # ------------------------------------------------------------------

    def add_item_listener(self, callback: _SlotCallback) -> Callable[[], None]:
        """Register callback(consumer_id, slot, item_id, timestamp_ns). Returns unsubscribe."""
        self._listeners.append(callback)
        return lambda: self._listeners.remove(callback)

    def _notify_item_available(self, consumer_id: str, slot: int, item_id: int, timestamp_ns: int) -> None:
        self._pending.setdefault(consumer_id, []).append((slot, item_id))
        for cb in list(self._listeners):
            cb(consumer_id, slot, item_id, timestamp_ns)

    # ------------------------------------------------------------------
    # Ack (called directly by WorkerHandle for subprocess consumers)
    # ------------------------------------------------------------------

    def ack_slot(self, slot: int, item_id: int, consumer_id: str) -> None:
        pending = self._pending.get(consumer_id, [])
        entry = (slot, item_id)
        if entry in pending:
            pending.remove(entry)
        self._commit_ack(slot, item_id, consumer_id)

    def _commit_ack(self, slot: int, item_id: int, consumer_id: str) -> None:
        result = self._coordinator.on_item_ack(slot=slot, item_id=item_id, consumer_id=consumer_id)
        if result.outcome == "completed" and result.grant is not None:
            self._send_grant(result.grant)

    # ------------------------------------------------------------------
    # Internal — bus wiring
    # ------------------------------------------------------------------

    def _publish_available(self, consumer_id: str, slot: int, item_id: int, timestamp_ns: int) -> None:
        self._bus.publish(self._available_cls(
            consumer_id=consumer_id, slot=slot, item_id=item_id, timestamp_ns=timestamp_ns,
        ))

    def _on_ack(self, event: AckT) -> None:
        slot: int = event.slot  # type: ignore[union-attr]
        item_id: int = event.item_id  # type: ignore[union-attr]
        consumer_id: str = event.consumer_id  # type: ignore[union-attr]
        pending = self._pending.get(consumer_id, [])
        entry = (slot, item_id)
        if entry in pending:
            pending.remove(entry)
        self._commit_ack(slot, item_id, consumer_id)
