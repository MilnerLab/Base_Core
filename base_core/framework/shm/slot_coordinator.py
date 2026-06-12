from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

from base_core.framework.events.event_bus import EventBus
from base_core.framework.shm.spec import MemorySpec

log = logging.getLogger(__name__)

TAvailable = TypeVar("TAvailable")
TAck = TypeVar("TAck")


@dataclass
class _SlotState:
    item_id: int = -1
    is_free: bool = True
    pending_consumers: set[str] = field(default_factory=set)


class SlotCoordinator(Generic[TAvailable, TAck]):
    """
    Tracks slot availability and consumer acknowledgements for one shared memory buffer.

    Lives in the main process. The owning WriterSubprocessService drives it:
      acquire_slot() → [subprocess writes] → on_written() → [consumers ack via bus] → slot freed

    The coordinator publishes TAvailable events on the bus when data is ready, and subscribes
    to TAck events to detect when all consumers have finished reading a slot.

    TAck instances must have .slot (int), .item_id (int), and .consumer_id (str) attributes.
    TAvailable is constructed via make_available(slot, item_id, timestamp_ns).
    """

    def __init__(
        self,
        spec: MemorySpec,
        owner_id: str,
        bus: EventBus,
        make_available: Callable[[int, int, int], TAvailable],
        ack_type: type[TAck],
    ) -> None:
        self._spec = spec
        self._owner_id = owner_id
        self._bus = bus
        self._make_available = make_available
        self._ack_type = ack_type
        self._slots: list[_SlotState] = [_SlotState() for _ in range(spec.slot_count)]
        self._consumers: set[str] = set()
        self._lock = threading.Lock()
        self._unsub: Callable[[], None] | None = None
        self._on_slot_freed_fn: Callable[[int], None] | None = None

    @property
    def owner_id(self) -> str:
        return self._owner_id

    def start(self, on_slot_freed: Callable[[int], None]) -> None:
        """Subscribe to ack events. on_slot_freed is called when a slot becomes free."""
        self._on_slot_freed_fn = on_slot_freed
        self._unsub = self._bus.subscribe(self._ack_type, self._on_ack)

    def stop(self) -> None:
        """Unsubscribe from ack events."""
        if self._unsub is not None:
            self._unsub()
            self._unsub = None

    # ------------------------------------------------------------------
    # Consumer registration
    # ------------------------------------------------------------------

    def register_consumer(self, consumer_id: str) -> None:
        """Add a consumer. No-op if already registered."""
        with self._lock:
            self._consumers.add(consumer_id)

    def unregister_consumer(self, consumer_id: str) -> None:
        """
        Remove a consumer. Auto-acks all pending slots for this consumer
        so no slot gets stuck waiting for a gone consumer.
        """
        freed: list[int] = []
        with self._lock:
            self._consumers.discard(consumer_id)
            for i, state in enumerate(self._slots):
                if not state.is_free:
                    state.pending_consumers.discard(consumer_id)
                    if not state.pending_consumers:
                        state.is_free = True
                        freed.append(i)
        for slot in freed:
            self._notify_freed(slot)

    # ------------------------------------------------------------------
    # Slot lifecycle
    # ------------------------------------------------------------------

    def acquire_slot(self) -> int | None:
        """Return a free slot index, or None if all slots are currently pending."""
        with self._lock:
            for i, state in enumerate(self._slots):
                if state.is_free:
                    return i
        return None

    def on_written(self, slot: int, item_id: int, timestamp_ns: int) -> None:
        """
        Mark slot as pending with the current consumer set, then publish the available event.
        If there are no registered consumers the slot is immediately freed.
        """
        with self._lock:
            state = self._slots[slot]
            state.item_id = item_id
            if self._consumers:
                state.is_free = False
                state.pending_consumers = set(self._consumers)
            else:
                state.is_free = True
                state.pending_consumers = set()

        if not self._consumers:
            self._notify_freed(slot)
            return

        self._bus.publish(self._make_available(slot, item_id, timestamp_ns))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_ack(self, event: TAck) -> None:
        slot: int = getattr(event, "slot")
        item_id: int = getattr(event, "item_id")
        consumer_id: str = getattr(event, "consumer_id")

        freed = False
        with self._lock:
            state = self._slots[slot]
            if state.is_free or state.item_id != item_id:
                return  # stale ack
            if consumer_id not in state.pending_consumers:
                return  # unknown or already acked
            state.pending_consumers.discard(consumer_id)
            if not state.pending_consumers:
                state.is_free = True
                freed = True

        if freed:
            self._notify_freed(slot)

    def _notify_freed(self, slot: int) -> None:
        if self._on_slot_freed_fn is not None:
            try:
                self._on_slot_freed_fn(slot)
            except Exception:
                log.exception("SlotCoordinator: on_slot_freed callback failed for slot %d", slot)
