from __future__ import annotations

import logging
import threading
from typing import Callable, Generic, TypeVar

from base_core.framework.events.event_bus import EventBus
from base_core.framework.shm.spec import MemorySpec

log = logging.getLogger(__name__)

TAvailable = TypeVar("TAvailable")
TAck = TypeVar("TAck")


class SlotCoordinator(Generic[TAvailable, TAck]):
    """
    Always-latest double-buffer coordinator for one shared memory buffer.

    Manages two slots:
    - shadow: producer writes here; re-granted immediately after each write so the
              producer can overwrite with fresher data as fast as it likes.
    - active: consumer reads here; replaced with the latest shadow content after all
              consumers ack.

    When the consumer is slower than the producer, intermediate shadow writes are
    silently dropped. The consumer always receives the write that completed just after
    it freed the previous slot — guaranteeing the freshest available frame.

    Safety invariant: the swap shadow→active happens only when on_written(shadow) fires.
    At that instant the producer has just finished writing shadow and holds no grant.
    The old active (new shadow) is immediately re-granted to the producer, so no
    outstanding grant ever points at the active slot.

    TAck instances must have .slot (int), .item_id (int), and .consumer_id (str).
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
        self._owner_id = owner_id
        self._bus = bus
        self._make_available = make_available
        self._ack_type = ack_type
        self._lock = threading.Lock()

        self._shadow: int = 0           # producer writes here (initial: slot 0)
        self._active: int = 1           # consumer reads here
        self._consumer_waiting: bool = True  # True = idle; False = reading active
        self._shadow_item_id: int = -1
        self._shadow_ts: int = 0
        self._active_item_id: int = -1
        self._consumers: set[str] = set()
        self._active_pending: set[str] = set()

        self._unsub: Callable[[], None] | None = None
        self._on_slot_freed_fn: Callable[[int], None] | None = None

    @property
    def owner_id(self) -> str:
        return self._owner_id

    @property
    def shadow(self) -> int:
        """Current shadow slot index. Equal to 0 before any write; rotates thereafter."""
        return self._shadow

    def start(self, on_slot_freed: Callable[[int], None]) -> None:
        """Subscribe to ack events. on_slot_freed sends SlotGrant back to subprocess."""
        self._on_slot_freed_fn = on_slot_freed
        self._unsub = self._bus.subscribe(self._ack_type, self._on_ack)

    def stop(self) -> None:
        if self._unsub is not None:
            self._unsub()
            self._unsub = None

    # ------------------------------------------------------------------
    # Consumer registration
    # ------------------------------------------------------------------

    def register_consumer(self, consumer_id: str) -> None:
        with self._lock:
            self._consumers.add(consumer_id)

    def unregister_consumer(self, consumer_id: str) -> None:
        """Remove consumer. If it was the last pending ack, mark consumer as waiting."""
        with self._lock:
            self._consumers.discard(consumer_id)
            self._active_pending.discard(consumer_id)
            if not self._active_pending and not self._consumer_waiting:
                self._consumer_waiting = True
        # Next on_written will see _consumer_waiting=True and promote; no action needed here.

    # ------------------------------------------------------------------
    # Slot lifecycle
    # ------------------------------------------------------------------

    def on_written(self, slot: int, item_id: int, timestamp_ns: int) -> None:
        """
        Called when subprocess reports it has written to `slot`.

        If the consumer is idle (_consumer_waiting): swap shadow↔active, re-grant new
        shadow (old active) to producer, publish availability event to consumers.

        If the consumer is busy: re-grant shadow immediately so the producer can
        overwrite with fresher data; record latest item_id/ts in case consumer acks soon.
        """
        new_shadow: int
        event: TAvailable | None = None

        with self._lock:
            if slot != self._shadow:
                return  # stale — shouldn't happen under normal operation
            self._shadow_item_id = item_id
            self._shadow_ts = timestamp_ns

            if not self._consumers or not self._consumer_waiting:
                # No consumers, or consumer still busy: re-grant shadow immediately.
                new_shadow = self._shadow
            else:
                # Consumer waiting: promote shadow → active.
                self._shadow, self._active = self._active, self._shadow
                self._active_item_id = item_id
                self._active_pending = set(self._consumers)
                self._consumer_waiting = False
                new_shadow = self._shadow   # old active is now the new shadow
                event = self._make_available(self._active, item_id, timestamp_ns)

        # Re-grant new shadow to producer (outside lock so callback can re-enter safely).
        self._notify_freed(new_shadow)
        if event is not None:
            self._bus.publish(event)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_ack(self, event: TAck) -> None:
        slot: int = getattr(event, "slot")
        item_id: int = getattr(event, "item_id")
        consumer_id: str = getattr(event, "consumer_id")

        with self._lock:
            if slot != self._active:
                return  # ack for shadow slot — stale
            if item_id != self._active_item_id:
                return  # stale item_id
            if consumer_id not in self._active_pending:
                return  # unknown or already acked
            self._active_pending.discard(consumer_id)
            if self._active_pending:
                return  # other consumers still reading
            self._consumer_waiting = True
        # No swap here: wait for next on_written to guarantee the shadow grant is consumed
        # before we rotate it into the active role.

    def _notify_freed(self, slot: int) -> None:
        if self._on_slot_freed_fn is not None:
            try:
                self._on_slot_freed_fn(slot)
            except Exception:
                log.exception("SlotCoordinator: on_slot_freed callback failed for slot %d", slot)
