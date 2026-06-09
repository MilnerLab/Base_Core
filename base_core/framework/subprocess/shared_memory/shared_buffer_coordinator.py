from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from base_core.framework.subprocess.shared_memory.models import SlotInfo


AckOutcome = Literal["ignored", "pending", "completed"]


@dataclass(frozen=True)
class AckResult:
    """
    Result of processing a consumer ack.

    outcome:
      - "ignored":   stale / duplicate / wrong-state ack, nothing changed
      - "pending":   ack recorded, still waiting on other consumers
      - "completed": last ack received; slot freed and re-granted
    grant: the new grant produced when outcome == "completed", else None.
    """
    outcome: AckOutcome
    grant: dict | None = None


class SharedBufferCoordinator:
    def __init__(
        self,
        *,
        slot_count: int,
        consumer_bits: dict[str, int],
    ) -> None:
        self._slots = [SlotInfo() for _ in range(slot_count)]
        self._consumer_bits = dict(consumer_bits)
        self._required_consumer_count = len(consumer_bits)
        self._next_item_id = 1

    def reset(self) -> None:
        """
        Reset all slot states to FREE so the coordinator can grant initial
        slots again after a subprocess restart.

        _next_item_id is intentionally NOT reset: keeping it monotonically
        increasing means any ItemAck messages in flight from before the
        restart carry a stale item_id and are silently ignored.
        """
        for slot in self._slots:
            slot.state = "FREE"
            slot.item_id = None
            slot.pending_consumers = 0
            slot.acked_mask = 0

    def register_consumer(self, consumer_id: str) -> None:
        """
        Add a consumer dynamically.  Must be called before grant_initial_slots()
        (i.e., during the DI register() phase, before any on_startup() runs).
        """
        if consumer_id in self._consumer_bits:
            return  # already pre-registered during DI phase — idempotent
        if any(s.state != "FREE" for s in self._slots):
            raise RuntimeError(
                f"Cannot register consumer {consumer_id!r}: slots already in flight."
            )
        bit = 1 << len(self._consumer_bits)
        self._consumer_bits[consumer_id] = bit
        self._required_consumer_count += 1

    def grant_initial_slots(self) -> list[dict]:
        out: list[dict] = []
        for slot in range(len(self._slots)):
            out.append(self._allocate_slot(slot))
        return out

    def _allocate_slot(self, slot: int) -> dict:
        info = self._slots[slot]
        if info.state != "FREE":
            raise RuntimeError(f"Slot {slot} is not free.")

        item_id = self._next_item_id
        self._next_item_id += 1

        info.state = "WRITING"
        info.item_id = item_id
        info.pending_consumers = 0
        info.acked_mask = 0

        return {
            "slot": slot,
            "item_id": item_id,
        }

    def on_item_written(self, *, slot: int, item_id: int) -> list[dict]:
        info = self._slots[slot]
        if info.state != "WRITING":
            raise RuntimeError(f"Slot {slot} is in state {info.state}, expected WRITING.")
        if info.item_id != item_id:
            raise RuntimeError(
                f"Slot {slot} has item_id={info.item_id}, got {item_id}."
            )

        info.state = "PUBLISHED"
        info.pending_consumers = self._required_consumer_count
        info.acked_mask = 0

        return [
            {"consumer_id": consumer_id, "slot": slot, "item_id": item_id}
            for consumer_id in self._consumer_bits
        ]

    def on_item_ack(
        self,
        *,
        slot: int,
        item_id: int,
        consumer_id: str,
    ) -> AckResult:
        info = self._slots[slot]

        if info.state != "PUBLISHED":
            return AckResult("ignored")
        if info.item_id != item_id:
            return AckResult("ignored")

        bit = self._consumer_bits[consumer_id]
        if info.acked_mask & bit:
            return AckResult("ignored")

        info.acked_mask |= bit
        info.pending_consumers -= 1

        if info.pending_consumers > 0:
            return AckResult("pending")

        info.state = "FREE"
        info.item_id = None
        info.pending_consumers = 0
        info.acked_mask = 0

        return AckResult("completed", grant=self._allocate_slot(slot))