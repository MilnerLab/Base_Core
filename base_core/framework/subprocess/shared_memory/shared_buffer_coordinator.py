from base_core.framework.subprocess.shared_memory.models import SlotInfo


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
        self._next_frame_id = 1

    def grant_initial_slots(self) -> list[dict]:
        out: list[dict] = []
        for slot in range(len(self._slots)):
            out.append(self._allocate_slot(slot))
        return out

    def _allocate_slot(self, slot: int) -> dict:
        info = self._slots[slot]
        if info.state != "FREE":
            raise RuntimeError(f"Slot {slot} is not free.")

        frame_id = self._next_frame_id
        self._next_frame_id += 1

        info.state = "WRITING"
        info.frame_id = frame_id
        info.pending_consumers = 0
        info.acked_mask = 0

        return {
            "slot": slot,
            "frame_id": frame_id,
        }

    def on_frame_written(self, *, slot: int, frame_id: int) -> list[dict]:
        info = self._slots[slot]
        if info.state != "WRITING":
            raise RuntimeError(f"Slot {slot} is in state {info.state}, expected WRITING.")
        if info.frame_id != frame_id:
            raise RuntimeError(
                f"Slot {slot} has frame_id={info.frame_id}, got {frame_id}."
            )

        info.state = "PUBLISHED"
        info.pending_consumers = self._required_consumer_count
        info.acked_mask = 0

        return [
            {"consumer_id": consumer_id, "slot": slot, "frame_id": frame_id}
            for consumer_id in self._consumer_bits
        ]

    def on_frame_ack(
        self,
        *,
        slot: int,
        frame_id: int,
        consumer_id: str,
    ) -> dict | None:
        info = self._slots[slot]

        if info.state != "PUBLISHED":
            return None
        if info.frame_id != frame_id:
            return None

        bit = self._consumer_bits[consumer_id]
        if info.acked_mask & bit:
            return None

        info.acked_mask |= bit
        info.pending_consumers -= 1

        if info.pending_consumers > 0:
            return None

        info.state = "FREE"
        info.frame_id = None
        info.pending_consumers = 0
        info.acked_mask = 0

        return self._allocate_slot(slot)