from __future__ import annotations

from typing import Callable

from base_core.framework.subprocess.shared_memory.shared_buffer_coordinator import (
    SharedBufferCoordinator,
)


class BufferOutput:
    """
    Pairs a SharedBufferCoordinator with the callable that forwards grants to the
    subprocess writer.  Services create one and expose it; VMs and BufferConsumerMixin
    use it directly so neither the coordinator nor the service internals leak out.
    """

    def __init__(
        self,
        coordinator: SharedBufferCoordinator,
        send_grant: Callable[[dict], None],
        on_register: Callable[[str], None] | None = None,
        on_unregister: Callable[[str], None] | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._send_grant = send_grant
        self._on_register = on_register
        self._on_unregister = on_unregister

    @property
    def coordinator(self) -> SharedBufferCoordinator:
        return self._coordinator

    def register_consumer(self, consumer_id: str) -> None:
        for grant in self._coordinator.register_consumer(consumer_id):
            self._send_grant(grant)
        if self._on_register is not None:
            self._on_register(consumer_id)

    def unregister_consumer(self, consumer_id: str) -> None:
        for grant in self._coordinator.unregister_consumer(consumer_id):
            self._send_grant(grant)
        if self._on_unregister is not None:
            self._on_unregister(consumer_id)

    def ack_slot(self, slot: int, item_id: int, consumer_id: str) -> None:
        result = self._coordinator.on_item_ack(slot=slot, item_id=item_id, consumer_id=consumer_id)
        if result.outcome == "completed" and result.grant is not None:
            self._send_grant(result.grant)
