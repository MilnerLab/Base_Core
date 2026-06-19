from __future__ import annotations

import logging
from typing import Callable, Generic, TypeVar

from base_core.framework.events.event_bus import EventBus
from base_core.framework.shm.buffer import SharedMemoryBuffer
from base_core.framework.shm.messages import ItemAvailable, SlotGrant
from base_core.framework.shm.slot_coordinator import SlotCoordinator
from base_core.framework.shm.spec import MemorySpec
from base_core.ipc.worker_handle import BaseWorkerHandle

log = logging.getLogger(__name__)

TBuffer = TypeVar("TBuffer", bound=SharedMemoryBuffer)
TAvailable = TypeVar("TAvailable")
TAck = TypeVar("TAck")


class WriterWorkerHandle(BaseWorkerHandle, Generic[TBuffer, TAvailable, TAck]):
    """
    Main-process counterpart to WriterWorker.

    Owns the SharedMemoryBuffer (created before the subprocess attaches, destroyed on
    unbind) and the SlotCoordinator that tracks slot availability and consumer acks.

    The caller must register the buffer with the service before adding this handle:
        service.add_buffer(buffer_cls, spec)
        service.add_handle(handle)

    Lifecycle (driven by SubprocessService):
      _bind(connector, service_bus)
      _on_pre_attach()   → create buffer in shared memory
      [AttachBuffer sent to subprocess]
      _on_attached()     → subscribe ItemAvailable, start coordinator, grant all slots
      _unbind()          → stop coordinator, unsubscribe, close+unlink buffer

    Consumer registration (called by read-only consumers, e.g. PhaseControlService):
      register_consumer(id) / unregister_consumer(id)
    """

    def __init__(
        self,
        worker_id: str,
        bus: EventBus,
        buffer_cls: type[TBuffer],
        spec: MemorySpec,
        make_available: Callable[[int, int, int], TAvailable],
        ack_type: type[TAck],
    ) -> None:
        super().__init__(worker_id, bus)
        self._buffer_cls = buffer_cls
        self._spec = spec
        self._coordinator: SlotCoordinator[TAvailable, TAck] = SlotCoordinator(
            spec, worker_id, bus, make_available, ack_type
        )
        self._writer_buffer: TBuffer | None = None

    @property
    def spec(self) -> MemorySpec:
        return self._spec

    def register_consumer(self, consumer_id: str) -> None:
        self._coordinator.register_consumer(consumer_id)

    def unregister_consumer(self, consumer_id: str) -> None:
        self._coordinator.unregister_consumer(consumer_id)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def _on_pre_attach(self) -> None:
        self._writer_buffer = self._buffer_cls.create(self._spec)

    def _on_attached(self) -> None:
        self._subscribe_service(ItemAvailable, self._on_item_available)
        self._coordinator.start(on_slot_freed=self._on_slot_freed)
        self._emit(SlotGrant(buffer_class_name=self._buffer_cls.__name__, slot=self._coordinator.shadow))

    def _on_detached(self) -> None:
        self._coordinator.stop()
        super()._on_detached()  # clears _unsubs (unsubscribes ItemAvailable)
        if self._writer_buffer is not None:
            self._writer_buffer.unlink()
            self._writer_buffer.close()
            self._writer_buffer = None

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _on_item_available(self, msg: ItemAvailable) -> None:
        if msg.buffer_class_name == self._buffer_cls.__name__:
            self._coordinator.on_written(msg.slot, msg.item_id, msg.timestamp_ns)

    def _on_slot_freed(self, slot: int) -> None:
        self._emit(SlotGrant(buffer_class_name=self._buffer_cls.__name__, slot=slot))
