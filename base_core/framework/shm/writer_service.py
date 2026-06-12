from __future__ import annotations

import logging
from typing import Callable, Generic, TypeVar

from base_core.framework.concurrency.task_runner import TaskRunner
from base_core.framework.events.event_bus import EventBus
from base_core.framework.shm.buffer import SharedMemoryBuffer
from base_core.framework.shm.messages import ItemAvailable, SlotGrant
from base_core.framework.shm.slot_coordinator import SlotCoordinator
from base_core.framework.shm.spec import MemorySpec
from base_core.ipc.subprocess_service import SubprocessService

log = logging.getLogger(__name__)

TAvailable = TypeVar("TAvailable")
TAck = TypeVar("TAck")


class WriterSubprocessService(SubprocessService, Generic[TAvailable, TAck]):
    """
    Extends SubprocessService for the case where the subprocess writes bulk data into
    shared memory. Owns the SharedMemoryBuffer and SlotCoordinator; orchestrates the
    SlotGrant ↔ ItemAvailable message loop automatically.

    The spec is typically created in the Module and passed in so that other services
    (read-only consumers) can receive it via DI.

    Lifecycle:
        start() → creates buffer, starts subprocess, sends AttachBuffer + initial SlotGrants
        [subprocess writes] → ItemAvailable → coordinator.on_written → TAvailable event on bus
        [consumers ack TAck] → coordinator frees slot → SlotGrant back to subprocess
        stop() → coordinator.stop, super.stop, buffer.unlink/close
    """

    def __init__(
        self,
        bus: EventBus,
        io: TaskRunner,
        buffer_cls: type[SharedMemoryBuffer],
        spec: MemorySpec,
        coordinator: SlotCoordinator[TAvailable, TAck],
        python_exe: str | None = None,
    ) -> None:
        super().__init__(bus, io, python_exe)
        self._buffer_cls = buffer_cls
        self._spec = spec
        self._coordinator = coordinator
        self._writer_buffer: SharedMemoryBuffer | None = None
        self._item_unsub: Callable[[], None] | None = None
        # Register the buffer so base start() sends AttachBuffer to the subprocess
        self.add_buffer(buffer_cls, spec)

    @property
    def spec(self) -> MemorySpec:
        return self._spec

    def register_consumer(self, consumer_id: str) -> None:
        self._coordinator.register_consumer(consumer_id)

    def unregister_consumer(self, consumer_id: str) -> None:
        self._coordinator.unregister_consumer(consumer_id)

    def start(self) -> None:
        self._writer_buffer = self._buffer_cls.create(self._spec)
        super().start()  # starts subprocess + sends AttachBuffer
        self._coordinator.start(on_slot_freed=self._on_slot_freed)
        self._item_unsub = self._bus.subscribe(ItemAvailable, self._on_item_available)
        # Grant all slots upfront; they are replenished as consumers ack
        for slot in range(self._spec.slot_count):
            self.emit(SlotGrant(buffer_class_name=self._buffer_cls.__name__, slot=slot))

    def stop(self) -> None:
        if self._item_unsub is not None:
            self._item_unsub()
            self._item_unsub = None
        self._coordinator.stop()
        super().stop()  # terminates subprocess
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
        self.emit(SlotGrant(buffer_class_name=self._buffer_cls.__name__, slot=slot))
