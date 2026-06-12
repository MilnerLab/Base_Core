from __future__ import annotations

import logging
from collections import defaultdict, deque
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING

from base_core.framework.shm.messages import ItemAvailable, SlotGrant
from base_core.ipc.subprocess_main import BaseSubprocessMain

if TYPE_CHECKING:
    from base_core.framework.shm.buffer import SharedMemoryBuffer

log = logging.getLogger(__name__)


class WriterSubprocessMain(BaseSubprocessMain):
    """
    Subprocess counterpart to WriterSubprocessService.

    Extends BaseSubprocessMain with slot-grant tracking and write-notification helpers.
    Workers obtain a granted slot, write data via get_buffer(), then call notify_written().

    Typical worker pattern:
        slot = self.get_granted_slot(SpectrumBuffer)
        if slot is None:
            return  # no slot available yet, retry later
        self.get_buffer(SpectrumBuffer).write_spectrum(slot, wavelengths, intensities)
        self.notify_written(SpectrumBuffer, slot, item_id=..., timestamp_ns=...)
    """

    def __init__(self, conn: Connection) -> None:
        super().__init__(conn)
        self._granted: dict[str, deque[int]] = defaultdict(deque)

    def run(self) -> None:
        # Subscribe to SlotGrant before setup() so grants arriving during startup are queued
        self.bus.subscribe(SlotGrant, self._on_slot_grant)
        super().run()  # subscribes AttachBuffer, calls setup(), runs connector loop

    def _on_slot_grant(self, msg: SlotGrant) -> None:
        self._granted[msg.buffer_class_name].append(msg.slot)
        log.debug("WriterSubprocessMain: granted slot %d for %s", msg.slot, msg.buffer_class_name)

    def get_granted_slot(self, buffer_cls: type[SharedMemoryBuffer]) -> int | None:
        """Pop the next available slot for buffer_cls, or return None if the queue is empty."""
        q = self._granted.get(buffer_cls.__name__)
        if q:
            return q.popleft()
        return None

    def notify_written(
        self,
        buffer_cls: type[SharedMemoryBuffer],
        slot: int,
        item_id: int,
        timestamp_ns: int,
    ) -> None:
        """Send ItemAvailable to the main-process service after writing a slot."""
        self.connector.send(ItemAvailable(
            buffer_class_name=buffer_cls.__name__,
            slot=slot,
            item_id=item_id,
            timestamp_ns=timestamp_ns,
        ))
