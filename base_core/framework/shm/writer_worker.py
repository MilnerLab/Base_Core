from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Callable, Generic, TypeVar

from base_core.framework.events.event_bus import EventBus
from base_core.framework.shm.messages import ItemAvailable, SlotGrant
from base_core.ipc.subprocess_connector import SubprocessPipelineConnector
from base_core.ipc.threaded_worker import ProducingThreadedWorker

log = logging.getLogger(__name__)

TBuffer = TypeVar("TBuffer")


class WriterWorker(ProducingThreadedWorker, Generic[TBuffer]):
    """
    Subprocess-side worker for shared memory writes.

    Handles the SlotGrant queue and ItemAvailable notification so subclasses
    only need to implement the acquisition logic. Buffer attachment remains at
    the BaseSubprocessMain level — this worker receives the buffer via the
    injected get_buffer callable.

    Subclass pattern:
        class MyWorker(WriterWorker[MyBuffer]):
            def _setup(self) -> None:
                super()._setup()  # registers SlotGrant subscription
                self._unsubs.append(self._bus.subscribe(MyCommand, self._on_cmd))

            def _start(self) -> None:
                self._driver.open()
                self._start_producing(self._producer, on_item=self._on_item)

            def _producer(self, stop):
                while not stop.is_set():
                    slot = self._get_slot()
                    if slot is None:
                        time.sleep(0.001)
                        continue
                    data = self._driver.read()
                    yield (slot, data)

            def _on_item(self, item):
                slot, data = item
                self._get_buffer().write(slot, data)
                self._item_id += 1
                self._notify_written(slot, self._item_id, data.timestamp_ns)
    """

    def __init__(
        self,
        worker_id: str,
        bus: EventBus,
        connector: SubprocessPipelineConnector,
        buffer_cls: type[TBuffer],
        get_buffer: Callable[[], TBuffer],
    ) -> None:
        super().__init__(worker_id, bus, connector)
        self._buffer_cls = buffer_cls
        self._get_buffer_fn = get_buffer
        self._granted: deque[int] = deque()
        self._granted_lock = threading.Lock()

    def _setup(self) -> None:
        self._unsubs.append(self._bus.subscribe(SlotGrant, self._on_slot_grant))

    def _on_slot_grant(self, msg: SlotGrant) -> None:
        if msg.buffer_class_name == self._buffer_cls.__name__:
            with self._granted_lock:
                self._granted.append(msg.slot)

    def _get_slot(self) -> int | None:
        with self._granted_lock:
            if self._granted:
                return self._granted.popleft()
        return None

    def _get_buffer(self) -> TBuffer:
        return self._get_buffer_fn()

    def _notify_written(self, slot: int, item_id: int, timestamp_ns: int) -> None:
        self._notify(ItemAvailable(
            buffer_class_name=self._buffer_cls.__name__,
            slot=slot,
            item_id=item_id,
            timestamp_ns=timestamp_ns,
        ))
