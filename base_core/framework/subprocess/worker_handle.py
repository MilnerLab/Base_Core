from __future__ import annotations

import dataclasses
from concurrent.futures import Future
from typing import Callable, Optional

from base_core.framework.subprocess.device_service import DeviceService
from base_core.framework.subprocess.shared_memory.base_protocol import (
    ConfigureBuffer,
    ItemAck,
    ItemAvailable,
    ItemWritten,
    SlotGranted,
)
from base_core.framework.subprocess.shared_memory.shared_buffer_coordinator import (
    SharedBufferCoordinator,
)
from base_core.framework.subprocess.shared_memory.shared_ring_buffer import SharedRingBuffer
from base_core.framework.events.event_bus import EventBus
from base_core.framework.lifecycle.cleanup_collection import CleanupCollection
from messages import Message


class WorkerHandle:
    """
    Main-process proxy for a named worker within a SubprocessService.

    Stamps the "target" field on every outgoing command so the SubprocessApp
    routes it to the correct worker thread. Thin wrapper — no state of its own.

    Obtain via SubprocessService.worker("name").
    """

    def __init__(self, service: DeviceService, name: str) -> None:
        self._service = service
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def send(self, cmd: Message) -> None:
        """Fire-and-forget command to this worker."""
        self._service.send(self._stamp(cmd))

    def request_async(
        self,
        cmd: Message,
        *,
        timeout_s: float = 2.0,
        key: Optional[str] = None,
        cancel_previous: bool = False,
        drop_outdated: bool = True,
        on_success: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ) -> Future:
        """Non-blocking request to this worker; returns a Future of the reply."""
        return self._service.request_async(
            self._stamp(cmd),
            timeout_s=timeout_s,
            key=key or f"worker.{self._name}.request",
            cancel_previous=cancel_previous,
            drop_outdated=drop_outdated,
            on_success=on_success,
            on_error=on_error,
        )

    def _stamp(self, cmd: Message) -> Message:
        """Return a copy of cmd with target set to this worker's name."""
        return dataclasses.replace(cmd, target=self._name)


class SharedWorkerHandle(WorkerHandle):
    """
    WorkerHandle for a ProducerWorker: also manages the shared-memory slot lifecycle.

    Replaces SharedMemoryDeviceService for the SubprocessApp pattern. One
    SharedWorkerHandle per ProducerWorker; the buffer_id is the worker name.

    Responsibilities:
      - start(): sends ConfigureBuffer (blocking), grants all initial slots.
      - Intercepts ItemWritten → drives coordinator → publishes ItemAvailable.
      - Intercepts ItemAck → re-grants freed slots back to the subprocess worker.
      - stop(): clears subscriptions.
      - ack_slot(): called by in-process consumers when done reading a slot.
    """

    def __init__(
        self,
        service: DeviceService,
        name: str,
        buffer: SharedRingBuffer,
        coordinator: SharedBufferCoordinator,
        bus: EventBus,
    ) -> None:
        super().__init__(service, name)
        self._buffer = buffer
        self._coordinator = coordinator
        self._bus = bus
        self._cleanup = CleanupCollection()
        # buffer_id matches the worker name so ItemWritten events can be filtered.
        self._buffer_id = name

    @property
    def buffer(self) -> SharedRingBuffer:
        return self._buffer

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Configure the shared buffer in the subprocess worker and start managing
        slot grants. Call after the SubprocessService has been started.
        """
        self._cleanup.add(
            self._bus.subscribe(ItemWritten, self._on_item_written)
        )
        self._cleanup.add(
            self._bus.subscribe(ItemAck, self._on_item_ack)
        )

        self._coordinator.reset()

        reply = self._service.request_sync(
            self._stamp(ConfigureBuffer(spec=self._buffer.spec, buffer_id=self._buffer_id))
        )
        if not reply.get("ok"):
            raise RuntimeError(
                f"ConfigureBuffer failed for worker {self._name!r}: {reply.get('error')}"
            )

        for grant in self._coordinator.grant_initial_slots():
            self._service.send(
                self._stamp(
                    SlotGranted(
                        slot=grant["slot"],
                        item_id=grant["item_id"],
                        buffer_id=self._buffer_id,
                    )
                )
            )

    def stop(self) -> None:
        """Remove bus subscriptions. Buffer lifetime is managed by the caller."""
        self._cleanup.clear()

    # ------------------------------------------------------------------
    # In-process consumer API
    # ------------------------------------------------------------------

    def ack_slot(self, slot: int, item_id: int, consumer_id: str) -> None:
        """Called by in-process consumers when done reading a slot."""
        self._do_ack(slot=slot, item_id=item_id, consumer_id=consumer_id)

    # ------------------------------------------------------------------
    # Internal coordinator handlers
    # ------------------------------------------------------------------

    def _on_item_written(self, msg: ItemWritten) -> None:
        if msg.buffer_id != self._buffer_id:
            return
        notifications = self._coordinator.on_item_written(slot=msg.slot, item_id=msg.item_id)
        for n in notifications:
            self._bus.publish(
                ItemAvailable(
                    consumer_id=n["consumer_id"],
                    slot=n["slot"],
                    item_id=n["item_id"],
                    timestamp_ns=msg.timestamp_ns,
                    buffer_id=self._buffer_id,
                )
            )

    def _on_item_ack(self, msg: ItemAck) -> None:
        if msg.buffer_id != self._buffer_id:
            return
        self._do_ack(slot=msg.slot, item_id=msg.item_id, consumer_id=msg.consumer_id)

    def _do_ack(self, *, slot: int, item_id: int, consumer_id: str) -> None:
        result = self._coordinator.on_item_ack(
            slot=slot, item_id=item_id, consumer_id=consumer_id
        )
        if result.outcome == "completed" and result.grant is not None:
            self._service.send(
                self._stamp(
                    SlotGranted(
                        slot=result.grant["slot"],
                        item_id=result.grant["item_id"],
                        buffer_id=self._buffer_id,
                    )
                )
            )
