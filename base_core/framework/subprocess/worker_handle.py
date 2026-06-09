from __future__ import annotations

import dataclasses
from concurrent.futures import Future
from typing import Callable, Optional

from base_core.framework.subprocess.subprocess_service import SubprocessService
from base_core.framework.subprocess.shared_memory.shared_memory_base_messages import (
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
from base_core.framework.subprocess.worker_protocol import StartWorker, StopWorker
from base_core.framework.events.event_bus import EventBus
from base_core.framework.lifecycle.cleanup_collection import CleanupCollection
from messages import Message


# ---------------------------------------------------------------------------
# WorkerHandle
# ---------------------------------------------------------------------------

class WorkerHandle:
    """
    Main-process proxy for a named worker within a SubprocessService.

    Stamps the "target" field on every outgoing command so the SubprocessApp
    routes it to the correct worker thread.

    start() / stop() / restart() control the worker thread via the app-level
    StartWorker / StopWorker commands (which do NOT carry a target — they are
    handled by SubprocessApp directly).

    Obtain via SubprocessService.worker("name").
    """

    def __init__(self, service: SubprocessService, name: str) -> None:
        self._service = service
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Worker thread lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the worker thread. Blocks until the subprocess confirms."""
        reply = self._service.request_sync(StartWorker(worker_name=self._name))
        if not reply.get("ok"):
            raise RuntimeError(
                f"StartWorker failed for {self._name!r}: {reply.get('error')}"
            )

    def stop(self) -> None:
        """Stop the worker thread. Blocks until the subprocess confirms."""
        reply = self._service.request_sync(StopWorker(worker_name=self._name))
        if not reply.get("ok"):
            raise RuntimeError(
                f"StopWorker failed for {self._name!r}: {reply.get('error')}"
            )

    def restart(self) -> None:
        """Stop then re-start the worker thread."""
        self.stop()
        self.start()

    # ------------------------------------------------------------------
    # Command API
    # ------------------------------------------------------------------

    def send(self, cmd: Message) -> None:
        """Fire-and-forget targeted command to this worker."""
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


# ---------------------------------------------------------------------------
# OutputBufferHandle
# ---------------------------------------------------------------------------

class OutputBufferHandle:
    """
    Manages the shared-memory slot lifecycle for a ProducerWorker.

    Replaces SharedMemoryDeviceService for the SubprocessApp pattern. One
    handle per ProducerWorker; buffer_id is the worker name by convention.

    start() sequence:
      1. Subscribe to ItemWritten / ItemAck on the EventBus.
      2. Reset coordinator.
      3. Send ConfigureBuffer (targeted, blocking) — worker attaches to shared memory.
      4. Grant all initial slots.
      5. Send StartWorker (app-level, blocking) — worker thread begins.

    stop() sequence:
      1. Send StopWorker (app-level, blocking) — worker thread exits.
      2. Clear EventBus subscriptions.

    restart() = stop() + start().

    ack_slot() is called by in-process consumers (e.g. UI) when done reading a slot.
    """

    def __init__(
        self,
        handle: WorkerHandle,
        buffer: SharedRingBuffer,
        coordinator: SharedBufferCoordinator,
        bus: EventBus,
    ) -> None:
        self._handle = handle
        self._service = handle._service
        self._name = handle.name
        self._buffer = buffer
        self._coordinator = coordinator
        self._bus = bus
        self._cleanup = CleanupCollection()
        self._buffer_id = self._name  # buffer_id == worker name by convention

    @property
    def buffer(self) -> SharedRingBuffer:
        return self._buffer

    def start(self) -> None:
        """Configure buffer, grant slots, and start the worker thread."""
        self._cleanup.add(self._bus.subscribe(ItemWritten, self._on_item_written))
        self._cleanup.add(self._bus.subscribe(ItemAck, self._on_item_ack))

        self._coordinator.reset()

        reply = self._service.request_sync(
            self._handle._stamp(
                ConfigureBuffer(spec=self._buffer.spec, buffer_id=self._buffer_id)
            )
        )
        if not reply.get("ok"):
            raise RuntimeError(
                f"ConfigureBuffer failed for worker {self._name!r}: {reply.get('error')}"
            )

        for grant in self._coordinator.grant_initial_slots():
            self._service.send(
                self._handle._stamp(
                    SlotGranted(
                        slot=grant["slot"],
                        item_id=grant["item_id"],
                        buffer_id=self._buffer_id,
                    )
                )
            )

        self._handle.start()

    def stop(self) -> None:
        """Stop the worker thread and clear EventBus subscriptions."""
        self._handle.stop()
        self._cleanup.clear()

    def restart(self) -> None:
        """Stop then re-start: resets coordinator, re-sends ConfigureBuffer and grants."""
        self.stop()
        self.start()

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
                self._handle._stamp(
                    SlotGranted(
                        slot=result.grant["slot"],
                        item_id=result.grant["item_id"],
                        buffer_id=self._buffer_id,
                    )
                )
            )


# ---------------------------------------------------------------------------
# InputBufferHandle
# ---------------------------------------------------------------------------

class InputBufferHandle:
    """
    Registers a ConsumerWorker or ProcessorWorker as a consumer of an upstream
    shared-memory buffer and forwards ItemAvailable notifications to it.

    One handle per (worker, upstream buffer) pair. Does NOT own the worker thread
    lifecycle — call WorkerHandle.start() / .stop() separately, or rely on an
    OutputBufferHandle to do it.

    start() sequence:
      1. Register consumer_id in the upstream coordinator.
      2. Send ConfigureBuffer (targeted, blocking) so the worker can attach
         to the upstream buffer's shared memory.
      3. Subscribe to ItemAvailable on the EventBus; forward matching messages
         as targeted commands to the worker.

    stop() sequence:
      Clear EventBus subscriptions. Upstream slot grants drain naturally (the
      upstream coordinator re-grants when all consumers ack, which won't happen
      until the worker thread stops sending ItemAcks).

    Note: ItemAck flows from the worker back through the EventBus automatically —
    the upstream OutputBufferHandle picks it up by buffer_id filter.
    """

    def __init__(
        self,
        service: SubprocessService,
        worker_name: str,
        *,
        consumer_id: str,
        upstream_coordinator: SharedBufferCoordinator,
        upstream_buffer: SharedRingBuffer,
        buffer_id: str,
        bus: EventBus,
    ) -> None:
        self._service = service
        self._worker_name = worker_name
        self._consumer_id = consumer_id
        self._upstream_coordinator = upstream_coordinator
        self._upstream_buffer = upstream_buffer
        self._buffer_id = buffer_id
        self._bus = bus
        self._cleanup = CleanupCollection()
        self._registered = False

    def start(self) -> None:
        """
        Register consumer (first call only), configure input buffer in worker,
        and start forwarding ItemAvailable.

        Must be called BEFORE the upstream OutputBufferHandle.start() on first
        use — coordinator.register_consumer() requires all slots to be FREE,
        which is only guaranteed before grant_initial_slots() runs.

        On subsequent start() calls (after stop()/restart()) the consumer is
        already registered; only the ConfigureBuffer and subscription are redone.
        """
        if not self._registered:
            self._upstream_coordinator.register_consumer(self._consumer_id)
            self._registered = True

        reply = self._service.request_sync(
            _stamp(
                ConfigureBuffer(
                    spec=self._upstream_buffer.spec, buffer_id=self._buffer_id
                ),
                target=self._worker_name,
            )
        )
        if not reply.get("ok"):
            raise RuntimeError(
                f"ConfigureBuffer (input, buffer_id={self._buffer_id!r}) "
                f"failed for worker {self._worker_name!r}: {reply.get('error')}"
            )

        self._cleanup.add(
            self._bus.subscribe(ItemAvailable, self._on_item_available)
        )

    def stop(self) -> None:
        """Stop forwarding ItemAvailable notifications."""
        self._cleanup.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_item_available(self, msg: ItemAvailable) -> None:
        if msg.consumer_id != self._consumer_id or msg.buffer_id != self._buffer_id:
            return
        self._service.send(_stamp(msg, target=self._worker_name))


# ---------------------------------------------------------------------------
# Module-level helper (avoids importing dataclasses at every call site)
# ---------------------------------------------------------------------------

def _stamp(msg: Message, *, target: str) -> Message:
    return dataclasses.replace(msg, target=target)
