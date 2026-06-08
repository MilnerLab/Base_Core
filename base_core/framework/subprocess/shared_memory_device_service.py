from __future__ import annotations

from typing import Optional

from base_core.framework.concurrency.task_runner import TaskRunner
from base_core.framework.events.event_bus import EventBus
from base_core.framework.lifecycle.cleanup_collection import CleanupCollection
from base_core.framework.subprocess.device_service import DeviceService
from base_core.framework.subprocess.json_endpoint import JsonlSubprocessEndpoint
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


class SharedMemoryDeviceService(DeviceService):
    """
    DeviceService extension for subprocesses that write bulk data into a
    shared-memory ring buffer.

    Responsibilities on top of DeviceService:
      - Owns the SharedBufferCoordinator for one output buffer.
      - On start(): sends ConfigureBuffer to the subprocess, then grants all
        initial slots so the subprocess can start writing immediately.
      - Intercepts ItemWritten events from the subprocess → drives the
        coordinator → publishes ItemAvailable to the EventBus for each consumer.
      - Intercepts ItemAck events (from subprocess consumers and from in-process
        consumers via ack_slot()) → drives the coordinator → re-grants freed
        slots back to the subprocess.

    Consumer types:
      - In-process (UI, analysis running in main process): call ack_slot() when
        done reading a slot.
      - Subprocess consumers: send ItemAck(buffer_id=...) back over their own
        JSONL connection; those events land on the shared EventBus and are
        picked up here automatically.

    `source` is the subprocess source identifier used to filter ItemWritten
    events on the bus (so this service only reacts to events from its own
    subprocess, not from others on the same bus).

    `buffer_id` must match the value used in ConfigureBuffer / SlotGranted /
    ItemWritten / ItemAck.  Use the default "" when the subprocess has a single
    output buffer.
    """

    def __init__(
        self,
        io: TaskRunner,
        endpoint: JsonlSubprocessEndpoint,
        bus: EventBus,
        buffer: SharedRingBuffer,
        coordinator: SharedBufferCoordinator,
        *,
        source: Optional[str] = None,
        buffer_id: str = "",
    ) -> None:
        super().__init__(io, endpoint, bus)
        self._buffer = buffer
        self._coordinator = coordinator
        self._source = source
        self._buffer_id = buffer_id
        self._cleanup = CleanupCollection()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def buffer(self) -> SharedRingBuffer:
        return self._buffer

    def start(self) -> None:
        super().start()

        # Subscribe to coordinator events on the shared bus.
        # ItemWritten: filtered by source so we only handle our own subprocess.
        # ItemAck: no source filter — acks can come from any consumer subprocess.
        self._cleanup.add(
            self._bus.subscribe(ItemWritten, self._on_item_written, source=self._source)
        )
        self._cleanup.add(
            self._bus.subscribe(ItemAck, self._on_item_ack)
        )

        # Reset coordinator so any stale slot state from a previous run is
        # cleared before we hand out fresh grants.
        self._coordinator.reset()

        # Tell the subprocess which shared memory segment to attach to.
        reply = self._endpoint.raw_request(
            ConfigureBuffer(spec=self._buffer.spec, buffer_id=self._buffer_id)
        )
        if not reply.get("ok"):
            raise RuntimeError(
                f"ConfigureBuffer failed for buffer_id={self._buffer_id!r}: "
                f"{reply.get('error')}"
            )

        # Grant every slot upfront so the subprocess can start filling them.
        for grant in self._coordinator.grant_initial_slots():
            self._endpoint.send(
                SlotGranted(
                    slot=grant["slot"],
                    item_id=grant["item_id"],
                    buffer_id=self._buffer_id,
                )
            )

    def stop(self) -> None:
        self._cleanup.clear()
        super().stop()
        # Buffer is NOT closed here — its lifetime spans subprocess restarts.
        # Register buffer.close() / buffer.unlink() in ctx.lifecycle so it is
        # cleaned up only when the application shuts down.

    def ack_slot(self, slot: int, item_id: int, consumer_id: str) -> None:
        """
        Called by in-process consumers when they are done reading a slot.
        Drives the coordinator and re-grants the slot to the subprocess if
        all consumers have acked.
        """
        self._do_ack(slot=slot, item_id=item_id, consumer_id=consumer_id)

    # ------------------------------------------------------------------
    # Internal coordinator handlers
    # ------------------------------------------------------------------

    def _on_item_written(self, msg: ItemWritten) -> None:
        if msg.buffer_id != self._buffer_id:
            return
        notifications = self._coordinator.on_item_written(
            slot=msg.slot, item_id=msg.item_id
        )
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
            self._endpoint.send(
                SlotGranted(
                    slot=result.grant["slot"],
                    item_id=result.grant["item_id"],
                    buffer_id=self._buffer_id,
                )
            )
