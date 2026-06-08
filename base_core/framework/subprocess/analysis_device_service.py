from __future__ import annotations

from typing import Optional

from base_core.framework.concurrency.task_runner import TaskRunner
from base_core.framework.events.event_bus import EventBus
from base_core.framework.subprocess.json_endpoint import JsonlSubprocessEndpoint
from base_core.framework.subprocess.shared_memory.base_protocol import (
    ConfigureBuffer,
    ItemAvailable,
)
from base_core.framework.subprocess.shared_memory.models import SharedRingBufferSpec
from base_core.framework.subprocess.shared_memory.shared_buffer_coordinator import (
    SharedBufferCoordinator,
)
from base_core.framework.subprocess.shared_memory.shared_ring_buffer import SharedRingBuffer
from base_core.framework.subprocess.shared_memory_device_service import SharedMemoryDeviceService


class AnalysisDeviceService(SharedMemoryDeviceService):
    """
    DeviceService for a subprocess that:
      - Reads from an upstream shared memory buffer (consumer role), and
      - Writes results into its own output shared memory buffer (producer role).

    On top of SharedMemoryDeviceService (which manages the output buffer), this
    service also:
      1. Sends a ConfigureBuffer for the input buffer so the subprocess can attach to it.
      2. Subscribes to ItemAvailable on the EventBus and forwards messages addressed
         to `consumer_id` to the subprocess endpoint.

    The subprocess receives ItemAvailable, reads the input slot (zero-copy view),
    computes a result, writes to its granted output slot, emits ItemWritten, then
    sends ItemAck for the input slot.

    Coordination:
      - The upstream service (e.g. SpectrometerService) must include this subprocess's
        consumer_id in its coordinator.  Call
        `c.get(UpstreamCoordinator).register_consumer(consumer_id)` during the
        analysis module's register() phase (before any on_startup()).
      - The output coordinator (passed to super().__init__) is independent and tracks
        downstream consumers of the analysis results.

    Parameters
    ----------
    input_spec : SharedRingBufferSpec
        Spec of the upstream (input) buffer.  The subprocess uses this to attach.
    input_buffer_id : str
        buffer_id used in the upstream service's protocol messages.
    consumer_id : str
        This subprocess's consumer_id in the upstream coordinator.  Only
        ItemAvailable messages with this consumer_id are forwarded.
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
        input_spec: SharedRingBufferSpec,
        input_buffer_id: str,
        consumer_id: str,
    ) -> None:
        super().__init__(
            io, endpoint, bus, buffer, coordinator,
            source=source, buffer_id=buffer_id,
        )
        self._input_spec = input_spec
        self._input_buffer_id = input_buffer_id
        self._consumer_id = consumer_id

    def start(self) -> None:
        super().start()  # ConfigureBuffer (output) + initial SlotGranted

        # Tell the subprocess about the input buffer so it can attach.
        reply = self._endpoint.raw_request(
            ConfigureBuffer(spec=self._input_spec, buffer_id=self._input_buffer_id)
        )
        if not reply.get("ok"):
            raise RuntimeError(
                f"ConfigureBuffer for input buffer {self._input_buffer_id!r} failed: "
                f"{reply.get('error')}"
            )

        # Forward ItemAvailable for our consumer_id to the subprocess.
        self._cleanup.add(
            self._bus.subscribe(ItemAvailable, self._forward_item_available)
        )

    def _forward_item_available(self, msg: ItemAvailable) -> None:
        if msg.consumer_id != self._consumer_id or msg.buffer_id != self._input_buffer_id:
            return
        self._endpoint.send(msg)
