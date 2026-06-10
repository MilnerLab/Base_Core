from __future__ import annotations

import dataclasses
from concurrent.futures import Future
from dataclasses import dataclass, field
from mailbox import Message
from typing import TYPE_CHECKING, Callable, Optional

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

if TYPE_CHECKING:
    from base_core.framework.subprocess.subprocess_service import SubprocessService


# ---------------------------------------------------------------------------
# Internal buffer config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class _InputConfig:
    buffer_id: str
    upstream_coordinator: SharedBufferCoordinator
    upstream_buffer: SharedRingBuffer
    registered: bool = field(default=False, compare=False)


@dataclass
class _OutputConfig:
    buffer: SharedRingBuffer
    coordinator: SharedBufferCoordinator


# ---------------------------------------------------------------------------
# WorkerHandle
# ---------------------------------------------------------------------------

class WorkerHandle:
    """
    Main-process proxy for a named worker within a SubprocessService.

    Can be used as-is for a CommandWorker (no buffer setup), or composed with
    .with_input() / .with_output() for workers that use shared-memory buffers:

        # ProducerWorker
        WorkerHandle(svc, "spectrometer", bus=bus).with_output(buf, coord)

        # ConsumerWorker
        WorkerHandle(svc, "consumer", bus=bus).with_input("spectrometer", coord, buf)

        # ProcessorWorker
        WorkerHandle(svc, "processor", bus=bus)
            .with_input("spectrometer", spec_coord, spec_buf)
            .with_input("camera", cam_coord, cam_buf)
            .with_output(out_buf, out_coord)

    All variants call svc.worker("name").start_async() — uniform interface.
    Obtain via SubprocessService.worker("name") or build directly and register
    via SubprocessService._register_handle("name", handle).
    """

    def __init__(
        self,
        service: SubprocessService,
        name: str,
        *,
        bus: Optional[EventBus] = None,
    ) -> None:
        self._service = service
        self._name = name
        self._bus = bus
        self._inputs: list[_InputConfig] = []
        self._output: Optional[_OutputConfig] = None
        self._cleanup = CleanupCollection()

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Fluent builder
    # ------------------------------------------------------------------

    def with_input(
        self,
        buffer_id: str,
        upstream_coordinator: SharedBufferCoordinator,
        upstream_buffer: SharedRingBuffer,
    ) -> WorkerHandle:
        """Register an input buffer channel. Returns self for chaining."""
        if self._bus is None:
            raise RuntimeError("WorkerHandle needs bus= to use with_input().")
        self._inputs.append(_InputConfig(
            buffer_id=buffer_id,
            upstream_coordinator=upstream_coordinator,
            upstream_buffer=upstream_buffer,
        ))
        return self

    def with_output(
        self,
        buffer: SharedRingBuffer,
        coordinator: SharedBufferCoordinator,
    ) -> WorkerHandle:
        """Register an output buffer. Returns self for chaining."""
        if self._bus is None:
            raise RuntimeError("WorkerHandle needs bus= to use with_output().")
        self._output = _OutputConfig(buffer=buffer, coordinator=coordinator)
        return self

    # ------------------------------------------------------------------
    # Worker thread lifecycle
    # ------------------------------------------------------------------

    def start_async(
        self,
        *,
        key: Optional[str] = None,
        on_success: Optional[Callable] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> Future:
        """Configure buffers (if any) and start the worker thread — off the main thread."""
        return self._service.run_async(
            self._do_start,
            key=key or f"worker.{self._name}.start",
            on_success=on_success,
            on_error=on_error,
        )

    def stop(self) -> None:
        """Stop the worker thread and clear any bus subscriptions. Blocks until confirmed."""
        reply = self._service.request_sync(StopWorker(worker_name=self._name))
        if not reply.get("ok"):
            raise RuntimeError(
                f"StopWorker failed for {self._name!r}: {reply.get('error')}"
            )
        self._cleanup.clear()

    def restart_async(
        self,
        *,
        key: Optional[str] = None,
        on_success: Optional[Callable] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> Future:
        """Stop (blocking) then re-start asynchronously."""
        self.stop()
        return self.start_async(key=key, on_success=on_success, on_error=on_error)

    # ------------------------------------------------------------------
    # Slot acknowledgement (output-buffer workers only)
    # ------------------------------------------------------------------

    @property
    def output_buffer(self) -> Optional[SharedRingBuffer]:
        return self._output.buffer if self._output else None

    def ack_slot(self, slot: int, item_id: int, consumer_id: str) -> None:
        """Called by in-process consumers when done reading an output slot."""
        if self._output is None:
            raise RuntimeError(
                f"Worker {self._name!r} has no output buffer configured."
            )
        self._do_ack(slot=slot, item_id=item_id, consumer_id=consumer_id)

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

    # ------------------------------------------------------------------
    # Internal — start sequence (runs on TaskRunner pool thread)
    # ------------------------------------------------------------------

    def _do_start(self) -> None:
        assert self._bus is not None or (not self._inputs and self._output is None)

        # 1. Register as consumer for each input (first call only)
        for cfg in self._inputs:
            if not cfg.registered:
                cfg.upstream_coordinator.register_consumer(self._name)
                cfg.registered = True

        # 2. Configure each input buffer in the worker + subscribe ItemAvailable forwarding
        for cfg in self._inputs:
            reply = self._service.request_sync(
                self._stamp(ConfigureBuffer(spec=cfg.upstream_buffer.spec, buffer_id=cfg.buffer_id))
            )
            if not reply.get("ok"):
                raise RuntimeError(
                    f"ConfigureBuffer (input, buffer_id={cfg.buffer_id!r}) "
                    f"failed for worker {self._name!r}: {reply.get('error')}"
                )
            buffer_id = cfg.buffer_id
            self._cleanup.add(
                self._bus.subscribe(ItemAvailable, self._make_item_available_forwarder(buffer_id))
            )

        # 3. Configure output buffer if present
        if self._output is not None:
            out = self._output
            out.coordinator.reset()
            reply = self._service.request_sync(
                self._stamp(ConfigureBuffer(spec=out.buffer.spec, buffer_id=self._name))
            )
            if not reply.get("ok"):
                raise RuntimeError(
                    f"ConfigureBuffer (output) failed for worker {self._name!r}: {reply.get('error')}"
                )
            for grant in out.coordinator.grant_initial_slots():
                self._service.send(
                    self._stamp(SlotGranted(
                        slot=grant["slot"],
                        item_id=grant["item_id"],
                        buffer_id=self._name,
                    ))
                )
            self._cleanup.add(self._bus.subscribe(ItemWritten, self._on_item_written))
            self._cleanup.add(self._bus.subscribe(ItemAck, self._on_item_ack))

        # 4. Start the worker thread
        reply = self._service.request_sync(StartWorker(worker_name=self._name))
        if not reply.get("ok"):
            raise RuntimeError(
                f"StartWorker failed for {self._name!r}: {reply.get('error')}"
            )

    # ------------------------------------------------------------------
    # Internal — ItemAvailable forwarding (one closure per input channel)
    # ------------------------------------------------------------------

    def _make_item_available_forwarder(self, buffer_id: str) -> Callable[[ItemAvailable], None]:
        def forward(msg: ItemAvailable) -> None:
            if msg.consumer_id != self._name or msg.buffer_id != buffer_id:
                return
            self._service.send(self._stamp(msg))
        return forward

    # ------------------------------------------------------------------
    # Internal — output coordinator handlers
    # ------------------------------------------------------------------

    def _on_item_written(self, msg: ItemWritten) -> None:
        if msg.buffer_id != self._name:
            return
        assert self._output is not None
        notifications = self._output.coordinator.on_item_written(
            slot=msg.slot, item_id=msg.item_id
        )
        for n in notifications:
            self._bus.publish(
                ItemAvailable(
                    consumer_id=n["consumer_id"],
                    slot=n["slot"],
                    item_id=n["item_id"],
                    timestamp_ns=msg.timestamp_ns,
                    buffer_id=self._name,
                )
            )

    def _on_item_ack(self, msg: ItemAck) -> None:
        if msg.buffer_id != self._name:
            return
        self._do_ack(slot=msg.slot, item_id=msg.item_id, consumer_id=msg.consumer_id)

    def _do_ack(self, *, slot: int, item_id: int, consumer_id: str) -> None:
        assert self._output is not None
        result = self._output.coordinator.on_item_ack(
            slot=slot, item_id=item_id, consumer_id=consumer_id
        )
        if result.outcome == "completed" and result.grant is not None:
            self._service.send(
                self._stamp(SlotGranted(
                    slot=result.grant["slot"],
                    item_id=result.grant["item_id"],
                    buffer_id=self._name,
                ))
            )

    def _stamp(self, cmd: Message) -> Message:
        """Return a copy of cmd with target set to this worker's name."""
        return dataclasses.replace(cmd, target=self._name)

