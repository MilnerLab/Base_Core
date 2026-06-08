from __future__ import annotations

import collections
import threading
import time
from typing import Generic, TypeVar

from base_core.framework.subprocess.shared_memory.base_protocol import ConfigureBuffer, ItemWritten, SlotGranted
from json_process_base import JsonlStdioAppBase
from messages import MessageRegistry
from base_core.framework.subprocess.shared_memory.models import (
    ItemDescriptor,
    SharedRingBufferSpec,
)


TBuffer = TypeVar("TBuffer")
TMeasurement = TypeVar("TMeasurement")


class GrantedSlotWriterProcessBase(
    JsonlStdioAppBase,
    Generic[TBuffer, TMeasurement],
):
    """
    Base class for subprocess writers that receive writable slots from the
    main process and write measurements into shared memory.

    Speaks the shared-memory slot lifecycle (ConfigureBuffer / SlotGranted /
    ItemWritten). Subclasses provide a registry (base_registry().extend(...))
    and implement attach_buffer / acquire_measurement / write_measurement_to_slot.
    """

    def __init__(self, registry: MessageRegistry, *, source: str) -> None:
        super().__init__(registry, source=source)

        self._buffer: TBuffer | None = None
        self._buffer_spec: SharedRingBufferSpec | None = None
        self._configured = threading.Event()

        # Grants consumed FIFO. Order is preserved even across acquire failures
        # so item_id (assigned by the coordinator at grant time) stays monotonic
        # in publish order.
        self._grants_lock = threading.Lock()
        self._granted_slots: collections.deque[ItemDescriptor] = collections.deque()
        self._grants_available = threading.Event()
        self._live_slots: set[int] = set()

        self.on(ConfigureBuffer, self._handle_configure)
        self.on(SlotGranted, self._handle_slot_granted)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def buffer(self) -> TBuffer:
        if self._buffer is None:
            raise RuntimeError("Shared buffer is not configured yet.")
        return self._buffer

    @property
    def buffer_spec(self) -> SharedRingBufferSpec:
        if self._buffer_spec is None:
            raise RuntimeError("Shared buffer spec is not configured yet.")
        return self._buffer_spec

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    def attach_buffer(self, spec: SharedRingBufferSpec) -> TBuffer:
        """Attach the domain-specific shared buffer, e.g. SharedSpectrumBuffer.attach(spec)."""
        raise NotImplementedError

    def acquire_measurement(self, stop_event: threading.Event) -> TMeasurement | None:
        """Acquire one measurement, or None if no data is ready yet."""
        raise NotImplementedError

    def write_measurement_to_slot(
        self, *, measurement: TMeasurement, item: ItemDescriptor
    ) -> None:
        """Write one acquired measurement into the granted slot."""
        raise NotImplementedError

    def on_buffer_configured(self) -> None:
        """Optional hook after the shared buffer is attached."""

    def on_slot_granted(self, item: ItemDescriptor) -> None:
        """Optional hook when a new writable slot arrives."""

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------
    def _handle_configure(self, msg: ConfigureBuffer, envelope: dict) -> None:
        self._buffer_spec = msg.spec
        self._buffer = self.attach_buffer(msg.spec)
        self._configured.set()
        self.on_buffer_configured()
        self.reply_ok(envelope)

    def _handle_slot_granted(self, msg: SlotGranted, envelope: dict) -> None:
        item = ItemDescriptor(slot=msg.slot, item_id=msg.item_id, timestamp_ns=0)
        with self._grants_lock:
            if msg.slot in self._live_slots:
                raise RuntimeError(f"Slot {msg.slot} granted while still live.")
            self._live_slots.add(msg.slot)
            self._granted_slots.append(item)
            self._grants_available.set()
        self.on_slot_granted(item)
        self.reply_ok(envelope)

    # ------------------------------------------------------------------
    # Grant queue helpers
    # ------------------------------------------------------------------
    def _next_grant(self, timeout: float) -> ItemDescriptor | None:
        if not self._grants_available.wait(timeout=timeout):
            return None
        with self._grants_lock:
            if not self._granted_slots:
                self._grants_available.clear()
                return None
            return self._granted_slots[0]

    def _retire_grant(self, slot: int) -> None:
        with self._grants_lock:
            if self._granted_slots and self._granted_slots[0].slot == slot:
                self._granted_slots.popleft()
            self._live_slots.discard(slot)
            if not self._granted_slots:
                self._grants_available.clear()

    # ------------------------------------------------------------------
    # Main device loop
    # ------------------------------------------------------------------
    def main(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            if not self._configured.wait(timeout=0.05):
                continue

            granted = self._next_grant(timeout=0.05)
            if granted is None:
                continue
            if stop_event.is_set():
                break

            measurement = self.acquire_measurement(stop_event)
            if measurement is None:
                # Leave the grant at the head; retry next iteration. Order
                # (and thus item_id) is preserved.
                time.sleep(0.001)
                continue

            item = ItemDescriptor(
                slot=granted.slot,
                item_id=granted.item_id,
                timestamp_ns=time.time_ns(),
            )
            self.write_measurement_to_slot(measurement=measurement, item=item)
            self._retire_grant(item.slot)

            self.emit(
                ItemWritten(
                    slot=item.slot,
                    item_id=item.item_id,
                    timestamp_ns=item.timestamp_ns,
                )
            )