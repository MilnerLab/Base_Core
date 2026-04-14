from __future__ import annotations

import queue
import threading
import time
from typing import Generic, TypeVar, Any

from base_core.framework.subprocess.json_process_base import JsonlStdioAppBase
from base_core.framework.subprocess.shared_memory.models import (
    FrameDescriptor,
    SharedRingBufferSpec,
)


TBuffer = TypeVar("TBuffer")
TMeasurement = TypeVar("TMeasurement")


class GrantedSlotWriterProcessBase(
    JsonlStdioAppBase,
    Generic[TBuffer, TMeasurement],
):
    """
    Base class for subprocess writers that receive writable slots from the main process.

    Expected protocol on subclasses:
        PROTOCOL.CMD_CONFIGURE_BUFFER
        PROTOCOL.CMD_SLOT_GRANTED
        PROTOCOL.EVT_FRAME_WRITTEN
    """

    PROTOCOL = None

    def __init__(self, *, source: str) -> None:
        super().__init__(source=source)

        self._buffer: TBuffer | None = None
        self._buffer_spec: SharedRingBufferSpec | None = None

        self._configured = threading.Event()
        self._granted_slots: queue.Queue[FrameDescriptor] = queue.Queue()

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
        """
        Create/attach the domain-specific shared buffer from spec.
        Example:
            return SharedSpectrumBuffer.attach(spec)
        """
        raise NotImplementedError

    def acquire_measurement(
        self,
        stop_event: threading.Event,
    ) -> TMeasurement | None:
        """
        Acquire one measurement from the device.

        Return:
            - measurement object if data is available
            - None if no data is ready yet and the loop should continue
        """
        raise NotImplementedError

    def write_measurement_to_slot(
        self,
        *,
        measurement: TMeasurement,
        frame: FrameDescriptor,
    ) -> None:
        """
        Write one acquired measurement into the granted slot.
        """
        raise NotImplementedError

    def on_buffer_configured(self) -> None:
        """
        Optional hook for subclasses.
        Called after the shared buffer has been attached.
        """
        pass

    def on_slot_granted(self, frame: FrameDescriptor) -> None:
        """
        Optional hook for subclasses.
        Called when a new writable slot arrives from the main process.
        """
        pass

    # ------------------------------------------------------------------
    # Command handling
    # ------------------------------------------------------------------
    def on_command(
        self,
        name: str,
        payload: dict[str, Any],
        message: dict[str, Any],
    ) -> None:
        protocol = self.PROTOCOL
        if protocol is None:
            self.reply_error(message, "Subclass must define PROTOCOL.")
            return

        if name == protocol.CMD_CONFIGURE_BUFFER:
            try:
                spec = SharedRingBufferSpec.from_dict(payload["spec"])
                self._buffer_spec = spec
                self._buffer = self.attach_buffer(spec)
                self._configured.set()
                self.on_buffer_configured()
                self.reply_ok(message)
            except Exception as exc:  # noqa: BLE001
                self.reply_error(message, str(exc))
            return

        if name == protocol.CMD_SLOT_GRANTED:
            try:
                frame = FrameDescriptor(
                    slot=int(payload["slot"]),
                    frame_id=int(payload["frame_id"]),
                    timestamp_ns=0,  # set when the actual write happens
                )
                self._granted_slots.put(frame)
                self.on_slot_granted(frame)
                self.reply_ok(message)
            except Exception as exc:  # noqa: BLE001
                self.reply_error(message, str(exc))
            return

        self.reply_error(message, f"Unknown command: {name}")

    # ------------------------------------------------------------------
    # Main device loop
    # ------------------------------------------------------------------
    def main(self, stop_event: threading.Event) -> None:
        protocol = self.PROTOCOL
        if protocol is None:
            raise RuntimeError("Subclass must define PROTOCOL.")

        while not stop_event.is_set():
            if not self._configured.wait(timeout=0.05):
                continue

            try:
                granted = self._granted_slots.get(timeout=0.05)
            except queue.Empty:
                continue

            if stop_event.is_set():
                break

            measurement = self.acquire_measurement(stop_event)
            if measurement is None:
                # No data yet. Put the slot back so we do not lose the grant.
                self._granted_slots.put(granted)
                time.sleep(0.001)
                continue

            timestamp_ns = time.time_ns()
            frame = FrameDescriptor(
                slot=granted.slot,
                frame_id=granted.frame_id,
                timestamp_ns=timestamp_ns,
            )

            self.write_measurement_to_slot(
                measurement=measurement,
                frame=frame,
            )

            self.emit_event(
                protocol.EVT_FRAME_WRITTEN,
                {
                    "slot": frame.slot,
                    "frame_id": frame.frame_id,
                    "timestamp_ns": frame.timestamp_ns,
                },
            )