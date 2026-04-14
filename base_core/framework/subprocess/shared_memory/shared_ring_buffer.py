from __future__ import annotations

from dataclasses import dataclass, asdict
from multiprocessing import shared_memory
from typing import Any
import json
import struct

import numpy as np

from base_core.framework.subprocess.shared_memory.models import _SLOT_HEADER_STRUCT, SharedRingBufferSpec, SlotHeader


# ---------------------------------------------------------------------
# Shared ring buffer
# ---------------------------------------------------------------------
class SharedRingBuffer:
    def __init__(
        self,
        *,
        spec: SharedRingBufferSpec,
        shm: shared_memory.SharedMemory,
        owner: bool,
    ) -> None:
        self.spec = spec
        self._shm = shm
        self._owner = owner

        expected_size = self.spec.slot_count * self.spec.slot_size
        if shm.size < expected_size:
            raise ValueError(
                f"Shared memory too small: got {shm.size} bytes, "
                f"expected at least {expected_size} bytes."
            )

    # --------------------------------------------------------------
    # Creation / attach
    # --------------------------------------------------------------
    @classmethod
    def create(
        cls,
        *,
        name: str,
        slot_count: int,
        shape: tuple[int, ...],
        dtype: str | np.dtype,
    ) -> "SharedRingBuffer":
        spec = SharedRingBufferSpec.build(
            name=name,
            slot_count=slot_count,
            shape=shape,
            dtype=dtype,
        )
        shm = shared_memory.SharedMemory(
            name=spec.name,
            create=True,
            size=spec.slot_count * spec.slot_size,
        )
        buffer = cls(spec=spec, shm=shm, owner=True)
        buffer.zero_initialize()
        return buffer

    @classmethod
    def attach(cls, spec: SharedRingBufferSpec) -> "SharedRingBuffer":
        shm = shared_memory.SharedMemory(name=spec.name, create=False)
        return cls(spec=spec, shm=shm, owner=False)

    # --------------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def is_owner(self) -> bool:
        return self._owner

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        if not self._owner:
            raise RuntimeError("Only the owner should unlink the shared memory.")
        self._shm.unlink()

    def zero_initialize(self) -> None:
        self._shm.buf[:] = b"\x00" * len(self._shm.buf)

    # --------------------------------------------------------------
    # Internal addressing helpers
    # --------------------------------------------------------------
    def _validate_slot(self, slot: int) -> None:
        if not 0 <= slot < self.spec.slot_count:
            raise IndexError(
                f"Slot index out of range: {slot}. "
                f"Expected 0 <= slot < {self.spec.slot_count}."
            )

    def _slot_base(self, slot: int) -> int:
        self._validate_slot(slot)
        return slot * self.spec.slot_size

    def _header_range(self, slot: int) -> tuple[int, int]:
        start = self._slot_base(slot)
        end = start + self.spec.slot_header_size
        return start, end

    def _payload_range(self, slot: int) -> tuple[int, int]:
        start = self._slot_base(slot) + self.spec.slot_header_size
        end = start + self.spec.slot_payload_size
        return start, end

    # --------------------------------------------------------------
    # Header access
    # --------------------------------------------------------------
    def read_header(self, slot: int) -> SlotHeader:
        start, end = self._header_range(slot)
        raw = self._shm.buf[start:end]
        frame_id, timestamp_ns, payload_nbytes, _reserved = _SLOT_HEADER_STRUCT.unpack(raw)
        return SlotHeader(
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            payload_nbytes=payload_nbytes,
        )

    def write_header(self, slot: int, header: SlotHeader) -> None:
        if header.payload_nbytes > self.spec.slot_payload_size:
            raise ValueError(
                f"payload_nbytes={header.payload_nbytes} exceeds slot payload size "
                f"{self.spec.slot_payload_size}."
            )

        start, end = self._header_range(slot)
        packed = _SLOT_HEADER_STRUCT.pack(
            int(header.frame_id),
            int(header.timestamp_ns),
            int(header.payload_nbytes),
            0,  # reserved
        )
        self._shm.buf[start:end] = packed

    def clear_header(self, slot: int) -> None:
        self.write_header(slot, SlotHeader.empty())

    # --------------------------------------------------------------
    # Payload views
    # --------------------------------------------------------------
    def payload_view(self, slot: int) -> np.ndarray:
        start, end = self._payload_range(slot)
        return np.ndarray(
            shape=self.spec.shape,
            dtype=self.spec.np_dtype,
            buffer=self._shm.buf[start:end],
        )

    # --------------------------------------------------------------
    # High-level read / write
    # --------------------------------------------------------------
    def write_frame(
        self,
        *,
        slot: int,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ns: int,
    ) -> None:
        """
        Write one frame into the slot.

        Important:
        We write PAYLOAD first and HEADER last.
        This matches your coordinator model well:
        readers should only touch the slot after the main process has received
        `frame_written` and then published `frame_available`.
        """
        if frame.shape != self.spec.shape:
            raise ValueError(
                f"Shape mismatch: got {frame.shape}, expected {self.spec.shape}."
            )

        if np.dtype(frame.dtype) != self.spec.np_dtype:
            raise ValueError(
                f"dtype mismatch: got {frame.dtype}, expected {self.spec.np_dtype}."
            )

        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        view = self.payload_view(slot)
        np.copyto(view, frame)

        self.write_header(
            slot,
            SlotHeader(
                frame_id=frame_id,
                timestamp_ns=timestamp_ns,
                payload_nbytes=self.spec.slot_payload_size,
            ),
        )

    def read_frame_copy(self, slot: int) -> tuple[SlotHeader, np.ndarray]:
        """
        Return header + a local copy of the frame.
        Useful if the consumer wants stable data for longer processing.
        """
        header = self.read_header(slot)
        frame = self.payload_view(slot).copy()
        return header, frame

    def read_frame_view(self, slot: int) -> tuple[SlotHeader, np.ndarray]:
        """
        Return header + zero-copy NumPy view into shared memory.
        Use this only if the slot is guaranteed not to be reused yet.
        """
        header = self.read_header(slot)
        view = self.payload_view(slot)
        return header, view