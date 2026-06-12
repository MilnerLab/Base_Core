from __future__ import annotations

import logging
from multiprocessing.shared_memory import SharedMemory
from typing import TypeVar

import numpy as np

from base_core.framework.shm.spec import MemorySpec

log = logging.getLogger(__name__)

TBuffer = TypeVar("TBuffer", bound="SharedMemoryBuffer")


class SharedMemoryBuffer:
    """
    Generic multi-slot shared memory buffer.

    Memory layout: [slot_0][slot_1]...[slot_N-1]
    Each slot is spec.slot_nbytes bytes, viewed as np.ndarray(spec.shape, spec.dtype).

    Lifecycle:
        Main process (owner):  buf = SharedMemoryBuffer.create(spec)
                               ...
                               buf.unlink(); buf.close()

        Subprocess (reader):   buf = SharedMemoryBuffer.attach(spec)
                               ...
                               buf.close()

    Subclasses add typed read/write helpers (e.g. SpectrumBuffer.wavelengths()).
    """

    def __init__(self, shm: SharedMemory, spec: MemorySpec) -> None:
        self._shm = shm
        self._spec = spec
        self._closed = False

    @classmethod
    def create(cls: type[TBuffer], spec: MemorySpec) -> TBuffer:
        """Allocate a new named shared memory region and return a buffer wrapping it."""
        shm = SharedMemory(name=spec.name, create=True, size=spec.nbytes)
        log.debug("Created shared memory %r (%d bytes)", spec.name, spec.nbytes)
        return cls(shm, spec)

    @classmethod
    def attach(cls: type[TBuffer], spec: MemorySpec) -> TBuffer:
        """Attach to an existing shared memory region."""
        shm = SharedMemory(name=spec.name, create=False, size=spec.nbytes)
        log.debug("Attached shared memory %r", spec.name)
        return cls(shm, spec)

    @property
    def spec(self) -> MemorySpec:
        return self._spec

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def write_slot(self, slot: int, data: np.ndarray) -> None:
        """Copy data into slot. data must match spec.shape and spec.dtype."""
        view = self.read_slot_view(slot)
        np.copyto(view, data.astype(np.dtype(self._spec.dtype), copy=False))

    def read_slot_copy(self, slot: int) -> np.ndarray:
        """Return an independent copy of the slot's array."""
        return self.read_slot_view(slot).copy()

    def read_slot_view(self, slot: int) -> np.ndarray:
        """
        Return a zero-copy view into the shared memory for the given slot.
        The view is valid only while this buffer is open. Do not hold it across close().
        """
        offset = slot * self._spec.slot_nbytes
        return np.ndarray(
            self._spec.shape,
            dtype=np.dtype(self._spec.dtype),
            buffer=self._shm.buf,
            offset=offset,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Detach from the shared memory. Safe to call multiple times."""
        if not self._closed:
            self._shm.close()
            self._closed = True

    def unlink(self) -> None:
        """Destroy the shared memory segment. Only the creator should call this."""
        self._shm.unlink()

    def __del__(self) -> None:
        if not self._closed:
            try:
                self._shm.close()
            except Exception:
                pass
