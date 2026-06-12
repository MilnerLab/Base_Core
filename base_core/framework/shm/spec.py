import dataclasses
import numpy as np

from base_core.framework.serialization.serde import Primitive, PrimitiveSerde


@dataclasses.dataclass(frozen=True)
class MemorySpec(PrimitiveSerde):
    """
    Describes one shared memory region: how many slots it has and what each slot looks like.

    Implements PrimitiveSerde so it can be embedded in IPC codec messages (e.g. AttachBuffer).
    The subprocess uses the spec to attach to existing shared memory without needing to know
    which Python class owns it.
    """

    name: str               # OS-level shared memory name
    slot_count: int         # number of independent slots
    shape: tuple[int, ...]  # numpy array shape of one slot
    dtype: str              # numpy dtype string, e.g. "float64", "uint16"

    @property
    def slot_nbytes(self) -> int:
        return int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize

    @property
    def nbytes(self) -> int:
        return self.slot_count * self.slot_nbytes

    def to_primitive(self) -> Primitive:
        return {
            "name": self.name,
            "slot_count": self.slot_count,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }

    @classmethod
    def from_primitive(cls, v: Primitive) -> "MemorySpec":
        return cls(
            name=v["name"],
            slot_count=v["slot_count"],
            shape=tuple(v["shape"]),
            dtype=v["dtype"],
        )
