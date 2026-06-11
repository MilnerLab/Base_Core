from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal
import json
import struct

from base_core.framework.serialization.serde import Primitive, PrimitiveSerde
import numpy as np



# ---------------------------------------------------------------------
# Slot header
# ---------------------------------------------------------------------
# Layout per slot:
#   uint64 item_id
#   uint64 timestamp_ns
#   uint32 payload_nbytes
#   uint32 reserved
#
# Keep this small and fixed-size.
# The coordinator-specific state (FREE / PUBLISHED / pending_consumers / etc.)
# lives in the main process, NOT in shared memory.
_SLOT_HEADER_STRUCT = struct.Struct("<Q Q I I")


@dataclass(frozen=True)
class ItemDescriptor:
    """Identifies a single item written into a particular slot."""
    slot: int = 0
    item_id: int = 0
    timestamp_ns: int = 0


@dataclass(frozen=True)
class SlotHeader:
    """The fixed-size header physically stored at the front of each slot."""
    item_id: int = 0
    timestamp_ns: int = 0
    payload_nbytes: int = 0

    @classmethod
    def empty(cls) -> "SlotHeader":
        return cls()


# ---------------------------------------------------------------------
# Shared memory metadata / startup info
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class SharedRingBufferSpec(PrimitiveSerde):
    name: str
    slot_count: int
    shape: tuple[int, ...]
    dtype: str
    slot_header_size: int
    slot_payload_size: int
    slot_size: int

    @property
    def np_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    @property
    def element_count(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    @property
    def payload_nbytes(self) -> int:
        return self.element_count * self.np_dtype.itemsize

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SharedRingBufferSpec":
        return cls(
            name=str(data["name"]),
            slot_count=int(data["slot_count"]),
            shape=tuple(int(x) for x in data["shape"]),
            dtype=str(data["dtype"]),
            slot_header_size=int(data["slot_header_size"]),
            slot_payload_size=int(data["slot_payload_size"]),
            slot_size=int(data["slot_size"]),
        )

    @classmethod
    def from_json(cls, raw: str) -> "SharedRingBufferSpec":
        return cls.from_dict(json.loads(raw))

    # PrimitiveSerde: spec travels over the wire inside ConfigureBuffer.
    def to_primitive(self) -> Primitive:
        return self.to_dict()

    @classmethod
    def from_primitive(cls, v: Primitive) -> "SharedRingBufferSpec":
        return cls.from_dict(v)

    @classmethod
    def build(
        cls,
        *,
        name: str,
        slot_count: int,
        shape: tuple[int, ...],
        dtype: str | np.dtype,
    ) -> "SharedRingBufferSpec":
        np_dtype = np.dtype(dtype)
        payload_nbytes = int(np.prod(shape)) * np_dtype.itemsize
        header_nbytes = _SLOT_HEADER_STRUCT.size
        slot_size = header_nbytes + payload_nbytes

        return cls(
            name=name,
            slot_count=slot_count,
            shape=shape,
            dtype=np_dtype.str,
            slot_header_size=header_nbytes,
            slot_payload_size=payload_nbytes,
            slot_size=slot_size,
        )


SlotState = Literal["FREE", "WRITING", "PUBLISHED"]


@dataclass
class SlotInfo:
    state: SlotState = "FREE"
    item_id: int | None = None
    pending_consumers: int = 0
    acked_mask: int = 0