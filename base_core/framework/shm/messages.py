from __future__ import annotations

from dataclasses import dataclass

from base_core.ipc.codec import register
from base_core.ipc.message import Message, OKReply, Request
from base_core.framework.shm.spec import MemorySpec


@register
@dataclass(frozen=True)
class AttachBuffer(Request[OKReply]):
    """Service → subprocess: attach to this shared memory buffer.
    Subprocess replies OKReply on success, ErrorReply on failure."""
    buffer_class_name: str = ""
    spec: MemorySpec = None  # type: ignore[assignment]


@register
@dataclass(frozen=True)
class SlotGrant(Message):
    """Service → subprocess: this slot is available for writing."""
    buffer_class_name: str = ""
    slot: int = 0


@register
@dataclass(frozen=True)
class ItemAvailable(Message):
    """Subprocess → service: data has been written to this slot."""
    buffer_class_name: str = ""
    slot: int = 0
    item_id: int = 0
    timestamp_ns: int = 0
