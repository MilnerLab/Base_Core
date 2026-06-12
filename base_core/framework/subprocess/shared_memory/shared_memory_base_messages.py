from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from base_core.framework.subprocess.messages import (
    ErrorMessage,
    Kind,
    Message,
    MessageRegistry,
    OKMessage,
    RequestMessage,
)
from base_core.framework.subprocess.shared_memory.models import SharedRingBufferSpec
from base_core.framework.subprocess.worker_protocol import WorkerError, StartWorker, StopWorker


# =====================================================================
# Base shared-memory slot protocol
# =====================================================================

# ----- main → subprocess (commands) -----

@dataclass(frozen=True)
class ConfigureBuffer(RequestMessage["OKMessage"]):
    """Attach to the shared buffer identified by buffer_id."""
    NAME = "configure_buffer"
    KIND = Kind.COMMAND
    spec: SharedRingBufferSpec
    buffer_id: str = ""
    buffer_type: Literal["read", "write"] = "read"


@dataclass(frozen=True)
class SlotGranted(RequestMessage["OKMessage"]):
    """You may write this slot in the output buffer."""
    NAME = "slot_granted"
    KIND = Kind.COMMAND
    slot: int
    item_id: int
    buffer_id: str = ""


@dataclass(frozen=True)
class ItemAvailable(Message):
    """A slot in an input buffer is ready for you to read."""
    NAME = "item_available"
    KIND = Kind.COMMAND
    slot: int
    item_id: int
    timestamp_ns: int
    buffer_id: str = ""


# ----- subprocess → main (events) -----

@dataclass(frozen=True)
class ItemWritten(Message):
    """Slot has been written; main process may publish it to consumers."""
    NAME = "item_written"
    KIND = Kind.EVENT
    slot: int
    item_id: int
    timestamp_ns: int
    buffer_id: str = ""


@dataclass(frozen=True)
class ItemAck(Message):
    """Consumer is done reading this slot."""
    NAME = "item_ack"
    KIND = Kind.EVENT
    slot: int
    item_id: int
    consumer_id: str
    buffer_id: str = ""


# ---- registry helpers ----

SHARED_MEMORY_MESSAGES = (
    ConfigureBuffer,
    SlotGranted,
    ItemAvailable,
    ItemWritten,
    ItemAck,
)

BASE_MESSAGES = SHARED_MEMORY_MESSAGES + (StartWorker, StopWorker, WorkerError, OKMessage, ErrorMessage)


def base_registry() -> MessageRegistry:
    """Fresh registry with all base protocol messages plus typed reply classes."""
    return MessageRegistry().register(*BASE_MESSAGES)
