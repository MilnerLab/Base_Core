from __future__ import annotations

from dataclasses import dataclass

from messages import Message, Kind, MessageRegistry
from base_core.framework.subprocess.shared_memory.models import SharedRingBufferSpec
from base_core.framework.subprocess.worker_protocol import WorkerError, StartWorker, StopWorker


# =====================================================================
# Base shared-memory slot protocol
# =====================================================================
# All messages carry a `buffer_id` so a single subprocess can own more
# than one buffer (e.g. an analysis subprocess that reads from an input
# buffer AND writes to an output buffer).  The empty string "" is the
# default for subprocesses that have exactly one buffer.
# =====================================================================


# ----- main → subprocess (commands) -----

@dataclass(frozen=True)
class ConfigureBuffer(Message):
    """Attach to the shared buffer identified by buffer_id."""
    NAME = "configure_buffer"
    KIND = Kind.COMMAND
    spec: SharedRingBufferSpec
    buffer_id: str = ""


@dataclass(frozen=True)
class SlotGranted(Message):
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
    consumer_id: str
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

BASE_MESSAGES = SHARED_MEMORY_MESSAGES + (StartWorker, StopWorker, WorkerError)


def base_registry() -> MessageRegistry:
    """Fresh registry preloaded with all base protocol messages (shared-memory + worker lifecycle)."""
    return MessageRegistry().register(*BASE_MESSAGES)
