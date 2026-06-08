from __future__ import annotations

from dataclasses import dataclass

from messages import Message, Kind, MessageRegistry
from base_core.framework.subprocess.shared_memory.models import SharedRingBufferSpec


# =====================================================================
# Base protocol
# =====================================================================
# These messages are device-independent. Two tiers of subprocess use them:
#
#   - A plain JSON subprocess uses only the lifecycle/control messages it
#     needs (it may use none of the shared-memory ones).
#   - A shared-memory writer subprocess uses the slot-lifecycle messages
#     (ConfigureBuffer / SlotGranted / ItemWritten / ItemAck).
#
# A concrete device builds its registry by extending the base:
#
#       from proto.base_protocol import base_registry
#       registry = base_registry().extend(SetGain, StartSweep)
#
# so the base lifecycle messages are shared while each device keeps its own
# messages isolated from every other device.
# =====================================================================


# ----- shared-memory slot lifecycle -----

@dataclass(frozen=True)
class ConfigureBuffer(Message):
    """main -> subprocess (request): attach this shared buffer."""
    NAME = "configure_buffer"
    KIND = Kind.COMMAND
    spec: SharedRingBufferSpec


@dataclass(frozen=True)
class SlotGranted(Message):
    """main -> subprocess: you may write this slot now."""
    NAME = "slot_granted"
    KIND = Kind.COMMAND
    slot: int
    item_id: int


@dataclass(frozen=True)
class ItemWritten(Message):
    """subprocess -> main (event): slot has been written, publish it."""
    NAME = "item_written"
    KIND = Kind.EVENT
    slot: int
    item_id: int
    timestamp_ns: int


@dataclass(frozen=True)
class ItemAck(Message):
    """consumer -> main (event): this item is done being read."""
    NAME = "item_ack"
    KIND = Kind.EVENT
    slot: int
    item_id: int
    consumer_id: str


# The base message classes, grouped so a device can pull in just what it needs.
SHARED_MEMORY_MESSAGES = (ConfigureBuffer, SlotGranted, ItemWritten, ItemAck)
BASE_MESSAGES = SHARED_MEMORY_MESSAGES


def base_registry() -> MessageRegistry:
    """A fresh registry preloaded with the base protocol messages."""
    return MessageRegistry().register(*BASE_MESSAGES)