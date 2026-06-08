from __future__ import annotations

import functools
from dataclasses import dataclass, field, fields, MISSING
from typing import Any, ClassVar, Optional, Type, TypeVar, get_type_hints

from serialization.serde import Primitive, PrimitiveSerde


# ---------------------------------------------------------------------
# Message envelope
# ---------------------------------------------------------------------
# A Message is the typed form of one protocol payload. It is NOT the whole
# wire envelope -- kind / id / reply_to / source are transport concerns and
# are added/stripped by the endpoint. A Message only owns:
#   - its stable wire NAME (class-level constant)
#   - its DIRECTION-agnostic payload (its own fields)
#
# Subclasses declare fields as a frozen dataclass. Field types may be plain
# primitives or anything implementing PrimitiveSerde.
# ---------------------------------------------------------------------

TMsg = TypeVar("TMsg", bound="Message")


class Kind:
    COMMAND = "command"
    REPLY = "reply"
    EVENT = "event"


@dataclass(frozen=True)
class Message:
    # Each concrete message sets these two class-vars.
    NAME: ClassVar[str] = ""
    KIND: ClassVar[str] = ""  # one of Kind.*

    # Envelope-level metadata, NOT part of the protocol payload. Populated by
    # the registry on decode from the wire envelope's "source" field. Excluded
    # from equality so two messages with identical payloads compare equal
    # regardless of which device they came from.
    source: Optional[str] = field(default=None, compare=False, kw_only=True)

    def _payload_fields(self):
        # All dataclass fields except the envelope-level `source`.
        return [f for f in fields(self) if f.name != "source"]

    def to_payload(self) -> dict[str, Primitive]:
        out: dict[str, Primitive] = {}
        for f in self._payload_fields():
            out[f.name] = _encode_value(getattr(self, f.name))
        return out

    @classmethod
    def from_payload(cls: Type[TMsg], payload: dict[str, Primitive]) -> TMsg:
        hints = _resolved_hints(cls)
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name == "source":
                continue  # comes from the envelope, not the payload
            if f.name not in payload:
                if f.default is MISSING and f.default_factory is MISSING:  # type: ignore[misc]
                    raise ValueError(
                        f"{cls.__name__}: missing required field '{f.name}'"
                    )
                continue
            kwargs[f.name] = _decode_value(hints.get(f.name, f.type), payload[f.name])
        return cls(**kwargs)


@functools.lru_cache(maxsize=None)
def _resolved_hints(cls: type) -> dict[str, Any]:
    # Resolves string annotations (from `from __future__ import annotations`)
    # back to real types. Cached because get_type_hints is not cheap.
    return get_type_hints(cls)


def _encode_value(value: Any) -> Primitive:
    if isinstance(value, PrimitiveSerde):
        return value.to_primitive()
    if isinstance(value, (list, tuple)):
        return [_encode_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode_value(v) for k, v in value.items()}
    return value


def _decode_value(declared_type: Any, value: Primitive) -> Any:
    # If the declared field type is a PrimitiveSerde subclass, delegate to it.
    if isinstance(declared_type, type) and issubclass(declared_type, PrimitiveSerde):
        return declared_type.from_primitive(value)
    return value


# ---------------------------------------------------------------------
# Registry: wire (kind, name) -> Message class
# ---------------------------------------------------------------------
class MessageRegistry:
    """
    Maps (kind, name) pairs to Message classes for decoding inbound wire dicts.

    A device builds its own registry by extending the base one:

        registry = base_registry().extend(SetGain, StartSweep)

    so every device shares the base lifecycle messages but keeps its own
    device-specific messages isolated from other devices.
    """

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], Type[Message]] = {}

    def register(self, *msg_classes: Type[Message]) -> "MessageRegistry":
        """
        Register one or more Message classes. Returns self so calls chain.
        Also usable as a decorator on a single class.
        """
        for msg_cls in msg_classes:
            if not msg_cls.NAME or not msg_cls.KIND:
                raise ValueError(f"{msg_cls.__name__} must set NAME and KIND.")
            key = (msg_cls.KIND, msg_cls.NAME)
            existing = self._by_key.get(key)
            if existing is not None and existing is not msg_cls:
                raise ValueError(
                    f"Duplicate registration for {key}: "
                    f"{msg_cls.__name__} vs existing {existing.__name__}"
                )
            self._by_key[key] = msg_cls
        # When used as a single-class decorator, return the class, not self.
        if len(msg_classes) == 1:
            return msg_classes[0]  # type: ignore[return-value]
        return self

    def extend(self, *msg_classes: Type[Message]) -> "MessageRegistry":
        """Return a NEW registry containing everything here plus msg_classes."""
        child = MessageRegistry()
        child._by_key = dict(self._by_key)
        child.register(*msg_classes)
        return child

    def decode(self, envelope: dict[str, Any]) -> Optional[Message]:
        """
        Turn a validated wire dict into a typed Message, or None if unknown.
        Transport fields (id/reply_to/source/ok) are left to the endpoint.
        """
        kind = envelope.get("kind")
        name = envelope.get("name")
        if not isinstance(kind, str) or not isinstance(name, str):
            return None
        msg_cls = self._by_key.get((kind, name))
        if msg_cls is None:
            return None
        payload = envelope.get("payload") or {}
        if not isinstance(payload, dict):
            return None
        msg = msg_cls.from_payload(payload)
        src = envelope.get("source")
        if isinstance(src, str):
            object.__setattr__(msg, "source", src)  # frozen dataclass
        return msg

    def envelope_for(self, msg: Message) -> dict[str, Any]:
        """Turn a typed Message into the wire envelope (sans id/reply_to)."""
        return {
            "kind": msg.KIND,
            "name": msg.NAME,
            "payload": msg.to_payload(),
        }