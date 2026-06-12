from __future__ import annotations

import functools
from dataclasses import dataclass, field, fields, MISSING
from typing import Any, ClassVar, Generic, Optional, Type, TypeVar, get_type_hints

from base_core.framework.serialization.serde import Primitive, PrimitiveSerde


# ---------------------------------------------------------------------
# Message envelope
# ---------------------------------------------------------------------

TMsg = TypeVar("TMsg", bound="Message")
TReply = TypeVar("TReply", bound="Message")


class Kind:
    COMMAND = "command"
    REPLY = "reply"
    EVENT = "event"


@dataclass(frozen=True)
class Message:
    NAME: ClassVar[str] = ""
    KIND: ClassVar[str] = ""

    # Routing metadata — NOT part of the protocol payload.
    # Excluded from equality so two messages with identical payloads compare equal.
    source: Optional[str] = field(default=None, compare=False, kw_only=True)
    # consumer_id: which worker within a SubprocessApp should receive this message.
    # Replaces the old `target` field; set by WorkerHandle._stamp().
    consumer_id: Optional[str] = field(default=None, compare=False, kw_only=True)
    # request_id: the envelope `id` of the inbound command, populated by decode().
    # Workers use msg.request_id to reply without a separate argument.
    request_id: Optional[str] = field(default=None, compare=False, kw_only=True)

    def _payload_fields(self):
        _ROUTING = {"source", "consumer_id", "request_id"}
        return [f for f in fields(self) if f.name not in _ROUTING]

    def to_payload(self) -> dict[str, Primitive]:
        out: dict[str, Primitive] = {}
        for f in self._payload_fields():
            out[f.name] = _encode_value(getattr(self, f.name))
        return out

    @classmethod
    def from_payload(cls: Type[TMsg], payload: dict[str, Primitive]) -> TMsg:
        hints = _resolved_hints(cls)
        kwargs: dict[str, Any] = {}
        _ROUTING = {"source", "consumer_id", "request_id"}
        for f in fields(cls):
            if f.name in _ROUTING:
                continue
            if f.name not in payload:
                if f.default is MISSING and f.default_factory is MISSING:  # type: ignore[misc]
                    raise ValueError(
                        f"{cls.__name__}: missing required field '{f.name}'"
                    )
                continue
            kwargs[f.name] = _decode_value(hints.get(f.name, f.type), payload[f.name])
        return cls(**kwargs)


# ---------------------------------------------------------------------
# Typed reply messages
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class OKMessage(Message):
    """Successful reply to a RequestMessage."""
    NAME: ClassVar[str] = "ok"
    KIND: ClassVar[str] = Kind.REPLY


@dataclass(frozen=True)
class ErrorMessage(Message):
    """Error reply to a RequestMessage."""
    NAME: ClassVar[str] = "error"
    KIND: ClassVar[str] = Kind.REPLY
    error: str = ""


# ---------------------------------------------------------------------
# RequestMessage[TReply] — typing marker for commands that expect a reply
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class RequestMessage(Message, Generic[TReply]):
    """
    Marker base for commands that expect a typed reply.
    TReply declares the expected reply class (e.g. OKMessage, CurrentStagePosition).
    No runtime behaviour is added; the generic parameter is for static analysis only.
    """
    pass


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

class MessageRegistry:
    """
    Maps (kind, name) pairs to Message classes for decoding inbound wire dicts.

        registry = base_registry().extend(SetGain, StartSweep)
    """

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], Type[Message]] = {}

    def register(self, *msg_classes: Type[Message]) -> "MessageRegistry":
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
        if len(msg_classes) == 1:
            return msg_classes[0]  # type: ignore[return-value]
        return self

    def extend(self, *msg_classes: Type[Message]) -> "MessageRegistry":
        child = MessageRegistry()
        child._by_key = dict(self._by_key)
        child.register(*msg_classes)
        return child

    def decode(self, envelope: dict[str, Any]) -> Optional[Message]:
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
            object.__setattr__(msg, "source", src)
        cid = envelope.get("consumer_id")
        if isinstance(cid, str):
            object.__setattr__(msg, "consumer_id", cid)
        req_id = envelope.get("id")
        if isinstance(req_id, str):
            object.__setattr__(msg, "request_id", req_id)
        return msg

    def envelope_for(self, msg: Message) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": msg.KIND,
            "name": msg.NAME,
            "payload": msg.to_payload(),
        }
        if msg.consumer_id is not None:
            out["consumer_id"] = msg.consumer_id
        return out


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _resolved_hints(cls: type) -> dict[str, Any]:
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
    if isinstance(declared_type, type) and issubclass(declared_type, PrimitiveSerde):
        return declared_type.from_primitive(value)
    return value
