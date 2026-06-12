from __future__ import annotations

import dataclasses
import json
import logging
import types
import typing
from typing import Any, get_args, get_origin, get_type_hints

from base_core.framework.serialization.serde import PrimitiveSerde
from base_core.framework.serialization.serialization import to_primitive
from base_core.ipc.message import ErrorReply, Message, OKReply, Reply

log = logging.getLogger(__name__)

_registry: dict[str, type[Message]] = {}


def register(cls: type[Message]) -> type[Message]:
    """Decorator — adds cls to the codec registry under cls.__name__."""
    _registry[cls.__name__] = cls
    return cls


register(Message)
register(Reply)
register(OKReply)
register(ErrorReply)


def encode(msg: Message) -> bytes:
    """
    Serialize msg to UTF-8 JSON bytes with a trailing newline.
    Wire format: {"_type": "ClassName", "id": "...", <fields>...}\\n
    """
    data: dict[str, Any] = {"_type": type(msg).__name__}
    for f in dataclasses.fields(msg):
        data[f.name] = to_primitive(getattr(msg, f.name))
    return (json.dumps(data) + "\n").encode("utf-8")


def decode(data: bytes) -> Message:
    """
    Deserialize bytes to a Message subclass.
    Raises KeyError if _type is not registered, json.JSONDecodeError on malformed input.
    """
    raw: dict[str, Any] = json.loads(data.decode("utf-8"))
    type_name: str = raw.pop("_type")
    cls = _registry[type_name]
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name in raw:
            kwargs[f.name] = _reconstruct(raw[f.name], hints.get(f.name, type(None)))
    return cls(**kwargs)


def _reconstruct(value: Any, t: Any) -> Any:
    if value is None:
        return None

    origin = get_origin(t)
    args = get_args(t)

    # Optional[X] (typing.Union) or X | None (types.UnionType, Python 3.10+)
    is_union = origin is typing.Union or (
        hasattr(types, "UnionType") and isinstance(t, types.UnionType)
    )
    if is_union:
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _reconstruct(value, non_none[0])
        return value

    # list[X]
    if origin is list and args:
        return [_reconstruct(item, args[0]) for item in value]

    # PrimitiveSerde takes priority over generic is_dataclass check
    if isinstance(t, type) and issubclass(t, PrimitiveSerde):
        return t.from_primitive(value)

    # Nested plain dataclass
    if isinstance(t, type) and dataclasses.is_dataclass(t):
        inner_hints = get_type_hints(t)
        kw: dict[str, Any] = {}
        for f in dataclasses.fields(t):
            if f.name in value:
                kw[f.name] = _reconstruct(value[f.name], inner_hints.get(f.name, type(None)))
        return t(**kw)

    return value
