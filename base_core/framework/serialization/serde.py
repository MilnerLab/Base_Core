from __future__ import annotations

from abc import ABC
from typing import Any

Primitive = Any  # JSON-friendly: dict/list/str/int/float/bool/None


class PrimitiveSerde(ABC):
    """Mark + API for types that serialize to/from JSON-friendly primitives.

    Dataclass subclasses get working to_primitive/from_primitive for free.
    Non-dataclass subclasses (e.g. float-based quantities) must override both.
    """

    def to_primitive(self) -> Primitive:
        from dataclasses import fields, is_dataclass
        if not is_dataclass(self):
            raise NotImplementedError(f"{type(self).__name__} must implement to_primitive")
        from base_core.framework.serialization.serialization import to_primitive as _tp
        return {f.name: _tp(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_primitive(cls, v: Primitive) -> "PrimitiveSerde":
        from dataclasses import fields, is_dataclass
        if not is_dataclass(cls):
            raise NotImplementedError(f"{cls.__name__} must implement from_primitive")
        from typing import get_type_hints
        from base_core.framework.serialization.serialization import _convert_field
        hints = get_type_hints(cls)
        return cls(**{
            f.name: _convert_field(hints.get(f.name, type(None)), v[f.name])
            for f in fields(cls)
            if f.name in v
        })