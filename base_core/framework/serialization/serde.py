from abc import ABC, abstractmethod
from typing import Any

Primitive = Any  # JSON-friendly: dict/list/str/int/float/bool/None

class PrimitiveSerde(ABC):
    """Mark + API for custom value types that know how to serialize themselves."""

    @abstractmethod
    def to_primitive(self) -> Primitive: ...

    @classmethod
    @abstractmethod
    def from_primitive(cls, v: Primitive) -> "PrimitiveSerde": ...
