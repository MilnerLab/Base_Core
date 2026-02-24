
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, Union, get_args, get_origin

import numpy as np
from base_core.framework.serialization.serde import Primitive, PrimitiveSerde


def to_primitive(obj: Any) -> Primitive:
    """Convert dataclasses + PrimitiveSerde objects into JSON-friendly primitives."""
    
    if isinstance(obj, PrimitiveSerde):
        return obj.to_primitive()
    
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # numpy scalars -> python scalars
    if isinstance(obj, np.generic):
        return obj.item()

    if is_dataclass(obj):
        return {f.name: to_primitive(getattr(obj, f.name)) for f in fields(obj)}

    if isinstance(obj, (list, tuple)):
        return [to_primitive(v) for v in obj]

    if isinstance(obj, dict):
        return {str(k): to_primitive(v) for k, v in obj.items()}

    raise TypeError(f"Cannot serialize type {type(obj)}. Add PrimitiveSerde or handle it explicitly.")


def from_primitive(cls: type[Any], data: Primitive) -> Any:
    """
    Rebuild a dataclass instance of type `cls` from primitives, using type hints.
    Works with:
      - dataclasses
      - PrimitiveSerde subclasses (Angle, Point, Range, ...)
      - Optional, list, tuple, dict
    """
    def convert(t: Any, v: Any) -> Any:
        if v is None:
            return None

        origin = get_origin(t)
        args = get_args(t)

        # Optional[T] / Union[T, None]
        if origin is Union and type(None) in args:
            inner = next(a for a in args if a is not type(None))
            return convert(inner, v)

        # list[T]
        if origin is list:
            (inner,) = args
            return [convert(inner, x) for x in v]

        # dict[K, V]
        if origin is dict:
            k_t, v_t = args
            return {convert(k_t, kk): convert(v_t, vv) for kk, vv in v.items()}

        # tuple[T, ...] or tuple[T1, T2, ...]
        if origin is tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(convert(args[0], x) for x in v)
            return tuple(convert(ti, xi) for ti, xi in zip(args, v))

        # If it's a parameterized type like Range[int], origin is Range
        if origin is not None and isinstance(origin, type) and issubclass(origin, PrimitiveSerde):
            return origin.from_primitive(v)

        # Plain PrimitiveSerde
        if isinstance(t, type) and issubclass(t, PrimitiveSerde):
            return t.from_primitive(v)

        # Nested dataclass type
        if isinstance(t, type) and is_dataclass(t):
            return from_primitive(t, v)

        # primitive / fallback
        return v

    # If the target itself is PrimitiveSerde (rare), build it directly
    if isinstance(cls, type) and issubclass(cls, PrimitiveSerde):
        return cls.from_primitive(data)

    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass (and not PrimitiveSerde).")

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict for {cls.__name__}, got {type(data)}")

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name in data:
            kwargs[f.name] = convert(f.type, data[f.name])
            continue

        # allow missing keys if dataclass has defaults
        if f.default is not MISSING:
            continue
        if getattr(f, "default_factory", MISSING) is not MISSING:  # type: ignore[attr-defined]
            continue

        raise KeyError(f"Missing key '{f.name}' for {cls.__name__}")

    return cls(**kwargs)