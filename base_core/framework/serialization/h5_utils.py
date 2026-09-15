# storage_h5/io_utils.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, TypeVar

import h5py
import numpy as np

T = TypeVar("T")

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def ensure_group(h5: h5py.File | h5py.Group, path: str) -> h5py.Group:
    g: h5py.Group | h5py.File = h5
    for part in path.strip("/").split("/"):
        g = g.require_group(part)
    return g  # type: ignore[return-value]

def write_utf8(g: h5py.Group, name: str, text: str) -> None:
    dt = h5py.string_dtype(encoding="utf-8")
    if name in g:
        del g[name]
    g.create_dataset(name, data=text, dtype=dt)

def read_utf8(g: h5py.Group, name: str) -> str:
    v = g[name][()]
    return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)

def write_primitive(g: h5py.Group, name: str, obj: Any) -> None:
    """Store ``obj`` as canonical JSON of its primitive form, tagged with its class.

    Anything structured -- configs, metadata snapshots -- goes to disk this way rather than
    as hand-flattened attrs, so any reader can rebuild it with :func:`read_primitive` or,
    without the class at hand, just ``json.loads`` the string.
    """
    from base_core.framework.serialization.serialization import to_primitive

    write_utf8(g, name, json.dumps(to_primitive(obj), sort_keys=True, ensure_ascii=False))
    cls = type(obj)
    g[name].attrs["type"] = f"{cls.__module__}.{cls.__qualname__}"

def read_primitive(g: h5py.Group, name: str, cls: type[T]) -> T:
    """Rebuild an object written by :func:`write_primitive`."""
    from base_core.framework.serialization.serialization import from_primitive

    return from_primitive(cls, json.loads(read_utf8(g, name)))

def write_array(
    g: h5py.Group,
    name: str,
    arr: Any,
    *,
    compression: str = "lzf",
    dtype: Any | None = None,
) -> None:
    a = np.asarray(arr if dtype is None else np.asarray(arr, dtype=dtype))
    if name in g:
        del g[name]
    g.create_dataset(name, data=a, chunks=True, compression=compression, shuffle=True)

def ensure_table(
    g: h5py.Group,
    name: str,
    dtype: np.dtype,
) -> h5py.Dataset:
    if name in g:
        ds = g[name]
        assert isinstance(ds, h5py.Dataset)
        return ds
    return g.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype)

def append_row(ds: h5py.Dataset, row: np.void) -> None:
    n = ds.shape[0]
    ds.resize((n + 1,))
    ds[n] = row