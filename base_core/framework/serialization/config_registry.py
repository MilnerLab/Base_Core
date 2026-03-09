# storage_h5/config_registry.py
from __future__ import annotations

import hashlib
import json
from dataclasses import is_dataclass
from typing import Any, Callable, Optional, TypeVar

import h5py

from storage_h5 import schema
from storage_h5.io_utils import ensure_group, now_utc_iso, read_utf8, write_utf8

from serialization import to_primitive, from_primitive  # :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

T = TypeVar("T")

class ConfigRegistry:
    """
    Global config registry:
      /configs/analysis/<config_id>/config_json
    """

    def __init__(self, h5: h5py.File):
        self.h5 = h5

    @staticmethod
    def _canonical_json(obj: Any) -> str:
        prim = to_primitive(obj)  # dataclass + PrimitiveSerde leaf types :contentReference[oaicite:4]{index=4}
        return json.dumps(prim, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def compute_id(cls, cfg: Any, *, n_hex: int = 16) -> tuple[str, str]:
        canon = cls._canonical_json(cfg)
        cid = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:n_hex]
        return cid, canon

    def ensure(self, cfg: Any, *, label: str | None = None) -> str:
        cid, canon = self.compute_id(cfg)
        g = ensure_group(self.h5, schema.config_path(cid))

        if "config_json" not in g:
            write_utf8(g, "config_json", canon)
            g.attrs["created_utc"] = now_utc_iso()
            g.attrs["hash_alg"] = "sha256"
            g.attrs["schema_version"] = 1
            g.attrs["kind"] = type(cfg).__name__

        # label kannst du jederzeit setzen/ändern (ändert NICHT den Hash)
        if label is not None:
            g.attrs["label"] = label

        return cid

    def get_label(self, cid: str) -> str | None:
        g = self.h5.get(schema.config_path(cid))
        if g is None:
            return None
        return g.attrs.get("label")

    def list(self) -> list[tuple[str, str | None]]:
        base = self.h5.get(schema.ROOT_CONFIGS)
        if base is None:
            return []
        out: list[tuple[str, str | None]] = []
        for cid in sorted(base.keys()):
            g = base[cid]
            out.append((cid, g.attrs.get("label")))
        return out

    def find_by_label(self, label: str) -> list[str]:
        base = self.h5.get(schema.ROOT_CONFIGS)
        if base is None:
            return []
        hits = []
        for cid in base.keys():
            if base[cid].attrs.get("label") == label:
                hits.append(cid)
        return sorted(hits)

    def read(self, cid: str, cfg_type: type[T]) -> T:
        g = self.h5[schema.config_path(cid)]
        prim = json.loads(read_utf8(g, "config_json"))
        return from_primitive(cfg_type, prim)