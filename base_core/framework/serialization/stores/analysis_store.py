# storage_h5/analysis_store.py
from __future__ import annotations

import hashlib
import json
from typing import Any

import h5py
import numpy as np

from base_core.framework.serialization import schema
from base_core.framework.serialization.h5_utils import ensure_group, now_utc_iso, write_array

class AnalysisStore:
    def __init__(self, h5: h5py.File):
        self.h5 = h5

    @staticmethod
    def _id_from_payload(payload: dict[str, Any], *, n_hex: int = 16) -> str:
        s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n_hex]

    def write_average_c2t(
        self,
        run_id: int,
        *,
        input_config_ids: list[str],
        params: dict[str, Any],
        provenance: dict[str, Any],
        delays_s: np.ndarray,
        value: np.ndarray,
        error: np.ndarray | None = None,
    ) -> str:
        payload = {"inputs": input_config_ids, "params": params}
        avg_id = self._id_from_payload(payload)

        g = ensure_group(self.h5, f"{schema.run_analysis_root(run_id)}/aggregates/average_c2t/{avg_id}")
        g.attrs["created_utc"] = now_utc_iso()

        write_utf8(g, "inputs_json", json.dumps(input_config_ids, ensure_ascii=False))
        write_utf8(g, "params_json", json.dumps(params, ensure_ascii=False))
        write_utf8(g, "provenance_json", json.dumps(provenance, ensure_ascii=False))

        dg = ensure_group(g, "data")
        write_array(dg, "delays_s", delays_s, compression="gzip")
        write_array(dg, "value", value, compression="gzip")
        if error is not None:
            write_array(dg, "error", error, compression="gzip")
        else:
            if "error" in dg:
                del dg["error"]

        return avg_id