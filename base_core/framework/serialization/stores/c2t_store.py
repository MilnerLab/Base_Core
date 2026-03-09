# storage_h5/c2t_store.py
from __future__ import annotations

import h5py
import numpy as np

from base_core.framework.serialization import schema
from base_core.framework.serialization.h5_utils import ensure_group, now_utc_iso, write_array, ensure_table, append_row

C2T_INDEX_DTYPE = np.dtype([
    ("config_id", "S32"),
    ("created_utc", "S32"),
    ("status", "S16"),
])

class C2TStore:
    def __init__(self, h5: h5py.File):
        self.h5 = h5

    def init_run_structures(self, run_id: int) -> None:
        ig = ensure_group(self.h5, schema.run_index(run_id))
        ensure_table(ig, "c2t_results", C2T_INDEX_DTYPE)
        ensure_group(self.h5, schema.run_c2t_root(run_id))

    def _upsert_index(self, run_id: int, config_id: str, status: str) -> None:
        ds = self.h5[f"{schema.run_index(run_id)}/c2t_results"]
        cid_b = config_id.encode("utf-8")
        status_b = status.encode("utf-8")
        ts_b = now_utc_iso().encode("utf-8")

        arr = ds[...]
        for i in range(arr.shape[0]):
            if arr[i]["config_id"] == cid_b:
                arr[i]["created_utc"] = ts_b
                arr[i]["status"] = status_b
                ds[...] = arr
                return

        # not found -> append
        n = ds.shape[0]
        ds.resize((n + 1,))
        ds[n] = np.array((cid_b, ts_b, status_b), dtype=C2T_INDEX_DTYPE)[()]

    def write_c2t(self, run_id: int, config_id: str, c2t_scan, *, link_config: bool = True) -> None:
        """
        c2t_scan expected to have:
          - delays: list[Time] (seconds internally)
          - measured_values: list[Measurement(value,error)]
          - ions_per_frame: list[float] | None
        overwrites existing data for same (run_id, config_id).
        """
        g = ensure_group(self.h5, f"{schema.run_c2t_root(run_id)}/{config_id}")
        g.attrs["created_utc"] = now_utc_iso()
        g.attrs["status"] = "ok"
        g.attrs["config_id"] = config_id

        # overwrite datasets
        delays_s = np.asarray([float(t) for t in c2t_scan.delays], dtype=np.float64)
        values = np.asarray([float(m.value) for m in c2t_scan.measured_values], dtype=np.float64)
        errors = np.asarray([float(m.error) for m in c2t_scan.measured_values], dtype=np.float64)

        write_array(g, "delays_s", delays_s, compression="gzip")
        write_array(g, "value", values, compression="gzip")
        write_array(g, "error", errors, compression="gzip")

        ions = getattr(c2t_scan, "ions_per_frame", None)
        if ions is not None:
            write_array(g, "ions_per_frame", np.asarray(ions, dtype=np.float64), compression="gzip")
        else:
            if "ions_per_frame" in g:
                del g["ions_per_frame"]

        # softlink to config registry
        if link_config:
            if "config" in g:
                del g["config"]
            g["config"] = h5py.SoftLink(schema.config_path(config_id))

        self._upsert_index(run_id, config_id, "ok")

    def list_c2t(self, run_id: int) -> np.ndarray:
        return self.h5[f"{schema.run_index(run_id)}/c2t_results"][...]

    def read_c2t_arrays(self, run_id: int, config_id: str) -> dict:
        g = self.h5[f"{schema.run_c2t_root(run_id)}/{config_id}"]
        out = {
            "config_id": config_id,
            "created_utc": str(g.attrs.get("created_utc", "")),
            "status": str(g.attrs.get("status", "")),
            "delays_s": np.asarray(g["delays_s"], dtype=np.float64),
            "value": np.asarray(g["value"], dtype=np.float64),
            "error": np.asarray(g["error"], dtype=np.float64),
        }
        if "ions_per_frame" in g:
            out["ions_per_frame"] = np.asarray(g["ions_per_frame"], dtype=np.float64)
        return out