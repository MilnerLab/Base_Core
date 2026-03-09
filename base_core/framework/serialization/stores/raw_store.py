# storage_h5/raw_store.py
from __future__ import annotations

from base_core.framework.serialization import schema
from base_core.framework.serialization.h5_utils import ensure_group, now_utc_iso, write_array, ensure_table, append_row
import h5py
import numpy as np


# Index-Tabelle pro Run: ion_data
ION_INDEX_DTYPE = np.dtype([
    ("ion_idx", "i8"),
    ("stage_position_m", "f8"),
    ("ions_per_frame", "f8"),
    ("n_points", "i8"),
    ("created_utc", "S32"),
    ("complete", "i1"),
])

RAW_STATE_RECORDING = "recording"
RAW_STATE_CLOSED = "closed"

class RawStore:
    def __init__(self, h5: h5py.File):
        self.h5 = h5

    def init_run_structures(self, run_id: int) -> None:
        rg = ensure_group(self.h5, schema.run_root(run_id))

        rg.attrs.setdefault("raw_state", RAW_STATE_RECORDING)
        rg.attrs.setdefault("raw_opened_utc", now_utc_iso())
        rg.attrs.setdefault("raw_last_append_utc", "")
        rg.attrs.setdefault("raw_closed_utc", "")
        rg.attrs.setdefault("next_ion_idx", 0)

        ig = ensure_group(self.h5, schema.run_index(run_id))
        ensure_table(ig, "ion_data", ION_INDEX_DTYPE)

        ensure_group(self.h5, schema.run_raw_ion_data(run_id))

    def get_state(self, run_id: int) -> str:
        rg = self.h5[schema.run_root(run_id)]
        return str(rg.attrs.get("raw_state", RAW_STATE_RECORDING))

    def close(self, run_id: int) -> None:
        rg = self.h5[schema.run_root(run_id)]
        rg.attrs["raw_state"] = RAW_STATE_CLOSED
        rg.attrs["raw_closed_utc"] = now_utc_iso()

    def append_ion_data(self, run_id: int, ion_data) -> int:
        """
        ion_data expected to have:
          - id (run_id)
          - ions_per_frame: float
          - stage_position: Length (float subclass in meters)
          - points: Points with x/y arrays
        """
        rg = self.h5[schema.run_root(run_id)]
        if str(rg.attrs.get("raw_state", RAW_STATE_RECORDING)) != RAW_STATE_RECORDING:
            raise RuntimeError("Raw scan is closed; cannot append IonData.")

        if int(getattr(ion_data, "id")) != run_id:
            raise ValueError("IonData.id must equal run_id (ownership check).")

        ion_idx = int(rg.attrs["next_ion_idx"])
        rg.attrs["next_ion_idx"] = ion_idx + 1
        rg.attrs["raw_last_append_utc"] = now_utc_iso()

        # group name: zero-padded so lexicographic order == numeric order
        ion_name = f"{ion_idx:06d}"
        g = ensure_group(self.h5, f"{schema.run_raw_ion_data(run_id)}/{ion_name}")

        # attrs (SI)
        stage_m = float(getattr(ion_data, "stage_position"))
        ions_pf = float(getattr(ion_data, "ions_per_frame"))
        g.attrs["created_utc"] = now_utc_iso()
        g.attrs["stage_position_m"] = stage_m
        g.attrs["ions_per_frame"] = ions_pf

        # points
        pts = getattr(ion_data, "points")
        pg = ensure_group(g, "points")
        write_array(pg, "x", pts.x, dtype=np.float64)
        write_array(pg, "y", pts.y, dtype=np.float64)

        n_points = int(np.asarray(pts.x).size)

        # index update
        ig = self.h5[schema.run_index(run_id)]
        ds = ig["ion_data"]
        row = np.array(
            (ion_idx, stage_m, ions_pf, n_points, now_utc_iso().encode("utf-8"), 1),
            dtype=ION_INDEX_DTYPE,
        )[()]
        append_row(ds, row)

        return ion_idx

    def list_ion_data(self, run_id: int) -> np.ndarray:
        """Returns the full ion_data index table as a numpy structured array."""
        ig = self.h5[schema.run_index(run_id)]
        return ig["ion_data"][...]

    def read_ion_data(self, run_id: int, ion_idx: int, *, load_points: bool = True):
        """
        Returns a minimal dict; you can rebuild IonData outside if you want.
        """
        ion_name = f"{ion_idx:06d}"
        g = self.h5[f"{schema.run_raw_ion_data(run_id)}/{ion_name}"]

        out = {
            "run_id": run_id,
            "ion_idx": ion_idx,
            "stage_position_m": float(g.attrs["stage_position_m"]),
            "ions_per_frame": float(g.attrs["ions_per_frame"]),
            "created_utc": str(g.attrs.get("created_utc", "")),
        }

        if load_points:
            pg = g["points"]
            out["x"] = np.asarray(pg["x"], dtype=np.float64)
            out["y"] = np.asarray(pg["y"], dtype=np.float64)

        return out