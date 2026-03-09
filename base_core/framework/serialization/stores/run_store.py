# storage_h5/run_store.py
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from base_core.framework.serialization import schema
from base_core.framework.serialization.h5_utils import ensure_group, now_utc_iso
import h5py


class RunH5Store:
    def __init__(self, path: str):
        self.path = path

    @contextmanager
    def open(self, mode: str = "a") -> Iterator[h5py.File]:
        with h5py.File(self.path, mode) as h5:
            # root attrs
            h5.attrs.setdefault("format_name", schema.FORMAT_NAME)
            h5.attrs.setdefault("format_version", schema.FORMAT_VERSION)
            h5.attrs.setdefault("created_utc", now_utc_iso())
            ensure_group(h5, schema.ROOT_RUNS)
            ensure_group(h5, schema.ROOT_CONFIGS)
            yield h5

    def init_run(self, run_id: int) -> None:
        with self.open("a") as h5:
            ensure_group(h5, schema.run_root(run_id))
            RawStore(h5).init_run_structures(run_id)
            C2TStore(h5).init_run_structures(run_id)
            ensure_group(h5, schema.run_analysis_root(run_id))

    # --- raw ---
    def get_raw_state(self, run_id: int) -> str:
        with self.open("r") as h5:
            return RawStore(h5).get_state(run_id)

    def close_raw(self, run_id: int) -> None:
        with self.open("a") as h5:
            RawStore(h5).close(run_id)

    def append_ion_data(self, run_id: int, ion_data) -> int:
        with self.open("a") as h5:
            return RawStore(h5).append_ion_data(run_id, ion_data)

    def list_ion_data(self, run_id: int):
        with self.open("r") as h5:
            return RawStore(h5).list_ion_data(run_id)

    def read_ion_data(self, run_id: int, ion_idx: int, *, load_points: bool = True):
        with self.open("r") as h5:
            return RawStore(h5).read_ion_data(run_id, ion_idx, load_points=load_points)

    # --- configs (global) ---
    def ensure_analysis_config(self, cfg, *, label: str | None = None) -> str:
        with self.open("a") as h5:
            return ConfigRegistry(h5).ensure(cfg, label=label)

    def list_configs(self):
        with self.open("r") as h5:
            return ConfigRegistry(h5).list()

    def find_configs_by_label(self, label: str):
        with self.open("r") as h5:
            return ConfigRegistry(h5).find_by_label(label)

    # --- c2t ---
    def write_c2t(self, run_id: int, c2t_scan, *, config_label: str | None = None) -> str:
        """
        c2t_scan must have .config (IonDataAnalysisConfig) :contentReference[oaicite:5]{index=5}
        Overwrites /derived/c2t/<config_id> for this run.
        """
        with self.open("a") as h5:
            reg = ConfigRegistry(h5)
            cid = reg.ensure(c2t_scan.config, label=config_label)
            C2TStore(h5).write_c2t(run_id, cid, c2t_scan, link_config=True)
            return cid

    def list_c2t(self, run_id: int):
        with self.open("r") as h5:
            return C2TStore(h5).list_c2t(run_id)

    def read_c2t_arrays(self, run_id: int, config_id: str):
        with self.open("r") as h5:
            return C2TStore(h5).read_c2t_arrays(run_id, config_id)

    # --- analysis ---
    def write_average_c2t(self, run_id: int, **kwargs) -> str:
        with self.open("a") as h5:
            return AnalysisStore(h5).write_average_c2t(run_id, **kwargs)