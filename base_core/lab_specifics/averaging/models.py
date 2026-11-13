from dataclasses import dataclass

from base_core.lab_specifics.base_models import ScanDataBase


@dataclass(frozen=True, kw_only=True)
class AveragedScansData(ScanDataBase):
    run_ids: list[int]