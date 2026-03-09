from dataclasses import dataclass
from pathlib import Path

from _domain.models import ScanDataBase

@dataclass(frozen=True)
class AveragedScansData(ScanDataBase):
    run_ids: list[int]