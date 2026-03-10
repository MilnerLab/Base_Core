
import csv
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


from base_core.lab_specifics.c2t.config import IonDataAnalysisConfig
from base_core.lab_specifics.helpers import calculate_time_delay
from base_core.math.models import Points
from base_core.quantities.models import Length, Time


@dataclass(frozen=True)
class Measurement:
    value: float
    error: float

@dataclass(frozen=True)
class ScanDataBase:
    delays: list[Time]
    measured_values: list[Measurement]
    run_id:int

    def to_csv(self, path: str | Path) -> None:
        if path == None:
            print('No path specified, did not save data.')
        else:
            with Path(path).open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for d, m in zip(self.delays, self.measured_values):
                    w.writerow([d, m.value, m.error])
            print("Data saved to:",path)
            
    def cut(self, start: int = 0, end: int = 0) -> None:
        n = len(self.delays)
        if len(self.measured_values) != n or start < 0 or end < 0 or start + end > n:
            raise ValueError("Invalid cut range.")

        if end:
            del self.delays[-end:]
            del self.measured_values[-end:]
        if start:
            del self.delays[:start]
            del self.measured_values[:start]

@dataclass
class IonData:
    id: int
    ions_per_frame: float
    stage_position: Length
    points: Points
    
    def get_points_after_config(self, config: IonDataAnalysisConfig) -> Points:
        pts = Points(self.points.x.copy(), self.points.y.copy())
        # --- use existing Points methods ---
        pts.subtract(config.center)
        pts.affine_transform(config.transform_parameter)
        pts.rotate(config.angle)  # rotation around origin (already centered)
        pts = pts.filter_by_distance_range(config.analysis_zone)
        return pts
    
    @staticmethod
    def avg_c2t(points: Points) -> Measurement:
        """
        Compute <cos^2(theta)> and its SEM for a set of 2D points, where
        theta = arctan2(y, x).

        Returns Measurement(mean, sem).
        """
        n = len(points)
        if n == 0:
            raise ValueError("No ions in data.")

        theta = np.arctan2(points.y, points.x)
        c2 = np.cos(theta) ** 2

        mean = float(np.mean(c2))
        std = float(np.std(c2, ddof=1)) if n > 1 else np.nan
        sem = float(std / math.sqrt(n)) if (n > 1 and np.isfinite(std)) else np.nan
        return Measurement(mean, sem)
    

@dataclass
class RawScanData:
    run_id: int
    ion_datas: list[IonData]
    
    def add_ion_data(self, ion_data: IonData) -> None:
        if ion_data.id != self.run_id:
            raise ValueError('IonData does not belonge to this run.')
    
        self.ion_datas.append(ion_data)
            
@dataclass(frozen=True)
class C2TScanData(ScanDataBase):
    config: IonDataAnalysisConfig
    ions_per_frame: list[float] | None = None

    @classmethod
    def from_raw(
        cls,
        raw: RawScanData,
        config: IonDataAnalysisConfig,
    ) -> "C2TScanData":
        delays: list[Time] = []
        c2t: list[Measurement] = []
        ions: list[float] = []

        for d in raw.ion_datas:
            # --- copy raw points so we do NOT mutate RawScanData ---
            pts = d.get_points_after_config(config)

            delays.append(calculate_time_delay(d.stage_position, config.delay_center))
            c2t.append(IonData.avg_c2t(pts))
            ions.append(d.ions_per_frame)


        return cls(
            delays=delays,
            measured_values=c2t,
            config=config,
            run_id=raw.run_id,
            ions_per_frame = ions)