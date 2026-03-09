
import csv
from dataclasses import dataclass
import math
from pathlib import Path


from _data_io.dat_loader import calculate_time_delay
from apps.c2t_calculation.domain.analysis import avg_c2t
from apps.c2t_calculation.domain.config import IonDataAnalysisConfig
from base_core.math.models import Point, Range, Points
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
            pts = Points(d.points.x.copy(), d.points.y.copy())

            # --- use existing Points methods ---
            pts.subtract(config.center)
            pts.affine_transform(config.transform_parameter)
            pts.rotate(config.angle)  # rotation around origin (already centered)
            pts = pts.filter_by_distance_range(config.analysis_zone)

            delays.append(calculate_time_delay(d.stage_position, config.delay_center))
            c2t.append(avg_c2t(pts))
            ions.append(d.ions_per_frame)


        return cls(
            delays=delays,
            measured_values=c2t,
            config=config,
            run_id=raw.run_id,
            ions_per_frame = ions)