# domain/config.py
from dataclasses import dataclass

from base_core.math.models import Angle, Point, Range
from base_core.quantities.models import Length


@dataclass(frozen=True, slots=True)
class IonDataAnalysisConfig:
    delay_center: Length
    center: Point
    angle: Angle
    analysis_zone: Range[float]
    transform_parameter: float