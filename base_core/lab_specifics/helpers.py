from base_core.quantities.constants import SPEED_OF_LIGHT
from base_core.quantities.models import Length, Time


def calculate_time_delay(stage_position: Length, delay_center: Length) -> Time:
    delta = (stage_position - delay_center) * 2

    return Time(delta / SPEED_OF_LIGHT)

