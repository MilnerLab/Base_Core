import numpy as np
from typing import Sequence, Tuple

def moving_average(
    x: Sequence[float],
    y: Sequence[float],
    window_size: int,
) -> Tuple[list[float], list[float]]:

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    n = len(x)
    if window_size < 1 or window_size > n:
        raise ValueError("window_size should be between 1 and len(x).")
    if window_size % 2 == 0:
        raise ValueError("window_size should be uneven.")

    half = window_size // 2

    x_smooth = []
    y_smooth = []

    for i in range(half, n - half):
        x_window = x[i - half : i + half + 1]
        y_window = y[i - half : i + half + 1]

        x_smooth.append(np.mean(x_window))  
        y_smooth.append(np.mean(y_window))   

    return x_smooth, y_smooth
