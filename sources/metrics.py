"""Common metric functions package"""
import math

from typing import Any, Callable, List


# Type hinting
Metric = Callable[[List[float], List[float]], Any]


def mean_square_error(outputs: List[float], targets: List[float]) -> float:
    return math.sqrt(sum((out - tgt) ** 2 for out, tgt in zip(outputs, targets)))
