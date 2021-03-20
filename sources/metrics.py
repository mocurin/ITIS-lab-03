"""Common metric functions package"""
from typing import Any, Callable, List


# Type hinting
Metric = Callable[[List[float], List[int]], Any]

