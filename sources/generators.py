"""Common data generator package"""
from typing import Generator, Sequence, Tuple, Callable, List
from itertools import islice, count


# Type hinting
RawDataGenerator = Generator[Tuple[Sequence[float], float], None, None]


class DataGenerator:
    def __init__(self, generator: RawDataGenerator, epoch_size: int, stop_epoch: int = None):
        self._generator = generator
        self._epoch_size = epoch_size
        self._stop_epoch = stop_epoch

    @property
    def epoch(self):
        return islice(self._generator, self._epoch_size)

    @property
    def eternity(self):
        if self._stop_epoch is None:
            return (self.epoch for _ in count())
        return (self.epoch for _ in range(self._stop_epoch))

    @property
    def epoch_size(self):
        return self._epoch_size


def function_as_points(function: Callable, bounds: Tuple[float, float], points: int):
    left, right = bounds
    step = abs(right - left) / (points)
    values = [
        function(left + idx * step)
        for idx
        in range(points)
    ]
    return values


def window_at_position(values: List[float], size: int, idx: int):
    return values[idx - size: idx], values[idx] if idx < len(values) else None
