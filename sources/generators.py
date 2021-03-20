"""Common data generator package"""
from typing import Generator, Sequence, Tuple
from itertools import compress, cycle, islice, product
from more_itertools import distinct_permutations


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
        return (self.epoch for _ in range(self._stop_epoch))

    @property
    def epoch_size(self):
        return self._epoch_size
