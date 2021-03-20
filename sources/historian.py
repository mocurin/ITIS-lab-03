"""Data-writer & logger class"""
import logging
import enum

import matplotlib.pyplot as plt

from typing import Callable, Dict, List

from . import models


class LoggingVerbosity(enum.IntEnum):
    SILENCE = 0
    ETERNITY = 1
    EPOCH = 2
    BATCH = 3
    SAMPLE = 4


class WriteVerbosity(enum.IntEnum):
    SILENCE = 0
    ETERNITY = 1
    EPOCH = 2
    BATCH = 3
    SAMPLE = 4


class FitEvents(enum.IntEnum):
    ETERNITY = 1
    EPOCH = 2
    BATCH = 3
    SAMPLE = 4


class Historian:
    def __init__(self,
                 logging_verbosity: LoggingVerbosity = LoggingVerbosity.EPOCH,
                 write_verbosity: WriteVerbosity = WriteVerbosity.EPOCH,
                 describers: Dict[FitEvents, Callable] = None):
        self._logging_verbosity = logging_verbosity
        self._write_verbosity = write_verbosity

        # Avoid mutable default arguments
        if describers is None:
            describers = dict()

        # Do not check wether all describers are implemented
        self._describers = describers
        self._storage = {
            event: list()
            for event
            in list(FitEvents)
            if not event > write_verbosity
        }

    def store(self, event: FitEvents, model: 'models.Model', *args):
        if not event > self._logging_verbosity:
            # Describers could be missing
            describer = self._describers.get(event)

            if describer:
                describer(model, *args)

        if not event > self._write_verbosity:
            # Storages could not be missing
            self._storage[event].append([*model.describe(), *args])
