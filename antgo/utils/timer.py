#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: timer.py
# Author: jian(jian@mltalker.com)
from __future__ import unicode_literals

from contextlib import contextmanager
import time
from collections import defaultdict
import six
import atexit
import numpy as np
from . import logger


class StatCounter(object):
    """ A simple counter"""
    def __init__(self):
        self.reset()

    def feed(self, v):
        self._values.append(v)

    def reset(self):
        self._values = []

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        assert len(self._values)
        return np.mean(self._values)

    @property
    def sum(self):
        assert len(self._values)
        return np.sum(self._values)

    @property
    def max(self):
        assert len(self._values)
        return max(self._values)

class IterSpeedCounter(object):
    """ To count how often some code gets reached"""
    def __init__(self, print_every, name=None):
        self.cnt = 0
        self.print_every = int(print_every)
        self.name = name if name else 'IterSpeed'

    def reset(self):
        self.start = time.time()

    def __call__(self):
        if self.cnt == 0:
            self.reset()
        self.cnt += 1
        if self.cnt % self.print_every != 0:
            return
        t = time.time() - self.start
        logger.info("{}: {:.2f} sec, {} times, {:.3g} sec/time".format(
            self.name, t, self.cnt, t / self.cnt))

@contextmanager
def timed_operation(msg, log_start=False):
    if log_start:
        logger.info('Start {} ...'.format(msg))
    start = time.time()
    yield
    logger.info('{} Finished, Time={:.2f}sec.'.format(
        msg, time.time() - start))

_TOTAL_TIMER_DATA = defaultdict(StatCounter)

@contextmanager
def total_timer(msg):
    start = time.time()
    yield
    t = time.time() - start
    _TOTAL_TIMER_DATA[msg].feed(t)

def get_elapsed_time(msg):
    if len(_TOTAL_TIMER_DATA) == 0:
        return 0.0

    msg_timer = _TOTAL_TIMER_DATA[msg]
    return msg_timer.sum