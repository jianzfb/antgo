import os
import sys
import copy
import numpy as np

# usage:
# control.Interval.xxx[(), ()]()

class Interval(object):
    def __init__(self, func, interval=1):
        self.func = func
        self.value = None
        self.count = 0
        self.interval = interval

    def __call__(self, *args):
        if self.count % self.interval == 0:
            self.value = self.func(*args)

        self.count += 1
        return self.value
