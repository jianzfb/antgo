import os
import sys
import copy
import numpy as np

# usage:
# control.For.xxx[(),()]()

class For(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        out = []
        for data in args[0]:
            out.append(self.func(data))
        return np.stack(out, 0)