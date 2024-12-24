import os
import sys
import copy
import numpy as np

# usage:
# control.Once.xxx[(), ()]()

class Once(object):
    def __init__(self, func, **kwargs):
        self.func = func
        self.is_success_call = False
        self.value = None

    def __call__(self, *args, **kwargs):
        if not self.is_success_call:
            val = self.func(*args, **kwargs)
            assert(val is not None)
            if isinstance(val, tuple) or isinstance(val, list):
                self.is_success_call = val[0].shape[0] > 0
            else:
                self.is_success_call = val.shape[0] > 0
            self.value = val

        return self.value
