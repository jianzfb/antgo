import os
import sys
import copy
import numpy as np
from antgo.pipeline.deploy.cpp_op import CppOp

# usage:
# control.IfNotNone.xxx[(), ()]()

class IfNotNone(object):
    def __init__(self, func, **kwargs):
        self.func = func
        self._index = []

    def __call__(self, *args):
        is_none = False
        for val in args:
            if val is None:
                is_none = True
                break

        if is_none:
            return [None] * len(self._index[1])

        if isinstance(self.func, CppOp):
            self.func._index = self._index
        return self.func(*args)
