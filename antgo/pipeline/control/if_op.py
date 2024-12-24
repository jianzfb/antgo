import os
import sys
import copy
import numpy as np

# usage:
# control.If.true_func.xxx.false_func.yyy[(), ()]()

class If(object):
    def __init__(self, true_func, false_func, **kwargs):
        self.true_func = true_func
        self.false_func = false_func

    def __call__(self, *args):
        true_or_false_val = args[0]
        func_args = args[1:] if len(args) > 1 else []
        if bool(true_or_false_val):
            return self.true_func(*func_args)
        else:
            return self.false_func(*func_args)
