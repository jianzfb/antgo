import os
import sys
import copy
import numpy as np

# usage:
# control.Asyn.xxx[(),()]()

class Asyn(object):
    def __init__(self, func, **kwargs):
        self.func = func

    def __call__(self, *args):
        return self.func(*args)
