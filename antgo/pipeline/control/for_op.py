import os
import sys
import copy
import numpy as np

# usage:
# control.For.xxx[(),()]()

class For(object):
    def __init__(self, func, parallel_num=1):
        self.func = func
        self.parallel_num = parallel_num

    def __call__(self, *args):
        out_list = None
        loop_num = max([v.shape[0] for v in args])
        if min([v.shape[0] for v in args]) == 0:
            return np.empty((0), dtype=np.float32)

        for loop_i in range(loop_num):
            data_tuple = [v[loop_i%v.shape[0]] for v in args]
            out = self.func(*data_tuple)
            if not isinstance(out, list) and not isinstance(out, tuple()):
                out = [out]

            if out_list is None:
                out_list = [[] for _ in range(len(out))]

            for elem_i, elem_v in enumerate(out):
                out_list[elem_i].append(elem_v)
        if out_list is None:
            return np.empty((0), dtype=np.float32)

        for elem_i, elem_v in enumerate(out_list):
            out_list[elem_i] = np.stack(elem_v, 0)
        return out_list if len(out_list) > 1 else out_list[0]