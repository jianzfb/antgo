import os
import sys
import copy
import numpy as np

# usage:
# control.RandomChoice.xxx[(),()]()

class RandomChoice(object):
    def __init__(self, func, sampling_num=1, **kwargs):
        self.func = func
        self.sampling_num = sampling_num
        self.sampling_i = 0

    def __call__(self, *args):
        out_list = []
        while(self.sampling_i < self.sampling_num):
            sampling_arg_data = []
            for arg_data in args:
                random_i = np.random.randint(0, len(arg_data))
                sampling_arg_data.append(arg_data[random_i])

            res = self.func(*sampling_arg_data)
            if res is not None:
                out_list.append(res)
            self.sampling_i += 1

        return out_list