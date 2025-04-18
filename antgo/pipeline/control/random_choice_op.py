import os
import sys
import copy
import numpy as np

# usage:
# control.RandomChoice.xxx[(),()]()

class RandomChoice(object):
    def __init__(self, func, sampling_num=1, sampling_group=None, **kwargs):
        self.func = func
        self.sampling_num = sampling_num
        self.sampling_group = sampling_group
        self.sampling_i = 0

    def __call__(self, *args):
        out_list = []
        while(self.sampling_i < self.sampling_num):
            sampling_arg_data = []
            for arg_i, arg_data in enumerate(args):
                if self.sampling_group is None:
                    random_i = np.random.randint(0, len(arg_data)) 
                    sampling_arg_data.append(arg_data[random_i])
                else:
                    data = []
                    for k in range(self.sampling_group[arg_i]):
                        random_i = np.random.randint(0, len(arg_data)) 
                        data.append(arg_data[random_i])
                    assert(len(data) > 0)
                    if len(data) == 1:
                        data = data[0]  
                    sampling_arg_data.append(data)
 
            res = self.func(*sampling_arg_data)
            if res is not None:
                out_list.append(res)
            self.sampling_i += 1

        return out_list