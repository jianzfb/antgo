import os
import sys
import copy
import numpy as np

# usage
# control.DetectOrTracking.xxx[]()

class DetectOrTracking(object):
    def __init__(self, func, interval=1, only_once=False):
        self.func = func
        self.count = 0
        self.interval = interval
        self.only_once = only_once

    def __call__(self, *args):
        assert(len(args)%2 == 0)
        # 运行func条件
        # 1: 首次调用
        # 2: 间隔interval调用
        # 3: args[...] 无效数据
        func_arg_num = len(args) // 2
        is_call_condition = False
        if self.count == 0:
            is_call_condition = True
        elif self.interval > 0 and self.count % self.interval == 0:
            is_call_condition = True
        elif func_arg_num > 0 and args[func_arg_num].shape[0] == 0:
            is_call_condition = True
        elif self.interval == 0:
            is_call_condition = True

        if self.only_once:
            is_call_condition = False

        # 计数+1
        self.count += 1
    
        if is_call_condition:
            return self.func(*args[:func_arg_num])
        return args[func_arg_num:]