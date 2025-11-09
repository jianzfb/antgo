import os
import sys
import copy
import numpy as np

# usage:
# control.If.true_func.xxx.false_func.yyy[(), ()](check_func=None)

class If(object):
    def __init__(self, true_func, false_func, check_func=None, **kwargs):
        self.true_func = true_func
        self.false_func = false_func
        self.check_func = check_func
        self.ext_info = []
        if getattr(true_func, 'info', None):
            self.ext_info.extend(getattr(true_func, 'info')())
        if getattr(false_func, 'info', None):
            self.ext_info.extend(getattr(false_func, 'info')())

    def info(self):
        return self.ext_info

    def __call__(self, *args, **kwargs):
        true_or_false_val = args[0]
        if self.check_func is not None:
            true_or_false_val = self.check_func(true_or_false_val)
        func_args = args[1:] if len(args) > 1 else []
        if bool(true_or_false_val):
            ext_data_dict = {}
            if getattr(self.true_func, 'info', None):
                for ext_data_name in getattr(self.true_func, 'info')():
                    ext_data_dict.update(
                        {
                            ext_data_name: kwargs.get(ext_data_name, None)
                        }
                    )
            return self.true_func(*func_args, **ext_data_dict)
        else:
            ext_data_dict = {}
            if getattr(self.false_func, 'info', None):
                for ext_data_name in getattr(self.false_func, 'info')():
                    ext_data_dict.update(
                        {
                            ext_data_name: kwargs.get(ext_data_name, None)
                        }
                    )
            return self.false_func(*func_args, **ext_data_name)
