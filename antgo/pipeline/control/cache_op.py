import os
import sys
import copy
import numpy as np

# usage:
# control.Cache.xxx[(), ()]()

class Cache(object):
    def __init__(self, func, **kwargs):
        self.func = func
        self.cache_map = {}
        self.empty = None
        self.check_empty_at_index = kwargs.get('check_empty_at_index', 0)

    def __call__(self, *args, **kwargs):
        cache_key = int(args[0][0])
        if cache_key not in self.cache_map:
            func_args = args[1:]
            val = self.func(*func_args, **kwargs)

            # 验证返回结果的有效性
            success = False
            if isinstance(val, tuple) or isinstance(val, list):
                success = val[self.check_empty_at_index].shape[0] > 0
                if self.empty is None:
                    self.empty = []
                    for v in val:
                        self.empty.append(np.empty((0), dtype=v.dtype))
                    self.empty = tuple(self.empty)
            else:
                success = val.shape[0] > 0
                if self.empty is None:
                    self.empty = np.empty((0), dtype=val.dtype)

            if success:
                self.cache_map[cache_key] = val

        if cache_key in self.cache_map:
            return self.cache_map[cache_key]
        
        return self.empty

