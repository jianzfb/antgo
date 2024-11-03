import os
import sys
import copy
import numpy as np
import pathlib
from antgo.pipeline.eagleeye.build import build_eagleeye_env
# usage:
# control.Cache.xxx[(), ()]()

class Cache(object):
    is_finish_import_eagleeye = False
    def __init__(self, func, **kwargs):
        self.func = func
        self.cache_map = {}
        self.empty = None
        self.check_empty_at_index = kwargs.get('check_empty_at_index', 0)
        self.cache_folder = kwargs.get('writable_path', './')
        self.prefix = kwargs.get('prefix', '')

    def __call__(self, *args, **kwargs):
        # 准备eagleeye环境，并加载
        if not Cache.is_finish_import_eagleeye:
            build_eagleeye_env()
            Cache.is_finish_import_eagleeye = True
        import eagleeye

        cache_key = int(args[0][0])
        if cache_key not in self.cache_map:
            # 尝试加载本地文件
            if os.path.exists(os.path.join(self.cache_folder, f'{self.prefix}_{cache_key}.bin')):
                # 使用eagleeye加载Tensor(兼容于c++)
                tensor_list = eagleeye.load_tensor_list(os.path.join(self.cache_folder, f'{self.prefix}_{cache_key}.bin'))
                self.cache_map[cache_key] = tensor_list[0] if len(tensor_list) == 1 else tensor_list
                return self.cache_map[cache_key]

            # 计算
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

                # 保存到文件
                # 使用eagleeye保存Tensor(兼容于c++)
                if isinstance(val, tuple) or isinstance(val, list):
                    new_val = []
                    for v in val:
                        if v.dtype.name == 'float32':
                            new_val.append(v.astype(np.float32))
                        elif v.dtype.name == 'uint8':
                            new_val.append(v.astype(np.uint8))
                        else:
                            new_val.append(v)

                    eagleeye.save_tensor_list(os.path.join(self.cache_folder, f'{self.prefix}_{cache_key}.bin'), new_val)
                else:
                    if val.dtype.name == 'float32':
                        val = val.astype(np.float32)
                    elif val.dtype.name == 'uint8':
                        val = val.astype(np.uint8)

                    eagleeye.save_tensor_list(os.path.join(self.cache_folder, f'{self.prefix}_{cache_key}.bin'), [val])

        if cache_key in self.cache_map:
            return self.cache_map[cache_key]
        
        return self.empty

