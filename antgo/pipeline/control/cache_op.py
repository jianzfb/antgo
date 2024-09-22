import os
import sys
import copy
import numpy as np
import pathlib

ANTGO_DEPEND_ROOT = os.environ.get('ANTGO_DEPEND_ROOT', f'{str(pathlib.Path.home())}/.3rd')
if not os.path.exists(ANTGO_DEPEND_ROOT):
    os.makedirs(ANTGO_DEPEND_ROOT)

if not os.path.exists(os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye', 'py')):
    if not os.path.exists(os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye')):
        os.system('cd {ANTGO_DEPEND_ROOT} && git clone https://github.com/jianzfb/eagleeye.git')

    if 'darwin' in sys.platform:
        os.system(f'cd {ANTGO_DEPEND_ROOT}/eagleeye && bash osx_build.sh BUILD_PYTHON_MODULE')
    else:
        first_comiple = False
        if not os.path.exists(os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye','py')):
            first_comiple = True
        os.system(f'cd {ANTGO_DEPEND_ROOT}/eagleeye && bash linux_x86_64_build.sh BUILD_PYTHON_MODULE')
        if first_comiple:
            # 增加搜索.so路径
            cur_abs_path = os.path.abspath(os.curdir)
            so_abs_path = os.path.join(cur_abs_path, f"{ANTGO_DEPEND_ROOT}/eagleeye/py/libs/x86-64")
            os.system(f'echo "{so_abs_path}" >> /etc/ld.so.conf && ldconfig')

if f'{ANTGO_DEPEND_ROOT}/eagleeye/py/libs/x86-64' not in sys.path:
    sys.path.append(f'{ANTGO_DEPEND_ROOT}/eagleeye/py/libs/x86-64')
import eagleeye

# usage:
# control.Cache.xxx[(), ()]()

class Cache(object):
    def __init__(self, func, **kwargs):
        self.func = func
        self.cache_map = {}
        self.empty = None
        self.check_empty_at_index = kwargs.get('check_empty_at_index', 0)
        self.cache_folder = kwargs.get('writable_path', './')
        self.prefix = kwargs.get('prefix', '')

    def __call__(self, *args, **kwargs):
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

