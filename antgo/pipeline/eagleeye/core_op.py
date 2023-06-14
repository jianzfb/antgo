import os
import sys
import copy
from typing import Any
ANTGO_DEPEND_ROOT = os.environ.get('ANTGO_DEPEND_ROOT', '/workspace/.3rd')
sys.path.append(f'{ANTGO_DEPEND_ROOT}/eagleeye/py/libs/X86-64')


import eagleeye
import numpy as np


class CoreOp(object):
    def __init__(self, func_op_name, **kwargs):
        if '_' in func_op_name:
            a,b = func_op_name.split('_')
            func_op_name = f'{a.capitalize()}{b.capitalize()}'
        self.func_op_name = func_op_name
        
        self.param_1 = dict()   # {"key": [float,float,float,...]}
        self.param_2 = dict()   # {"key": ["","","",...]}
        self.param_3 = dict()   # {"key": [[float,float,...],[],...]}

        for var_key, var_value in kwargs.items():
            if isinstance(var_value, list):
                if len(var_value) > 0:
                    if isinstance(var_value[0], str):
                        self.param_2[var_key] = var_value
                    elif isinstance(var_value[0], list):
                        temp = np.array(var_value).astype(np.float32).tolist()
                        if len(temp.shape) == 1:
                            self.param_3[var_key] = temp
                    else:
                        self.param_1[var_key] = np.array(var_value).astype(np.float32).tolist()
            elif isinstance(var_value, np.ndarray):
                if len(var_value.shape) == 1:
                    self.param_1[var_key] = var_value.astype(np.float32).tolist()
                elif len(var_value.shape) == 2:
                    self.param_3[var_key] = var_value.astype(np.float32).tolist()
                else:
                    print(f'Dont support {var_key}')
                    print(var_value)

    def __call__(self, *args):
        input_tensors = []
        for tensor in args:
            assert(isinstance(tensor, np.ndarray))
            input_tensors.append(tensor)

        output_tensors = eagleeye.execute(self.func_op_name, self.func_op_name,self.param_1, self.param_2,self.param_3, input_tensors)
        return output_tensors if len(output_tensors) > 1 else output_tensors[0]