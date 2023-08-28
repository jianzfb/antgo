import os
import sys
import copy
from typing import Any
ANTGO_DEPEND_ROOT = os.environ.get('ANTGO_DEPEND_ROOT', '/workspace/.3rd')
sys.path.append(f'{ANTGO_DEPEND_ROOT}/eagleeye/py/libs/X86-64')

import eagleeye
import numpy as np
import uuid


class CoreNode(object):
    def __init__(self, func_op_name, **kwargs):
        self.node_name = func_op_name
        self.node_id = str(uuid.uuid4())
        self.param = kwargs

    def __call__(self, *args):
        input_tensors = []
        for tensor in args:
            assert(isinstance(tensor, np.ndarray) or isinstance(tensor, bool) or isinstance(tensor, str))
            input_tensors.append(tensor)

        output_tensors = eagleeye.node_execute(self.node_name,  self.node_id, self.node_name,  self.param, input_tensors)
        return output_tensors if len(output_tensors) > 1 else output_tensors[0]