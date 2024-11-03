import os
import sys
import copy
import numpy as np
import uuid
from .build import build_eagleeye_env


class CoreNode(object):
    is_finish_import_eagleeye = False
    def __init__(self, func_op_name, **kwargs):
        self.node_name = func_op_name
        self.node_id = str(uuid.uuid4())
        self.param = kwargs

    def __call__(self, *args):
        # 准备eagleeye环境，并加载
        if not CoreNode.is_finish_import_eagleeye:
            build_eagleeye_env()
            CoreNode.is_finish_import_eagleeye = True
        import eagleeye

        input_tensors = []
        for tensor in args:
            assert(isinstance(tensor, np.ndarray) or isinstance(tensor, bool) or isinstance(tensor, str))
            input_tensors.append(tensor)

        output_tensors = eagleeye.node_execute(self.node_name,  self.node_id, self.node_name,  self.param, input_tensors)
        return output_tensors if len(output_tensors) > 1 else output_tensors[0]