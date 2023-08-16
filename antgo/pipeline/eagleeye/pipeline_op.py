import os
import sys
import copy
from typing import Any
import numpy as np
import importlib

class CorePipeline(object):
    def __init__(self, func_op_name, plugin_name=None, **kwargs):
        self.pipeline_name = func_op_name
        self.plugin_name = plugin_name
        self.plugin_id = str(uuid.uuid4())
        if self.plugin_name is None:
            # 自动查找当前文件夹下的plugin
            for subfolder in os.listdir('./deploy'):
                if subfolder.endswith('_plugin'):
                    self.plugin_name = subfolder.replace('_plugin', '')
                    break

        so_abs_path = os.path.join(f'./deploy/{self.plugin_name}_plugin', 'bin', 'X86-64')
        so_abs_path = os.path.abspath(so_abs_path)
        sys.path.append(so_abs_path)

        self.plugin_module = importlib.import_module(self.plugin_name)
        self.param = kwargs

    def __call__(self, *args):
        input_tensors = []
        for tensor in args:
            assert(isinstance(tensor, np.ndarray))
            input_tensors.append(tensor)

        output_tensors = getattr(self.plugin_module, 'pipeline_execute')(self.pipeline_name, f'{self.pipeline_name}Pipeline', self.param, input_tensors)
        return output_tensors if len(output_tensors) > 1 else output_tensors[0]
