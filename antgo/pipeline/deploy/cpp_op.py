import os
from antgo.pipeline import extent
from antgo.pipeline.extent.glue.common import *
import copy

 
class CppOp(object):
    def __init__(self, *args, func_op_name, **kwargs):
        # get cpp func
        self.func_op_name = func_op_name
        self.func = getattr(extent.func, func_op_name)

        # parameter
        self.args = list(args) + [None] * (len(self.func.func.arg_names)-len(args))
        for i, (var_name, var_type) in enumerate(zip(self.func.func.arg_names, self.func.func.arg_types)):
            if kwargs.get(var_name, None) is not None and var_type.is_const:
                self.args[i] = kwargs.get(var_name)

    def __call__(self, *args):
        # 输入数据部分
        func_args = copy.deepcopy(self.args)

        # 绑定上游数据 (顺序绑定)
        input_i = 0
        for i, var_type in enumerate(self.func.func.arg_types):
            if var_type.is_const and func_args[i] is None and input_i < len(args):
                func_args[i] = args[input_i]
                input_i += 1

        # 输出数据(仅用于占位)
        type_map = {
            'CDTensor': np.float64,
            'CFTensor': np.float32,
            'CITensor': np.int32,
            'CUCTensor': np.uint8
        }
        out_placeholders = [np.empty((1), type_map[self.func.func.arg_types[i].cname.replace('*', '')]) \
                for i in range(len(self.func.func.arg_types)) if not self.func.func.arg_types[i].is_const]

        # 绑定输出数据（顺序绑定）
        output_i = 0
        for i, var_type in enumerate(self.func.func.arg_types):
            if not var_type.is_const and func_args[i] is None and output_i < len(args):
                func_args[i] = out_placeholders[output_i]
                output_i += 1

        # 检查函数参数
        for func_arg in func_args:
            if func_arg is None:
                print(f'cpp func {self.func_op_name} arg abnormal.')
                print(func_args)
                break

        # 运行
        output_data = self.func(*func_args)

        # 返回结果
        return tuple(output_data) if len(output_data) > 1 else output_data[0]
