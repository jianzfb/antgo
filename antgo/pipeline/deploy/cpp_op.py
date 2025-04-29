import os
from antgo.pipeline import extent
from antgo.pipeline.extent.glue.common import *
from antgo.pipeline.extent.func import CFuncDef
import copy

 
class CppOp(object):
    def __init__(self, *args, func_op_name, **kwargs):
        # get cpp func
        self.func_op_name = func_op_name
        self.func = getattr(extent.func, func_op_name)()
        self.func_kind = self.func.func.func_kind
        # 上下游绑定名称
        self._index = []

        # parameter
        self.kwargs = {}
        self.args = ()
        if self.func_kind == CFuncDef.FUNC:
            # 函数类型算子
            self.args = list(args) + [None] * (len(self.func.func.arg_names)-len(args))
            for i, (var_name, var_type) in enumerate(zip(self.func.func.arg_names, self.func.func.arg_types)):
                if kwargs.get(var_name, None) is not None and var_type.is_const:
                    self.args[i] = kwargs.get(var_name)
        else:
            # 类类型算子
            # self.kwargs = kwargs
            # self.args = [None] * len(self.func.func.arg_names)
            self.args = list(args) + [None] * (len(self.func.func.arg_names)-len(args))
            for i, (var_name, var_type) in enumerate(zip(self.func.func.arg_names, self.func.func.arg_types)):
                if kwargs.get(var_name, None) is not None and var_type.is_const:
                    self.args[i] = kwargs.get(var_name)
                    kwargs.pop(var_name)
            
            self.kwargs = kwargs

    def __call__(self, *args):
        # 输入数据部分
        func_args = copy.deepcopy(self.args)

        # 绑定上游数据 (顺序绑定)
        input_i = 0
        for i, var_type in enumerate(self.func.func.arg_types):
            if var_type.is_const and func_args[i] is None and input_i < len(args):
                func_args[i] = args[input_i]
                input_i += 1

        # 输出数据(Tensor<->numpy 类型)(仅用于占位)
        type_map = {
            'CDTensor': np.float64,
            'CFTensor': np.float32,
            'CITensor': np.int32,
            'CUCTensor': np.uint8,
            'CBTensor': np.bool_
        }

        out_placeholders = []
        for i in range(len(self.func.func.arg_types)):
            arg_type = self.func.func.arg_types[i].cname.replace('*', '')
            if arg_type in type_map and not self.func.func.arg_types[i].is_const:
                # numpy 类型
                out_placeholders.append(
                    np.empty((1), type_map[arg_type])
                )
            elif not self.func.func.arg_types[i].is_const:
                # dict 类型（需要自己实现初始化字典）
                out_placeholders.append(None)

        # 绑定输出数据（顺序绑定）
        output_i = 0
        for i, var_type in enumerate(self.func.func.arg_types):
            if not var_type.is_const and func_args[i] is None:
                func_args[i] = out_placeholders[output_i]
                output_i += 1

        # 运行
        for kname, kval in zip(self._index[0][input_i:], args[input_i:]):
            self.kwargs.update({
                kname: kval
            })

        output_data = self.func(*func_args, **self.kwargs)

        # 返回结果
        return tuple(output_data) if len(output_data) > 1 else output_data[0]
