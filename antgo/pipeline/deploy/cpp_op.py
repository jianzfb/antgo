import os
from antgo.pipeline import extent
from antgo.pipeline.extent.glue.common import *

 
class CppOp(object):
    def __init__(self, *args, func_op_name):
        self.args = args
        self.func = getattr(extent.func, func_op_name)

    def __call__(self, *args):
        # 输入数据部分
        # part1: 参数
        input_data = list(self.args)
        # part2: 上游数据
        input_data.extend(list(args))

        # 输出数据部分(仅用于占位)
        type_map = {
            'CFTensor': np.float32,
            'CITensor': np.int32,
            'CUCTensor': np.uint8
        }
        output_data = [np.empty((1), type_map[self.func.func.arg_types[i].cname.replace('*', '')]) for i in self.func.wait_to_write_list]

        # 拼接函数参数
        func_args = input_data + output_data
    
        # 运行
        output_data = self.func(*func_args)

        # 返回结果
        return tuple(output_data) if len(output_data) > 1 else output_data[0]
        
