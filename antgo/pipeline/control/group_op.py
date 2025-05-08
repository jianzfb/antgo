import os
import sys
import copy
import numpy as np
from antgo.pipeline import *
from antgo.utils.cache import *

# usage:
# control.Group.xxx.yyy.zzz[(), ()]()
# control.Group.subgroup[]()
# relation [(),()],[(),()],[(),()]

class Group(object):
    def __init__(self, func_list, arg_list, relation, input, output, **kwargs):
        self.func_list = []
        self.ext_info = []
        for func_op_info, func_arg_info in zip(func_list, arg_list):
            func_op = func_op_info(*func_arg_info)
            func_name = func_op.__class__.__name__ 
            func_op.__dict__.update(kwargs.get(func_name, {}))
            self.func_list.append(func_op)

            if getattr(func_op, 'info', None):
                 self.ext_info.extend(getattr(func_op, 'info')())

        self.relation = relation
        self.input_map = {}
        self.input = None
        if input is not None:
            self.input = input if isinstance(input, tuple) else (input,)
            for i, v in enumerate(self.input):
                self.input_map[v] = i

        self.output_map = {}
        self.output = None
        if output is not None:
            self.output = output if isinstance(output, tuple) else (output,)
            for i, v in enumerate(self.output):
                self.output_map[v] = i

    def info(self):
        return self.ext_info

    def __call__(self, *args, **kwargs):
        input_args = []
        if self.input is not None:
            # 管线需要输入,(1) args, (2) cache
            input_args = list(args)
            offset = min(len(self.input), len(args))
            for input_tag in self.input[offset:]:
                input_data = get_data_in_cache("default", input_tag, None)
                input_args.append(input_data)
            args = input_args

        inner_data_dict = {}
        for input_output_info, func_op in zip(self.relation, self.func_list):
            input_info, output_info = input_output_info
            # if isinstance(input_output_info, tuple) and len(input_output_info) == 2:
            #     input_info, output_info = input_output_info
            # elif isinstance(input_output_info, str) or (isinstance(input_output_info, tuple) and len(input_output_info) == 1):
            #     input_info = None
            #     output_info = input_output_info
    
            # input_info, output_info = input_output_info
            # TODO, 需要验证
            # func_op._index = input_info if isinstance(input_info, tuple) else (input_info,)
            func_op._index = input_output_info
            input_data_list = []
            if input_info is not None:
                # 算子可能是无输入计算
                if isinstance(input_info, tuple):
                    for input_tag in input_info:
                        if input_tag in inner_data_dict:
                            # 来自group产生的内部节点数据
                            input_data_list.append(inner_data_dict[input_tag])
                        else:
                            # 来自外部输入的数据
                            input_data_list.append(args[self.input_map[input_tag]])
                else:
                    if input_info in inner_data_dict:
                        # 来自group产生的内部节点数据
                        input_data_list.append(inner_data_dict[input_info])
                    else:
                        # 来自外部输入的数据
                        input_data_list.append(args[self.input_map[input_info]])

            ext_data_dict = {}
            if getattr(func_op, 'info', None):
                for ext_data_name in getattr(func_op, 'info')():
                    ext_data_dict.update(
                        {
                            ext_data_name: kwargs.get(ext_data_name, None)
                        }
                    )

            output_data_list = func_op(*input_data_list, **ext_data_dict)
            if isinstance(output_info, tuple):
                for output_tag, output_data in zip(output_info, output_data_list):
                    inner_data_dict[output_tag] = output_data
            else:
                inner_data_dict[output_info] = output_data_list

        output = []
        for output_i in range(len(self.output)):
            output_data = inner_data_dict[self.output[output_i]]
            # 记录到输出
            output.append(output_data)
            # 记录到缓存（跨管线使用）
            set_data_in_cache('default', self.output[output_i], output_data)

        return tuple(output) if len(output) > 1 else output[0]