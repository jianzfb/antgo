import os
import sys
import copy
import numpy as np

# usage:
# control.Group.xxx.yyy.zzz[(), ()]()
# control.Group.subgroup[]()
# relation [(),()],[(),()],[(),()]

class Group(object):
    def __init__(self, func_list, relation, input, output, **kwargs):
        self.func_list = func_list
        for func_op in self.func_list:
            # 更新参数
            func_name = func_op.__class__.__name__ 
            func_op.__dict__.update(kwargs.get(func_name, {}))

        self.relation = relation
        self.input_map = {}
        self.input = input
        for i, v in enumerate(input):
            self.input_map[v] = i

        self.output_map = {}
        self.output = output
        for i, v in enumerate(output):
            self.output_map[v] = i

        # self.group_out = []
        # for input_output_info in self.relation:
        #     input_info, output_info = input_output_info
        #     if isinstance(output_info, tuple):
        #         for output_tag in output_info:
        #             if output_tag.isnumeric():                    
        #                 self.group_out.append(output_tag)
        #     else:
        #         if output_info.isnumeric():
        #             self.group_out.append(output_info)

    def __call__(self, *args):
        inner_data_dict = {}
        for input_output_info, func_op in zip(self.relation, self.func_list):
            input_info, output_info = input_output_info
            # TODO, 需要验证
            # func_op._index = input_info if isinstance(input_info, tuple) else (input_info,)
            func_op._index = input_output_info
            input_data_list = []
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

            output_data_list = func_op(*input_data_list)
            if isinstance(output_info, tuple):
                for output_tag, output_data in zip(output_info, output_data_list):
                    inner_data_dict[output_tag] = output_data
            else:
                inner_data_dict[output_info] = output_data_list

        output = []
        for output_i in range(len(self.output)):
            output.append(inner_data_dict[self.output[output_i]])

        return tuple(output) if len(output) > 1 else output[0]