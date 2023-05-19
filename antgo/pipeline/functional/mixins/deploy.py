# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 21:56
# @File    : deploy.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import os
import json
from antgo.pipeline.functional.common.config import *
from antgo.pipeline import extent
from antgo.pipeline.extent.glue.common import *
from antgo.pipeline.extent.op.loader import *


def auto_generate_eagleeye_op(op_name, op_index, op_args, op_kwargs, output_folder):
    func = getattr(extent.func, op_name)

    input_ctx, output_ctx = op_index
    if isinstance(input_ctx, str):
        input_ctx = [input_ctx]
    if isinstance(output_ctx, str):
        output_ctx = [output_ctx]
    
    # 创建header文件
    eagleeye_warp_h_code_content = \
        gen_code('./templates/op_code.h')(            
            op_name=f'{op_name.capitalize()}',
            input_num=len(input_ctx),
            output_num=len(output_ctx)
        )
    
    # folder tree
    # output_folder/
    #   include/
    #   src/
    include_folder = os.path.join(output_folder, 'include')
    os.makedirs(include_folder, exist_ok=True)
    with open(os.path.join(include_folder, f'{op_name}_op_warp.h'), 'w') as fp:
        fp.write(eagleeye_warp_h_code_content)

    # 创建cpp文件
    # 函数参数部分
    func_args = [None] * len(func.func.arg_names)
    for i, (var_name, var_type) in enumerate(zip(func.func.arg_names, func.func.arg_types)):
        if i < len(op_args):
            func_args[i] =  ('param', var_name, var_type, op_args[i])

    for i, (var_name, var_type) in enumerate(zip(func.func.arg_names, func.func.arg_types)):
        if op_kwargs.get(var_name, None) is not None and var_type.is_const:
            func_args[i] = ('param', var_name, var_type, op_kwargs.get(var_name))
    
    # 函数输入部分
    input_i = 0
    for i, (var_name, var_type) in enumerate(zip(func.func.arg_names, func.func.arg_types)):
        if var_type.is_const and func_args[i] is None:
            func_args[i] = (f'input_{input_i}',var_name, var_type, None)
            input_i += 1

    # 函数输出部分
    output_i = 0
    for i, (var_name, var_type) in enumerate(zip(func.func.arg_names, func.func.arg_types)):
        if not var_type.is_const and func_args[i] is None:
            func_args[i] = (f'output_{output_i}', var_name, var_type, None)
            output_i += 1
    
    
    convert_map_1 = {
        'CFTensor': 'convert_tensor_cftensor',
        'CITensor': 'convert_tensor_citensor',
        'CUCTensor':'convert_tensor_cuctensor'
    }
    convert_map_2 = {
        'CFTensor': 'convert_cftensor_tensor',
        'CITensor': 'convert_citensor_tensor',
        'CUCTensor':'convert_cuctensor_tensor'
    }
    
    new_map = {
        'CFTensor': 'new_cftensor',
        'CITensor': 'new_citensor',
        'CUCTensor':'new_cuctensor'
    }
    
    init_map = {
        '<f4': 'init_cftensor',
        '<i4': 'init_citensor',
        '|u1': 'init_cuctensor'
    }
    
    args_convert = ''
    output_covert = ''
    args_clear = ''
    for arg in func_args:
        flag, arg_name, arg_type, arg_value = arg
        if flag.startswith('input'):
            input_p = int(flag[6:])
            arg_type = arg_type.cname.replace('*','')
            args_convert += f'{arg_type}*{arg_name}={convert_map_1[arg_type]}(input[{input_p}]);\n'
            
            args_clear += f'{arg_name}.destroy();\n'
        elif flag.startswith('output'):
            arg_type = arg_type.cname.replace('*','')
            args_convert += f'{arg_type}*{arg_name}={new_map[arg_type]}();\n'
            
            output_p = int(flag[7:])
            output_covert += f'output[{output_p}]={convert_map_2[arg_type]}({arg_name});\n'
            args_clear += f'{arg_name}.destroy();\n'
        else:
            if isinstance(arg_value, np.ndarray):
                assert(arg_value.dtype == np.float32 or arg_value.dtype == np.int32 or arg_value.dtype == np.uint8)
                
                data_value_str = '{'+','.join([f'{v}' for v in arg_value.tolist()])+'}'
                data_shape_str = '{'+','.join([f'{v}' for v in arg_value.shape])+'}'
                arg_type = arg_type.cname.replace('*','')
                args_convert += f'{arg_type}*{arg_name}={init_map[arg_value.dtype.str]}({data_value_str}, {data_shape_str});\n'
                
                args_clear += f'{arg_name}.destroy();\n'
            else:
                arg_type = arg_type.cname.replace('const', '')
                args_convert += f'{arg_type} {arg_name}={str(arg_value).lower()};\n'
                
    args_inst = ''
    for _, arg_name, _, _ in func_args:
        if args_inst == '':
            args_inst = f'{arg_name}'
        else:
            args_inst += f',{arg_name}'
    
    
    eagleeye_warp_cpp_code_content = \
        gen_code('./templates/op_code.cpp')(
            op_name=f'{op_name.capitalize()}',
            func_name=op_name,
            inc_fname='',
            args_convert=args_convert,
            args_inst=args_inst,
            return_statement='',
            output_covert=output_covert,
            args_clear=args_clear
        )
    
    src_folder = os.path.join(output_folder, 'src')
    os.makedirs(src_folder, exist_ok=True)
    with open(os.path.join(src_folder, f'{op_name}_op_warp.cpp'), 'w') as fp:
        fp.write(eagleeye_warp_cpp_code_content)


def android_package_build(output_folder):
    graph_config = get_graph_info()
    # 准备算子函数代码
    for graph_op_info in graph_config:
        op_name = graph_op_info['op_name']
        op_index = graph_op_info['op_index']
        op_args = graph_op_info['op_args']
        op_kwargs = graph_op_info['op_kwargs']

        if op_name.startswith('deploy'):
            # 需要独立编译
            # 1.step 生成eagleeye算子封装
            auto_generate_eagleeye_op(op_name[7:], op_index, op_args, op_kwargs, output_folder)
            print('sdf')
        else:
            # eagleey核心库中存在的算子
            pass
        
    # 使用eagleeye-cli创建插件工程
    
    # 更新cmake
    
    # 编译插件工程

    print('sdf')


def linux_package_build(output_folder):
    pass


class DeployMixin:
    def build(self, platform='android', output_folder='./deploy'):
        # 编译
        # 1.step 基于不同平台对CPP算子编译，并生成静态库
        if platform == 'android':
            android_package_build(output_folder)
        elif platform == 'linux':
            linux_package_build(output_folder)
        else:
            return False

        # 2.step 编译eagleeye 插件库，并关联上面的静态库

        # print(graph_config)
        return True

    def run(self, platform='android'):
        # 编译
        # 运行
        pass