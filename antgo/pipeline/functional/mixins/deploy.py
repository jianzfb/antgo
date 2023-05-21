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
import re


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
    include_folder = os.path.join(output_folder, 'extent','include')
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
    
    # 从Tensor到C*Tensor转换
    convert_map_1 = {
        'CFTensor': 'convert_tensor_cftensor',
        'CITensor': 'convert_tensor_citensor',
        'CUCTensor':'convert_tensor_cuctensor'
    }
    # 从C*Tensor到Tensor转换
    convert_map_2 = {
        'CFTensor': 'convert_cftensor_tensor',
        'CITensor': 'convert_citensor_tensor',
        'CUCTensor':'convert_cuctensor_tensor'
    }
    
    # 创建C*Tensor    
    new_map = {
        'CFTensor': 'new_cftensor',
        'CITensor': 'new_citensor',
        'CUCTensor':'new_cuctensor'
    }
    
    # 初始化C*Tensor
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
            inc_fname=os.path.abspath(func.func.loader_kwargs['cpp_info'].cpp_fname),
            args_convert=args_convert,
            args_inst=args_inst,
            return_statement='',
            output_covert=output_covert,
            args_clear=args_clear
        )

    src_folder = os.path.join(output_folder, 'extent', 'src')
    os.makedirs(src_folder, exist_ok=True)
    with open(os.path.join(src_folder, f'{op_name}_op_warp.cpp'), 'w') as fp:
        fp.write(eagleeye_warp_cpp_code_content)

    info = {
        'input': input_ctx,
        'output': output_ctx,
        'args': None,
        'include': os.path.join('extent','include', f'{op_name}_op_warp.h')
    }
    return info


def convert_args_eagleeye_op_args(op_args):
    converted_op_args = {}
    for arg_name, arg_info in op_args.items():
        if isinstance(arg_info, np.ndarray):
            # numpy
            converted_op_args[arg_name] = arg_info.flatten().astype(np.float32)
        elif isinstance(arg_info, list) or isinstance(arg_info, tuple):
            # list
            converted_op_args[arg_name] = np.array(arg_info).flatten().astype(np.float32)
        else:
            # scalar
            converted_op_args[arg_name] = [float(arg_info)]
    return converted_op_args


def android_package_build(output_folder, eagleeye_path, project_info):
    graph_config = get_graph_info()
    
    core_op_set = load_eagleeye_op_set(eagleeye_path)

    # 准备算子函数代码
    deploy_graph_info = {}
    for graph_op_info in graph_config:
        op_name = graph_op_info['op_name']
        op_index = graph_op_info['op_index']
        op_args = graph_op_info['op_args']
        op_kwargs = graph_op_info['op_kwargs']

        input_ctx, output_ctx = op_index
        if op_name.startswith('deploy'):
            # 需要独立编译
            # 1.step 生成eagleeye算子封装
            op_info = auto_generate_eagleeye_op(op_name[7:], op_index, op_args, op_kwargs, output_folder)
            deploy_graph_info[op_name[7:]] = op_info
        elif op_name.startswith('eagleeye'):
            # eagleeye核心算子 (op级别算子)

            if op_name.endswith('_op'):
                op_name = op_name.capitalize()                
                op_name = op_name.replace('_op', 'Op')

            if op_name not in core_op_set:
                print(f'{op_name} not support')
                print(f'eagleeye core op set include {core_op_set.keys()}')
                return False

            # op_args， 包含scalar, numpy, list类型，转换成std::vector<float>类型
            deploy_graph_info[op_name] = {
                'input': input_ctx,
                'output': output_ctx,
                'args': convert_args_eagleeye_op_args(op_args),
                'include': core_op_set[op_name]['include']
            }
        else:
            # eagleey核心算子 （op级别算子）
            if op_name.endswith('_op'):
                op_name = op_name.capitalize()                     
                op_name = op_name.replace('_op', 'Op')

            if op_name not in core_op_set:
                print(f'{op_name} not support')
                print(f'eagleeye core op set include {core_op_set.keys()}')
                return False

            # op_args， 包含scalar, numpy, list类型，转换成std::vector<float>类型
            deploy_graph_info[op_name] = {
                'input': input_ctx,
                'output': output_ctx,
                'args': convert_args_eagleeye_op_args(op_args),
                'include': core_op_set[op_name]['include']
            }

    # 准备插件文件代码
    # 包括任务管线建立, nano算子图，任务输入信号设置，任务输出信号设置
    
    # 更新CMakeLists.txt
    
    # 编译插件工程

    print('sdf')


def linux_package_build(output_folder):
    pass


NANO_OP_REG = re.compile('^class\s*\w*\s*:')

def load_eagleeye_op_set(eagleeye_path):
    nano_op_path = os.path.join(eagleeye_path, 'include', "eagleeye", "engine", "nano", "op")
    op_set = {}
    for file_name in os.listdir(nano_op_path):
        if file_name.endswith('.h'):
            # 解析算子类名称
            for line in open(os.path.join(nano_op_path, file_name)):
                match = NANO_OP_REG.search(line)
                if match is not None:
                    op_name = match.group()[:-1]
                    op_name = re.split(r'\s+', op_name)[1]
                    op_set[op_name]={
                        'include': os.path.join('eagleeye', 'engine', 'nano', 'op', file_name)
                    }
    return op_set


class DeployMixin:
    def build(self, 
              platform='android/arm64-v8a',
              output_folder='./deploy', 
              project_name='demo', 
              project_version='1.0.0.0', 
              project_git=None,
              eagleeye_path='./eagleeye/install',
              ):
        system_platform, abi_platform = platform.split('/')
        
        # 创建工程
        os.makedirs(output_folder, exist_ok=True)  
        # if project_git is not None:
        #     pass
        # else:
        #     os.system(f'cd {output_folder} && eagleeye-cli project --project={project_name} --version={project_version} --signature=xxxxx --build_type=Release --abi={abi_platform} --eagleeye={eagleeye_path}')
        # # 
        # output_folder = os.path.join(output_folder, project_name)

        # 编译        
        # 1.step 基于不同平台对CPP算子编译，并生成静态库
        if system_platform == 'android':
            android_package_build(output_folder, eagleeye_path, project_info={})
        elif system_platform == 'linux':
            linux_package_build(output_folder, eagleeye_path, project_info={})
        else:
            return False

        # 2.step 编译eagleeye 插件库，并关联上面的静态库

        # print(graph_config)
        return True
