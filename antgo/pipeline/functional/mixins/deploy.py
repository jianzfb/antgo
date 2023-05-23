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
            op_name=f"{op_name.replace('_','').capitalize()}Op",
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
            
            args_clear += f'{arg_name}->destroy();\ndelete {arg_name};\n'
        elif flag.startswith('output'):
            arg_type = arg_type.cname.replace('*','')
            args_convert += f'{arg_type}*{arg_name}={new_map[arg_type]}();\n'
            
            output_p = int(flag[7:])
            output_covert += f'm_outputs[{output_p}]={convert_map_2[arg_type]}({arg_name});\n'
            args_clear += f'{arg_name}->destroy();\ndelete {arg_name};\n'
        else:
            if isinstance(arg_value, np.ndarray):
                assert(arg_value.dtype == np.float32 or arg_value.dtype == np.int32 or arg_value.dtype == np.uint8)
                
                data_value_str = '{'+','.join([f'{v}' for v in arg_value.tolist()])+'}'
                data_shape_str = '{'+','.join([f'{v}' for v in arg_value.shape])+'}'
                arg_type = arg_type.cname.replace('*','')
                args_convert += f'{arg_type}*{arg_name}={init_map[arg_value.dtype.str]}({data_value_str}, {data_shape_str});\n'
                
                args_clear += f'{arg_name}->destroy();\ndelete {arg_name};\n'
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
            op_name=f"{op_name.replace('_','').capitalize()}Op",
            func_name=op_name,
            inc_fname1=os.path.abspath(os.path.join(output_folder, 'extent', 'include', f'{op_name}_op_warp.h')),
            inc_fname2=os.path.abspath(func.func.loader_kwargs['cpp_info'].cpp_fname),
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
        'type': f"{op_name.replace('_','').capitalize()}Op",
        'input': input_ctx,
        'output': output_ctx,
        'args': {},
        'include': os.path.join('extent','include', f'{op_name}_op_warp.h'),
        'src': os.path.join('./', 'extent', 'src', f'{op_name}_op_warp.cpp')
    }
    return info


def convert_args_eagleeye_op_args(op_args, op_kwargs):
    # ignore op_args
    converted_op_args = {}
    for arg_name, arg_info in op_kwargs.items():
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


def update_cmakelist(output_folder, project_name, src_op_warp_list):
    info = []
    is_found_include_directories_insert = False
    is_start_add_src_code = False

    src_op_warp_flag = [0 for _ in range(len(src_op_warp_list))]
    for line in open(os.path.join(output_folder, 'CMakeLists.txt')):
        if is_start_add_src_code and not line.strip().endswith(')'):
            for src_op_warp_file_i, src_op_warp_file in enumerate(src_op_warp_list):
                if src_op_warp_file == line.strip():
                    src_op_warp_flag[src_op_warp_file_i] = 1
                    break
        if is_start_add_src_code and line.strip().endswith(')'):
            for src_op_warp_file_i, src_op_warp_file_flag in enumerate(src_op_warp_flag):
                if src_op_warp_file_flag == 0:
                    info.append(f'{src_op_warp_list[src_op_warp_file_i]}\n')

            is_start_add_src_code = False

        if f'set({project_name}_SRC' in line:
            is_start_add_src_code = True

        if line.startswith('include_directories') and not is_found_include_directories_insert:
            extent_cpp_include_folder = os.path.dirname(os.path.realpath(__file__))
            extent_cpp_include_folder = os.path.dirname(extent_cpp_include_folder)
            extent_cpp_include_folder = os.path.dirname(extent_cpp_include_folder)
            extent_cpp_include_folder = os.path.join(extent_cpp_include_folder, 'extent','cpp','include')
            if extent_cpp_include_folder not in line:
                info.append(f'include_directories({extent_cpp_include_folder})\n')

            is_found_include_directories_insert = True

        info.append(line)
    
    with open(os.path.join(output_folder, 'CMakeLists.txt'), 'w') as fp:
        for line in info:
            fp.write(line)


def package_build(output_folder, eagleeye_path, project_config, platform):    
    # 获得eagleeye核心算子集合
    core_op_set = load_eagleeye_op_set(eagleeye_path)

    # 获得计算图配置信息
    graph_config = get_graph_info()['op']

    # 准备算子函数代码
    deploy_graph_info = {}
    op_name_count = {}
    for graph_op_info in graph_config:
        op_name = graph_op_info['op_name']
        op_index = graph_op_info['op_index']
        op_args = graph_op_info['op_args']
        op_kwargs = graph_op_info['op_kwargs']

        input_ctx, output_ctx = op_index
        if not isinstance(input_ctx, tuple):
            input_ctx = (input_ctx,)
        if not isinstance(output_ctx, tuple):
            output_ctx = (output_ctx,)

        if op_name.startswith('deploy'):
            # 需要独立编译
            # 1.step 生成eagleeye算子封装
            op_name = op_name[7:]
            if op_name not in op_name_count:
                op_name_count[op_name] = 0
            op_unique_name = f'{op_name}_{op_name_count[op_name]}'
            op_name_count[op_name] += 1

            op_info = auto_generate_eagleeye_op(op_name, op_index, op_args, op_kwargs, output_folder)
            deploy_graph_info[op_unique_name] = op_info
        elif op_name.startswith('eagleeye'):
            # eagleeye核心算子 (op级别算子)
            if op_name not in op_name_count:
                op_name_count[op_name] = 0
            op_unique_name = f'{op_name}_{op_name_count[op_name]}'
            op_name_count[op_name] += 1

            if op_name.endswith('_op'):
                op_name = op_name.capitalize()                
                op_name = op_name.replace('_op', 'Op')

            if op_name not in core_op_set:
                print(f'{op_name} not support')
                print(f'eagleeye core op set include {core_op_set.keys()}')
                return False

            # op_args， 包含scalar, numpy, list类型，转换成std::vector<float>类型
            deploy_graph_info[op_unique_name] = {
                'type': op_name,
                'input': input_ctx,
                'output': output_ctx,
                'args': convert_args_eagleeye_op_args(op_args, op_kwargs),
                'include': core_op_set[op_name]['include']
            }
        else:
            # eagleey核心算子 （op级别算子）
            if op_name not in op_name_count:
                op_name_count[op_name] = 0
            op_unique_name = f'{op_name}_{op_name_count[op_name]}'
            op_name_count[op_name] += 1

            if op_name.endswith('_op'):
                op_name = op_name.capitalize()                     
                op_name = op_name.replace('_op', 'Op')

            if op_name not in core_op_set:
                print(f'{op_name} not support')
                print(f'eagleeye core op set include {core_op_set.keys()}')
                return False

            # op_args， 包含scalar, numpy, list类型，转换成std::vector<float>类型
            deploy_graph_info[op_unique_name] = {
                'type': op_name,
                'input': input_ctx,
                'output': output_ctx,
                'args': convert_args_eagleeye_op_args(op_args, op_kwargs),
                'include': core_op_set[op_name]['include']
            }

    # 准备插件文件代码
    # 包括任务管线建立, nano算子图，任务输入信号设置，任务输出信号设置
    t = [v['include'] for v in deploy_graph_info.values()]
    include_list = ''
    for f in t:
        include_list += f'#include "{f}"\n'

    op_graph_code = ''
    deploy_output_data_name_inv_link = {}
    for deploy_op_name, deploy_op_info in deploy_graph_info.items():
        op_graph_code += f'dataflow::Node* {deploy_op_name} = op_graph->add("{deploy_op_name}", {deploy_op_info["type"]}(), EagleeyeRuntime(EAGLEEYE_CPU));\n'

        deploy_op_args = deploy_op_info['args']
        arg_code = ''
        for deploy_arg_name, deploy_arg_list in deploy_op_args.items():
            if arg_code == '':
                arg_code = '{"'+deploy_arg_name+'",{'+','.join([str(v) for v in deploy_arg_list])+'}}'
            else:
                arg_code += ',{"'+deploy_arg_name+'",{'+','.join([str(v) for v in deploy_arg_list])+'}}'

        args_init_code = '{'+arg_code+'}'
        op_graph_code += f'{deploy_op_name}->init({args_init_code});\n\n'

        print(deploy_op_info['output'])
        for data_i, data_name in enumerate(deploy_op_info['output']):
            deploy_output_data_name_inv_link[data_name] = (deploy_op_name, data_i)

    for deploy_op_name, deploy_op_info in deploy_graph_info.items():
        if deploy_op_info['input'] is not None:
            for input_data_i, input_data_name in enumerate(deploy_op_info['input']):
                if input_data_name is not None:
                    from_op_name, from_op_out_i = deploy_output_data_name_inv_link[input_data_name]
                    op_graph_code += f'op_graph->bind("{from_op_name}", {from_op_out_i}, "{deploy_op_name}", {input_data_i});\n'

    op_graph_code += 'op_graph->init(NULL);'

    eagleeye_plugin_code_content = \
        gen_code('./templates/plugin_code.cpp')(            
            project=project_config['name'],
            version=project_config.get('version', '1.0.0.0'),
            signature=project_config.get('signature', 'xxx'),
            include_list=include_list,
            in_port='{'+','.join([str(i) for i in range(len(project_config['input']))]) + '}',
            in_signal='{'+','.join(['"'+info[-1]+'"' for info in project_config['input']])+'}',
            out_port='{'+','.join([str(i) for i in range(len(project_config['output']))]) + '}',
            out_signal='{'+','.join(['"'+info[-1]+'"' for info in project_config['output']])+'}',
            graph_in_ops='{'+','.join(['"'+deploy_output_data_name_inv_link[info[0]][0]+'"' for info in project_config['input']])+'}',
            graph_out_ops='{'+','.join(['"'+deploy_output_data_name_inv_link[info[0]][0]+'"' for info in project_config['output']])+'}',
            op_graph=op_graph_code
        )

    with open(os.path.join(output_folder, f'{project_config["name"]}_plugin.cpp'), 'w') as fp:
        fp.write(eagleeye_plugin_code_content)

    # 更新CMakeLists.txt
    update_cmakelist(output_folder, project_config["name"], [s['src'] for s in deploy_graph_info.values() if 'src' in s])

    # 编译插件工程
    shell_code_content = gen_code('./templates/android_build.sh')(
        project=project_config['name'],
        ANDROID_NDK_HOME=os.environ['ANDROID_NDK_HOME']
    )
    with open(os.path.join(output_folder, 'android_build.sh'), 'w') as fp:
        fp.write(shell_code_content)

    shell_code_content = gen_code('./templates/linux_build.sh')(
        project=project_config['name']
    )
    with open(os.path.join(output_folder, 'linux_build.sh'), 'w') as fp:
        fp.write(shell_code_content)

    os.system(f'cd {output_folder} && bash {platform}_build.sh')


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
              eagleeye_path='./eagleeye/install',
              project_config=None):
        system_platform, abi_platform = platform.split('/')

        # 创建工程
        os.makedirs(output_folder, exist_ok=True)
        if not os.path.exists(os.path.join(output_folder, f'{project_config["name"]}_plugin')):
            if project_config.get('git', None) is not None and project_config['git'] != '':
                pass
            else:
                os.system(f'cd {output_folder} && eagleeye-cli project --project={project_config["name"]} --version={project_config.get("version", "1.0.0.0")} --signature=xxxxx --build_type=Release --abi={abi_platform} --eagleeye={eagleeye_path}')
        output_folder = os.path.join(output_folder, f'{project_config["name"]}_plugin')

        # 编译        
        # 1.step 基于不同平台对CPP算子编译，并生成静态库
        package_build(
            output_folder, 
            eagleeye_path, 
            project_config=project_config, platform=system_platform)

        return True
