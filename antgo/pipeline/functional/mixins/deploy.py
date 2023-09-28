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
from antgo.pipeline.extent.building.build_utils import *
import onnx
import onnxruntime
import re
import shutil
import subprocess

ANTGO_DEPEND_ROOT = os.environ.get('ANTGO_DEPEND_ROOT', '/workspace/.3rd')

def snpe_import_config(output_folder, project_name, platform, abi, device='GPU'):
    # load snpe lib
    # step1: 下载snpe库, 并解压到固定为止
    root_folder = os.path.abspath(ANTGO_DEPEND_ROOT)
    os.makedirs(root_folder, exist_ok=True)
    if not os.path.exists(os.path.join(root_folder, 'snpe-2.9.0.4462')):
        os.system(f'cd {root_folder} ; wget http://files.mltalker.com/snpe-2.9.0.4462.zip ; unzip snpe-2.9.0.4462.zip')
    snpe_path = os.path.join(root_folder, 'snpe-2.9.0.4462')

    # step2: 推送依赖库到包位置
    os.makedirs(os.path.join(output_folder, '3rd', abi), exist_ok=True)
    shutil.copyfile(os.path.join(f'{snpe_path}/lib/aarch64-android-clang8.0', 'libSNPE.so'), os.path.join(output_folder, '3rd', abi, 'libSNPE.so'))

    # step3: 生成cmake代码片段
    snpe_cmake_code_snippet = ''
    snpe_cmake_code_snippet += f'set(SNPE_LIB_DIR {snpe_path}/lib/aarch64-android-clang8.0)\n'
    snpe_cmake_code_snippet += f'set(SNPE_INCLUDE_DIR {snpe_path}/include/zdl)\n'
    snpe_cmake_code_snippet += 'include_directories(${SNPE_INCLUDE_DIR})\n'
    snpe_cmake_code_snippet += 'add_library(libSNPE SHARED IMPORTED)\n'
    snpe_cmake_code_snippet += 'set_target_properties(libSNPE PROPERTIES IMPORTED_LOCATION ${SNPE_LIB_DIR}/libSNPE.so)\n'
    snpe_cmake_code_snippet += f'target_link_libraries({project_name} libSNPE)\n'

    code_line_list = []
    for line in open(os.path.join(output_folder, 'CMakeLists.txt')):
        if len(code_line_list) > 0 and code_line_list[-1].strip() == '# model engine' and line == '\n':
            code_line_list.append(snpe_cmake_code_snippet)

        code_line_list.append(line)

    with open(os.path.join(output_folder, 'CMakeLists.txt'), 'w') as fp:
        for line in code_line_list:
            fp.write(line)


def tnn_import_config(output_folder, project_name, platform, abi, device=''):
    # step1: 下载tnn库，并编译
    root_folder = os.path.abspath(ANTGO_DEPEND_ROOT)
    os.makedirs(root_folder, exist_ok=True)
    tnn_path = os.path.join(root_folder, 'TNN')
    if not os.path.exists(tnn_path):
        os.system(f'cd {root_folder} ; git clone https://github.com/Tencent/TNN.git')
    if not os.path.exists(os.path.join(tnn_path, 'scripts', 'release')):
        os.system(f'cd {tnn_path}/scripts; export ANDROID_NDK=$ANDROID_NDK_HOME ; bash build_android.sh')

    # step2: 推送依赖库到包为止
    os.makedirs(os.path.join(output_folder, '3rd', abi), exist_ok=True)
    TNN_ROOT_PATH = os.path.join(tnn_path, 'scripts', 'release')
    shutil.copyfile(os.path.join(TNN_ROOT_PATH, abi, 'libTNN.so'), os.path.join(output_folder, '3rd', abi, 'libTNN.so'))

    # step3: 生成cmake代码片段 (include, so)
    tnn_cmake_code_snippet = f'include_directories({TNN_ROOT_PATH}/include)\n'
    tnn_cmake_code_snippet += f'add_library(libtnn SHARED IMPORTED)\n'
    tnn_cmake_code_snippet += f'set_target_properties(libtnn PROPERTIES IMPORTED_LOCATION {TNN_ROOT_PATH}/{abi}/libTNN.so)\n'
    tnn_cmake_code_snippet += f'target_link_libraries({project_name} libtnn)\n'

    code_line_list = []
    for line in open(os.path.join(output_folder, 'CMakeLists.txt')):
        if len(code_line_list) > 0 and code_line_list[-1].strip() == '# model engine' and line == '\n':
            code_line_list.append(tnn_cmake_code_snippet)

        code_line_list.append(line)

    with open(os.path.join(output_folder, 'CMakeLists.txt'), 'w') as fp:
        for line in code_line_list:
            fp.write(line)


def rknn_import_config(output_folder, project_name, platform, abi, device='rk3588'):
    # step1: 下载rknn库，并解压到固定为止
    root_folder = os.path.abspath(ANTGO_DEPEND_ROOT)
    os.makedirs(root_folder, exist_ok=True)
    if not os.path.exists(os.path.join(root_folder, 'rknpu2')):
        os.system(f'cd {root_folder} ; git clone https://github.com/rockchip-linux/rknpu2.git')
    rknn_path = os.path.join(root_folder, 'rknpu2')

    rknn_runtime_folder = os.path.join(rknn_path, 'runtime')
    if device.startswith('rk356'):
        device = 'rk356X'
    assert(os.path.exists(os.path.join(rknn_runtime_folder, device.upper())))
    device_rknn_runtime_folder = os.path.join(rknn_runtime_folder, device.upper())
    assert(os.path.exists(os.path.join(device_rknn_runtime_folder, platform.capitalize())))
    RKNN_API_PATH = os.path.join(device_rknn_runtime_folder, platform.capitalize(), 'librknn_api')
    # device_rknn_so_folder = os.path.join(device_rknn_runtime_folder, platform.capitalize(), 'librknn_api', abi)
    # device_rknn_include_folder = os.path.join(device_rknn_runtime_folder, platform.capitalize(), 'librknn_api', 'include')

    # step2: 推送依赖库到包为止
    os.makedirs(os.path.join(output_folder, '3rd', abi), exist_ok=True)
    shutil.copyfile(os.path.join(RKNN_API_PATH, abi, 'librknnrt.so'), os.path.join(output_folder, '3rd', abi, 'librknnrt.so'))

    # step3: 生成cmake代码片段
    rknn_cmake_code_snippet = f'set(RKNN_API_PATH {RKNN_API_PATH})\n'
    rknn_cmake_code_snippet += f'include_directories({RKNN_API_PATH}/include)\n'
    rknn_cmake_code_snippet += f'set(RKNN_RT_LIB {RKNN_API_PATH}/{abi}/librknnrt.so)\n'
    rknn_cmake_code_snippet += 'add_library(librknnrt SHARED IMPORTED)\n'
    rknn_cmake_code_snippet += 'set_target_properties(librknnrt PROPERTIES IMPORTED_LOCATION ${RKNN_RT_LIB})\n'
    rknn_cmake_code_snippet += f'target_link_libraries({project_name} librknnrt)\n'

    code_line_list = []
    for line in open(os.path.join(output_folder, 'CMakeLists.txt')):
        if len(code_line_list) > 0 and code_line_list[-1].strip() == '# model engine' and line == '\n':
            code_line_list.append(rknn_cmake_code_snippet)

        code_line_list.append(line)

    with open(os.path.join(output_folder, 'CMakeLists.txt'), 'w') as fp:
        for line in code_line_list:
            fp.write(line)


def tensorrt_import_config(output_folder, project_name, platform, abi, device=''):
    # step1: 下载tensorrt库，并解压到固定为止
    root_folder = os.path.abspath(ANTGO_DEPEND_ROOT)
    os.makedirs(root_folder, exist_ok=True)
    
    # tensorrt 仅支持linux
    assert(platform.lower() == 'linux')

    # 涉及cudnn, tensorrt, cuda库
    # 下载 cudnn cudnn-linux-x86_64-8.8.0.121_cuda12-archive.tar.xz https://developer.nvidia.com/rdp/cudnn-archive
    # 下载 tensorrt TensorRT-8.6.1.6 https://developer.nvidia.com/nvidia-tensorrt-8x-download
    if not os.path.exists(os.path.join(root_folder, 'cudnn-linux-x86_64-8.8.0.121_cuda12-archive')):
        print('download cudnn')
        if not os.path.exists(os.path.join(root_folder, 'cudnn-linux-x86_64-8.8.0.121_cuda12-archive.tar.xz')):
            os.system(f'cd {root_folder} ; wget http://files.mltalker.com/cudnn-linux-x86_64-8.8.0.121_cuda12-archive.tar.xz && tar -xf cudnn-linux-x86_64-8.8.0.121_cuda12-archive.tar.xz')
        else:
            os.system(f'cd {root_folder} ; tar -xf cudnn-linux-x86_64-8.8.0.121_cuda12-archive.tar.xz')

    if not os.path.exists(os.path.join(root_folder, 'TensorRT-8.6.1.6')):
        print('download tensorrt')
        if not os.path.exists(os.path.join(root_folder, 'TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz')):
            os.system(f'cd {root_folder} ; wget http://files.mltalker.com/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz && tar -xf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz')
        else:
            os.system(f'cd {root_folder} ; tar -xf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz')

    cudnn_path = os.path.join(root_folder, 'cudnn-linux-x86_64-8.8.0.121_cuda12-archive')
    tensorrt_path = os.path.join(root_folder, 'TensorRT-8.6.1.6')

    # step2: 推送依赖库到包为止
    os.makedirs(os.path.join(output_folder, '3rd', abi), exist_ok=True)

    # step3: 生成cmake代码片段
    tensorrt_cmake_code_snippet = f'set(TensorRT_DIR {tensorrt_path})\n'
    tensorrt_cmake_code_snippet += 'set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)\n'
    tensorrt_cmake_code_snippet += 'find_package(TensorRT REQUIRED)\n'
    tensorrt_cmake_code_snippet += 'find_package(CUDA REQUIRED)\n'
    tensorrt_cmake_code_snippet += f'target_include_directories({project_name} '+'PUBLIC ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})\n'
    tensorrt_cmake_code_snippet += f'target_link_libraries({project_name} '+'${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})\n'

    code_line_list = []
    for line in open(os.path.join(output_folder, 'CMakeLists.txt')):
        if len(code_line_list) > 0 and code_line_list[-1].strip() == '# model engine' and line == '\n':
            code_line_list.append(tensorrt_cmake_code_snippet)

        code_line_list.append(line)

    with open(os.path.join(output_folder, 'CMakeLists.txt'), 'w') as fp:
        for line in code_line_list:
            fp.write(line)

    # step4: 更新run.sh片段
    code_line_list = []
    for line in open(os.path.join(output_folder, 'run.sh')):
        if 'export LD_LIBRARY_PATH=' in line:
            aa,bb,cc = line.split(';')
            bb = f'export LD_LIBRARY_PATH=.:{cudnn_path}/lib:{tensorrt_path}/lib'
            line = f'{aa}; {bb}; {cc}'
        code_line_list.append(line)
    with open(os.path.join(output_folder, 'run.sh'), 'w') as fp:
        for line in code_line_list:
            fp.write(line)


def generate_func_op_eagleeye_code(op_name, op_index, op_args, op_kwargs, output_folder):
    func = getattr(extent.func, op_name)
    input_ctx, output_ctx = op_index
    if isinstance(input_ctx, str):
        input_ctx = [input_ctx]
    if isinstance(output_ctx, str):
        output_ctx = [output_ctx]

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

    # 输入/输出定义映射
    args_define_map = {
        'CFTensor': 'CFTensor* %s;',
        'CITensor': 'CITensor* %s;',
        'CUCTensor': 'CUCTensor* %s;',
        'CDTensor': 'CDTensor* %s;'
    }

    # 输入/输出默认值映射
    args_default_map = {
        'CFTensor': '%s = NULL;',
        'CITensor': '%s = NULL;',
        'CUCTensor': '%s = NULL;',
        'CDTensor': '%s = NULL;',
    }

    # 输入/输出删除映射
    args_delete_map = {
        'CFTensor': 'if(%s != NULL){delete %s;};',
        'CITensor': 'if(%s != NULL){delete %s;};',
        'CUCTensor': 'if(%s != NULL){delete %s;};',
        'CDTensor': 'if(%s != NULL){delete %s;};',
    }

    # 输入/输出创建映射
    args_create_map = {
        'CFTensor': 'if(%s == NULL){%s=new_cftensor();};',
        'CITensor': 'if(%s == NULL){%s=new_citensor();};',
        'CUCTensor': 'if(%s == NULL){%s=new_cuctensor();};',
        'CDTensor': 'if(%s == NULL){%s=new_cdtensor();};',
    }

    # 输入初始化映射
    args_init_map = {
        'CFTensor': '%s->mirror(input[%d].cpu<float>(), input[%d].dims().data());',
        'CITensor': '%s->mirror(input[%d].cpu<int>(), input[%d].dims().data());',
        'CUCTensor': '%s->mirror(input[%d].cpu<unsigned char>(), input[%d].dims().data());',
        'CDTensor': '%s->mirror(input[%d].cpu<double>(), input[%d].dims().data());',
    }

    # 输出导出映射
    args_export_map = {
        'CFTensor': 'm_outputs[%d]=Tensor(std::vector<int64_t>(%s->dims, %s->dims+%s->dim_size),EAGLEEYE_FLOAT, DataFormat::AUTO,%s->data);',
        'CITensor': 'm_outputs[%d]=Tensor(std::vector<int64_t>(%s->dims, %s->dims+%s->dim_size),EAGLEEYE_INT, DataFormat::AUTO,%s->data);',
        'CUCTensor': 'm_outputs[%d]=Tensor(std::vector<int64_t>(%s->dims, %s->dims+%s->dim_size),EAGLEEYE_UCHAR, DataFormat::AUTO,%s->data);',
        'CDTensor': 'm_outputs[%d]=Tensor(std::vector<int64_t>(%s->dims, %s->dims+%s->dim_size),EAGLEEYE_DOUBLE, DataFormat::AUTO,%s->data);',
    }

    # 初始化C*Tensor
    init_map = {
        '<f4': 'init_cftensor',
        '<i4': 'init_citensor',
        '|u1': 'init_cuctensor'
    }

    input_define = ''
    output_define = ''
    input_default = ''
    output_default = ''
    input_delete = ''
    output_delete = ''
    input_create = ''
    output_create = ''
    input_init = ''
    output_export = ''

    const_define = ''
    const_default = ''
    const_delete = ''
    const_init = ''
    ext_cont_init = ''
    for arg in func_args:
        flag, arg_name, arg_type, arg_value = arg
        if flag.startswith('input'):
            input_p = int(flag[6:])
            arg_type = arg_type.cname.replace('*','')
            input_define += f'{args_define_map[arg_type]}\n' % arg_name
            input_default += f'{args_default_map[arg_type]}\n' % arg_name
            input_delete += f'{args_delete_map[arg_type]}\n' % (arg_name,arg_name)
            input_create += f'{args_create_map[arg_type]}\n' % (arg_name,arg_name)
            
            input_init += f'{args_init_map[arg_type]}\n' % (arg_name, input_p, input_p)

        elif flag.startswith('output'):
            arg_type = arg_type.cname.replace('*','')
            output_define += f'{args_define_map[arg_type]}\n' % arg_name
            output_default += f'{args_default_map[arg_type]}\n' % arg_name
            output_delete += f'{args_delete_map[arg_type]}\n' % (arg_name,arg_name)
            output_create += f'{args_create_map[arg_type]}\n' % (arg_name,arg_name)

            output_p = int(flag[7:])
            output_export += f'{args_export_map[arg_type]}\n' % (output_p,arg_name,arg_name,arg_name,arg_name)
        else:
            if isinstance(arg_value, np.ndarray):
                # array const 参数
                assert(arg_value.dtype == np.float32 or arg_value.dtype == np.int32 or arg_value.dtype == np.uint8)

                data_value_str = '{'+','.join([f'{v}' for v in arg_value.tolist()])+'}'
                data_shape_str = '{'+','.join([f'{v}' for v in arg_value.shape])+'}'
                arg_type = arg_type.cname.replace('*','')

                const_define += f'{args_define_map[arg_type]}\n' % arg_name
                const_default += f'{args_default_map[arg_type]}\n' % arg_name
                const_delete += f'{args_delete_map[arg_type]}\n' % (arg_name,arg_name)
                const_init += f'{arg_name}={init_map[arg_value.dtype.str]}({data_value_str}, {data_shape_str});\n'
            else:
                arg_type = arg_type.cname.replace('const', '')
                ext_cont_init += f'{arg_type} {arg_name}={str(arg_value).lower()};\n'

    # 创建header文件
    eagleeye_warp_h_code_content = \
        gen_code('./templates/op_func_code.h')(            
            op_name=f"{op_name.replace('_','').capitalize()}Op",
            input_num=len(input_ctx),
            output_num=len(output_ctx),
            input_define=input_define,
            output_define=output_define,
            const_define=const_define
        )

    # folder tree
    # output_folder/
    #   include/
    #   src/
    include_folder = os.path.join(output_folder, 'extent','include')
    os.makedirs(include_folder, exist_ok=True)
    with open(os.path.join(include_folder, f'{op_name}_op_warp.h'), 'w') as fp:
        fp.write(eagleeye_warp_h_code_content)

    args_inst = ''
    for _, arg_name, _, _ in func_args:
        if args_inst == '':
            args_inst = f'{arg_name}'
        else:
            args_inst += f',{arg_name}'
    
    eagleeye_warp_cpp_code_content = \
        gen_code('./templates/op_func_code.cpp')(
            op_name=f"{op_name.replace('_','').capitalize()}Op",
            func_name=op_name,
            inc_fname1=os.path.abspath(os.path.join(output_folder, 'extent', 'include', f'{op_name}_op_warp.h')),
            inc_fname2=os.path.abspath(func.func.loader_kwargs['cpp_info'].cpp_fname),
            # args_convert=args_convert,
            args_inst=args_inst,
            return_statement='',
            # output_covert=output_covert,
            # args_clear=args_clear
            input_default=input_default,
            output_default=output_default,

            input_delete=input_delete,
            output_delete=output_delete,

            input_create=input_create,
            output_create=output_create,

            input_init=input_init,
            output_export=output_export,

            const_default=const_default,
            const_delete=const_delete,
            const_init=const_init,
            ext_cont_init=ext_cont_init   
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


def generate_cls_op_eagleeye_code(op_name, op_index, op_args, op_kwargs, output_folder):
    func = getattr(extent.func, op_name)
    input_ctx, output_ctx = op_index
    if isinstance(input_ctx, str):
        input_ctx = [input_ctx]
    if isinstance(output_ctx, str):
        output_ctx = [output_ctx]

    # 创建cpp文件
    # 函数参数部分
    func_args = [None] * len(func.func.arg_names)
    # for i, (var_name, var_type) in enumerate(zip(func.func.arg_names, func.func.arg_types)):
    #     if i < len(op_args):
    #         func_args[i] =  ('param', var_name, var_type, op_args[i])

    # for i, (var_name, var_type) in enumerate(zip(func.func.arg_names, func.func.arg_types)):
    #     if op_kwargs.get(var_name, None) is not None and var_type.is_const:
    #         func_args[i] = ('param', var_name, var_type, op_kwargs.get(var_name))

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

    # 输入/输出定义映射
    args_define_map = {
        'CFTensor': 'CFTensor* %s;',
        'CITensor': 'CITensor* %s;',
        'CUCTensor': 'CUCTensor* %s;',
        'CDTensor': 'CDTensor* %s;'
    }

    # 输入/输出默认值映射
    args_default_map = {
        'CFTensor': '%s = NULL;',
        'CITensor': '%s = NULL;',
        'CUCTensor': '%s = NULL;',
        'CDTensor': '%s = NULL;',
    }

    # 输入/输出删除映射
    args_delete_map = {
        'CFTensor': 'if(%s != NULL){delete %s;};',
        'CITensor': 'if(%s != NULL){delete %s;};',
        'CUCTensor': 'if(%s != NULL){delete %s;};',
        'CDTensor': 'if(%s != NULL){delete %s;};',
    }

    # 输入/输出创建映射
    args_create_map = {
        'CFTensor': 'if(%s == NULL){%s=new_cftensor();};',
        'CITensor': 'if(%s == NULL){%s=new_citensor();};',
        'CUCTensor': 'if(%s == NULL){%s=new_cuctensor();};',
        'CDTensor': 'if(%s == NULL){%s=new_cdtensor();};',
    }

    # 输入初始化映射
    args_init_map = {
        'CFTensor': '%s->mirror(input[%d].cpu<float>(), input[%d].dims().data());',
        'CITensor': '%s->mirror(input[%d].cpu<int>(), input[%d].dims().data());',
        'CUCTensor': '%s->mirror(input[%d].cpu<unsigned char>(), input[%d].dims().data());',
        'CDTensor': '%s->mirror(input[%d].cpu<double>(), input[%d].dims().data());',
    }

    # 输出导出映射
    args_export_map = {
        'CFTensor': 'm_outputs[%d]=Tensor(std::vector<int64_t>(%s->dims, %s->dims+%s->dim_size),EAGLEEYE_FLOAT, DataFormat::AUTO,%s->data);',
        'CITensor': 'm_outputs[%d]=Tensor(std::vector<int64_t>(%s->dims, %s->dims+%s->dim_size),EAGLEEYE_INT, DataFormat::AUTO,%s->data);',
        'CUCTensor': 'm_outputs[%d]=Tensor(std::vector<int64_t>(%s->dims, %s->dims+%s->dim_size),EAGLEEYE_UCHAR, DataFormat::AUTO,%s->data);',
        'CDTensor': 'm_outputs[%d]=Tensor(std::vector<int64_t>(%s->dims, %s->dims+%s->dim_size),EAGLEEYE_DOUBLE, DataFormat::AUTO,%s->data);',
    }

    # 初始化C*Tensor
    init_map = {
        '<f4': 'init_cftensor',
        '<i4': 'init_citensor',
        '|u1': 'init_cuctensor'
    }

    input_define = ''
    output_define = ''
    input_default = ''
    output_default = ''
    input_delete = ''
    output_delete = ''
    input_create = ''
    output_create = ''
    input_init = ''
    output_export = ''

    const_define = ''
    const_default = ''
    const_delete = ''
    const_init = ''
    ext_cont_init = ''
    for arg in func_args:
        flag, arg_name, arg_type, arg_value = arg
        if flag.startswith('input'):
            input_p = int(flag[6:])
            arg_type = arg_type.cname.replace('*','')
            input_define += f'{args_define_map[arg_type]}\n' % arg_name
            input_default += f'{args_default_map[arg_type]}\n' % arg_name
            input_delete += f'{args_delete_map[arg_type]}\n' % (arg_name,arg_name)
            input_create += f'{args_create_map[arg_type]}\n' % (arg_name,arg_name)
            
            input_init += f'{args_init_map[arg_type]}\n' % (arg_name, input_p, input_p)

        elif flag.startswith('output'):
            arg_type = arg_type.cname.replace('*','')
            output_define += f'{args_define_map[arg_type]}\n' % arg_name
            output_default += f'{args_default_map[arg_type]}\n' % arg_name
            output_delete += f'{args_delete_map[arg_type]}\n' % (arg_name,arg_name)
            output_create += f'{args_create_map[arg_type]}\n' % (arg_name,arg_name)

            output_p = int(flag[7:])
            output_export += f'{args_export_map[arg_type]}\n' % (output_p,arg_name,arg_name,arg_name,arg_name)
        else:
            if isinstance(arg_value, np.ndarray):
                # array const 参数
                assert(arg_value.dtype == np.float32 or arg_value.dtype == np.int32 or arg_value.dtype == np.uint8)

                data_value_str = '{'+','.join([f'{v}' for v in arg_value.tolist()])+'}'
                data_shape_str = '{'+','.join([f'{v}' for v in arg_value.shape])+'}'
                arg_type = arg_type.cname.replace('*','')

                const_define += f'{args_define_map[arg_type]}\n' % arg_name
                const_default += f'{args_default_map[arg_type]}\n' % arg_name
                const_delete += f'{args_delete_map[arg_type]}\n' % (arg_name,arg_name)
                const_init += f'{arg_name}={init_map[arg_value.dtype.str]}({data_value_str}, {data_shape_str});\n'
            else:
                arg_type = arg_type.cname.replace('const', '')
                ext_cont_init += f'{arg_type} {arg_name}={str(arg_value).lower()};\n'

    # 创建header文件
    eagleeye_warp_h_code_content = \
        gen_code('./templates/op_class_code.h')(            
            op_name=f"{op_name.replace('_','').capitalize()}Op",
            input_num=len(input_ctx),
            output_num=len(output_ctx),
            cls_name=func.func.func_name,
            input_define=input_define,
            output_define=output_define,
            const_define=const_define
        )

    # folder tree
    # output_folder/
    #   include/
    #   src/
    include_folder = os.path.join(output_folder, 'extent','include')
    os.makedirs(include_folder, exist_ok=True)
    with open(os.path.join(include_folder, f'{op_name}_op_warp.h'), 'w') as fp:
        fp.write(eagleeye_warp_h_code_content)

    args_run = ''
    for _, arg_name, _, _ in func_args:
        if args_run == '':
            args_run = f'{arg_name}'
        else:
            args_run += f',{arg_name}'

    args_run_names = func.func.loader_kwargs['construct_arg_names']
    args_init = ','.join([f'{op_kwargs[n]}' for n in args_run_names])
    eagleeye_warp_cpp_code_content = \
        gen_code('./templates/op_class_code.cpp')(
            op_name=f"{op_name.replace('_','').capitalize()}Op",
            func_name=op_name,
            inc_fname1=os.path.abspath(os.path.join(output_folder, 'extent', 'include', f'{op_name}_op_warp.h')),
            inc_fname2=os.path.abspath(func.func.loader_kwargs['cpp_info'].cpp_fname),
            cls_name=func.func.func_name,
            args_init=args_init,
            args_run=args_run,
            return_statement='',

            input_default=input_default,
            output_default=output_default,

            input_delete=input_delete,
            output_delete=output_delete,

            input_create=input_create,
            output_create=output_create,

            input_init=input_init,
            output_export=output_export,

            const_default=const_default,
            const_delete=const_delete,
            const_init=const_init,
            ext_cont_init=ext_cont_init
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



def auto_generate_eagleeye_op(op_name, op_index, op_args, op_kwargs, output_folder):
    func = getattr(extent.func, op_name)
    if func.func.func_kind == 3:
        # class
        result = generate_cls_op_eagleeye_code(op_name, op_index, op_args, op_kwargs, output_folder)
        return result
    else:
        # func
        result = generate_func_op_eagleeye_code(op_name, op_index, op_args, op_kwargs, output_folder)
        return result


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

    converted_op_args['c++_type'] = 'std::map<std::string, std::vector<float>>'
    return converted_op_args


def update_cmakelist(output_folder, project_name, pipeline_name, src_op_warp_list, compile_info):
    info = []
    is_found_include_directories_insert = False
    is_start_add_src_code = False
    pipeline_plugin_flag = []
    src_op_warp_flag = [0 for _ in range(len(src_op_warp_list))]
    for line in open(os.path.join(output_folder, 'CMakeLists.txt')):
        if is_start_add_src_code and not line.strip().endswith(')'):
            for src_op_warp_file_i, src_op_warp_file in enumerate(src_op_warp_list):
                if src_op_warp_file == line.strip():
                    src_op_warp_flag[src_op_warp_file_i] = 1
                    break
            pipeline_plugin_flag.append(line.strip())

        if is_start_add_src_code and line.strip().endswith(')'):
            if f'./{pipeline_name}_plugin.cpp' not in pipeline_plugin_flag:
                info.append(f'./{pipeline_name}_plugin.cpp\n')

            for src_op_warp_file_i, src_op_warp_file_flag in enumerate(src_op_warp_flag):
                if src_op_warp_file_flag == 0:
                    info.append(f'{src_op_warp_list[src_op_warp_file_i]}\n')

            is_start_add_src_code = False

        if f'set({project_name}_SRC' in line:
            is_start_add_src_code = True
        
        # 添加扩展c++代码
        if line.startswith('include_directories') and not is_found_include_directories_insert:
            extent_cpp_include_folder = os.path.dirname(os.path.realpath(__file__))
            extent_cpp_include_folder = os.path.dirname(extent_cpp_include_folder)
            extent_cpp_include_folder = os.path.dirname(extent_cpp_include_folder)
            extent_cpp_include_folder = os.path.join(extent_cpp_include_folder, 'extent','cpp','include')
            if extent_cpp_include_folder not in line:
                info.append(f'include_directories({extent_cpp_include_folder})\n')

            is_found_include_directories_insert = True

        # 添加opencv依赖
        if 'set(OpenCV_DIR "")' in line and (config.USING_OPENCV or 'opencv' in compile_info):
            opencv_path = os.path.join(ANTGO_DEPEND_ROOT, 'opencv-install')
            line = f'set(OpenCV_DIR "{opencv_path}")'

        info.append(line)

    with open(os.path.join(output_folder, 'CMakeLists.txt'), 'w') as fp:
        for line in info:
            fp.write(line)


tensortype_map = {
    'tensor(float)': 6
}


def convert_onnx_to_platform_engine(op_name, op_index, op_args, op_kwargs, output_folder, platform, abi, project_name):
    # 0.step 参数
    platform_engine = op_kwargs.get('engine', 'tensorrt')   # 在平台，使用什么模型引擎
    platform_engine_args = op_kwargs.get('engine_args', {}) # 在平台，使用什么模型引擎参数
    platform_device = platform_engine_args.get('device', 'CPU')        # 在平台，使用什么设备

    # 1.step 转换模型格式文件
    # TODO, 临时
    platform_model_path = platform_engine_args.get('model', None)
    if platform_model_path is None and platform == 'android':
        onnx_file_path = op_kwargs.get('onnx_path')
        # TODO,支持自动转换模型
        if platform_engine == 'snpe':
            if platform_engine_args.get('quantize', False):
                # 转量化模型
                os.system(f'mkdir /tmp/onnx ; mkdir /tmp/onnx/snpe ; cp {onnx_file_path} /tmp/onnx/')
                # 确保存在校正数据集
                assert(os.path.exists(platform_engine_args.get('calibration-images')))
                shutil.copytree(platform_engine_args.get('calibration-images'), '/tmp/onnx/calibration-images')

                prefix = os.path.basename(onnx_file_path)[:-5]
                onnx_dir_path = os.path.dirname(onnx_file_path)
                os.system(f'cd /tmp/onnx ; docker run --rm -v $(pwd):/workspace snpeconvert bash convert.sh --i={prefix}.onnx --o=./snpe/{prefix} --quantize --npu --data-folder=calibration-images')
                converted_model_file = ''
                for file_name in os.listdir('/tmp/onnx/snpe/'):
                    if file_name[0] != '.':
                        converted_model_file = file_name
                        break

                os.system(f'cp -r /tmp/onnx/snpe/* {onnx_dir_path} ; rm -rf /tmp/onnx/')
                platform_model_path = os.path.join(onnx_dir_path, converted_model_file)
            else:
                # 转浮点模型
                os.system(f'mkdir /tmp/onnx ; mkdir /tmp/onnx/snpe ; cp {onnx_file_path} /tmp/onnx/')

                prefix = os.path.basename(onnx_file_path)[:-5]
                onnx_dir_path = os.path.dirname(onnx_file_path)
                os.system(f'cd /tmp/onnx ; docker run --rm -v $(pwd):/workspace snpeconvert bash convert.sh --i={prefix}.onnx --o=./snpe/{prefix}')
                converted_model_file = ''
                for file_name in os.listdir('/tmp/onnx/snpe/'):
                    if file_name[0] != '.':
                        converted_model_file = file_name
                        break

                os.system(f'cp -r /tmp/onnx/snpe/* {onnx_dir_path} ; rm -rf /tmp/onnx/')
                platform_model_path = os.path.join(onnx_dir_path, converted_model_file)
        elif platform_engine == 'rknn':
            if platform_engine_args.get('quantize', False):
                # 转量化模型
                os.system(f'mkdir /tmp/onnx ; mkdir /tmp/onnx/rknn ; cp {onnx_file_path} /tmp/onnx/')
                # 确保存在校正数据集
                assert(os.path.exists(platform_engine_args.get('calibration-images')))
                shutil.copytree(platform_engine_args.get('calibration-images'), '/tmp/onnx/calibration-images')

                prefix = os.path.basename(onnx_file_path)[:-5]
                onnx_dir_path = os.path.dirname(onnx_file_path)
                mean_values = ','.join([str(v) for v in  op_kwargs.get('mean')])
                std_values = ','.join([str(v) for v in  op_kwargs.get('std')])
                os.system(f'cd /tmp/onnx ; docker run --rm -v $(pwd):/workspace rknnconvert bash convert.sh --i={prefix}.onnx --quantize --image-folder=./calibration-images --o=./rknn/{prefix} --device={platform_device} --mean-values={mean_values} --std-values={std_values}')
                converted_model_file = ''
                for file_name in os.listdir('/tmp/onnx/rknn/'):
                    if file_name[0] != '.':
                        converted_model_file = file_name
                        break

                os.system(f'cp -r /tmp/onnx/rknn/* {onnx_dir_path} ; rm -rf /tmp/onnx/')
                platform_model_path = os.path.join(onnx_dir_path, converted_model_file)
            else:
                # 转浮点模型
                os.system(f'mkdir /tmp/onnx ; mkdir /tmp/onnx/rknn ; cp {onnx_file_path} /tmp/onnx/')
                
                prefix = os.path.basename(onnx_file_path)[:-5]
                onnx_dir_path = os.path.dirname(onnx_file_path)
                mean_values = ','.join([str(v) for v in  op_kwargs.get('mean')])
                std_values = ','.join([str(v) for v in  op_kwargs.get('std')])
                os.system(f'cd /tmp/onnx ; docker run --rm -v $(pwd):/workspace rknnconvert bash convert.sh --i={prefix}.onnx --o=./rknn/{prefix} --device={platform_device} --mean-values={mean_values} --std-values={std_values}')
                converted_model_file = ''
                for file_name in os.listdir('/tmp/onnx/rknn/'):
                    if file_name[0] != '.':
                        converted_model_file = file_name
                        break
                
                os.system(f'cp -r /tmp/onnx/rknn/* {onnx_dir_path} ; rm -rf /tmp/onnx/')
                platform_model_path = os.path.join(onnx_dir_path, converted_model_file)
        elif platform_engine == 'tnn':
            os.system(f'mkdir /tmp/onnx ; mkdir /tmp/onnx/tnn ; cp {onnx_file_path} /tmp/onnx/')                 
            prefix = os.path.basename(onnx_file_path)[:-5]
            onnx_dir_path = os.path.dirname(onnx_file_path)
            os.system(f'cd /tmp/onnx/ ; docker run --rm -v $(pwd):/workspace tnnconvert bash convert.sh --i={prefix}.onnx --o=./tnn/{prefix}')
            converted_model_file = []
            for file_name in os.listdir('/tmp/onnx/tnn/'):
                if file_name[0] != '.' and '.tnnproto' in file_name:
                    converted_model_file = file_name
                    break

            os.system(f'cp -r /tmp/onnx/tnn/* {onnx_dir_path} ; rm -rf /tmp/onnx/')
            platform_model_path = os.path.join(onnx_dir_path, converted_model_file)

    if platform_model_path is None and platform == 'linux':
        platform_model_path = op_kwargs.get('onnx_path')

    # 将平台关联模型放入输出目录中
    os.makedirs(os.path.join(output_folder, 'model'), exist_ok=True)
    if os.path.exists(platform_model_path):
        os.makedirs(os.path.join(output_folder, 'model'), exist_ok=True)
        shutil.copyfile(platform_model_path, os.path.join(output_folder, 'model', os.path.basename(platform_model_path)))
        if '.tnnproto' in  platform_model_path:
            another_model_path = platform_model_path.replace('.tnnproto', '.tnnmodel')
            shutil.copyfile(another_model_path, os.path.join(output_folder, 'model', os.path.basename(another_model_path)))

        if '.tnnmodel' in platform_model_path:
            another_model_path = platform_model_path.replace('.tnnmodel', '.tnnproto')
            shutil.copyfile(another_model_path, os.path.join(output_folder, 'model', os.path.basename(another_model_path)))

    # 2.step 参数转换及代码生成
    config_func_map = {
        'snpe': snpe_import_config,
        'rknn': rknn_import_config,
        'tensorrt': tensorrt_import_config,
        'tnn': tnn_import_config
    }
    config_func_map[platform_engine](output_folder, project_name, platform, abi, platform_device)

    input_ctx, output_ctx = op_index
    if not isinstance(input_ctx, tuple):
        input_ctx = (input_ctx,)
    if not isinstance(output_ctx, tuple):
        output_ctx = (output_ctx,)

    template_args = f'<{len(input_ctx)},{len(output_ctx)}>'

    # 解析onnx模型获得输入、输出信息
    onnx_session = onnxruntime.InferenceSession(op_kwargs['onnx_path'], providers=['CPUExecutionProvider'])
    mode_name = os.path.basename(op_kwargs['onnx_path']).split('.')[0]
    input_names = []
    input_shapes = []
    input_types = []
    output_names = []
    output_shapes = []
    output_types = []
    model_folder = f'/sdcard/{project_name}/.model/' if platform == 'android' else os.path.dirname(op_kwargs['onnx_path'])          # 考虑将转好的模型放置的位置
    writable_path = f'/sdcard/{project_name}/.tmp/' if platform == 'android' else os.path.dirname(op_kwargs['onnx_path'])           # 考虑到 设备可写权限位置(android)

    # 更新run.sh（仅设备端运行时需要添加推送模型代码）
    if platform.lower() == 'android':
        run_shell_code_list = []
        is_found_model_push_line = False
        is_found_model_platform_folder = False
        for line in open(os.path.join(output_folder, 'run.sh')):
            if '.tnnmodel' in line:
                continue
            if line.startswith(f'adb push {platform_model_path}') and not is_found_model_push_line:
                # 替换
                line = f'adb push {platform_model_path} {model_folder}\n'
                if platform_model_path.endswith('.tnnproto'):
                    line += f'adb push {platform_model_path.replace(".tnnproto", ".tnnmodel")} {model_folder}\n'
                is_found_model_push_line = True

            if f'mkdir -p {model_folder};' in line:
                is_found_model_platform_folder = True

            if line.startswith('adb shell "cd /data/local/tmp') and not is_found_model_platform_folder:
                # 插入
                run_shell_code_list.append(f'adb shell "if [ ! -d {model_folder} ]; then mkdir -p {model_folder}; fi;"\n')

            if line.startswith('adb shell "cd /data/local/tmp') and not is_found_model_push_line:
                # 插入
                run_shell_code_list.append(f'adb push {platform_model_path} {model_folder}\n')
                if platform_model_path.endswith('.tnnproto'):
                    run_shell_code_list.append(f'adb push {platform_model_path.replace(".tnnproto", ".tnnmodel")} {model_folder}\n')
            
            run_shell_code_list.append(line)

        with open(os.path.join(output_folder, 'run.sh'), 'w') as fp:
            for line in run_shell_code_list:
                fp.write(line)

    num_threads = 2
    for input_tensor in onnx_session.get_inputs():
        input_names.append(input_tensor.name)
        input_shapes.append(input_tensor.shape)
        input_types.append(tensortype_map[input_tensor.type])

    alias_output_names = []
    for output_tensor in onnx_session.get_outputs():
        if 'alias_output_names' in platform_engine_args:
            output_names.append(platform_engine_args['alias_output_names'][output_tensor.name])
            alias_output_names.append(output_tensor.name)
        else:
            output_names.append(output_tensor.name)
        output_shapes.append(output_tensor.shape)
        output_types.append(tensortype_map[output_tensor.type])

    # string args, vector args, other args
    model_name = '.'.join(platform_model_path.split("/")[-1].split('.')[:-1])
    op_args = (
        {
            'model_name': [f'"{model_name}"'],
            'device': [f'"{platform_device}"'],
            'input_names': [f'"{c}"' for c in input_names],
            'output_names': [f'"{c}"' for c in output_names],
            'alias_output_names': [f'"{c}"' for c in alias_output_names],
            'model_folder': [f'"{model_folder}"'],
            'writable_path': [f'"{writable_path}"'],
            'c++_type': 'std::map<std::string, std::vector<std::string>>'
        },
        {
            'input_shapes': ['{'+','.join(str(m) for m in n)+'}' for n in input_shapes],
            'output_shapes': ['{'+','.join(str(m) for m in n)+'}' for n in output_shapes],
            'c++_type': 'std::map<std::string, std::vector<std::vector<float>>>'
        },
        {
            'input_types': input_types,
            'output_types': output_types,
            'num_threads': [num_threads],
            'c++_type': 'std::map<std::string, std::vector<float>>'
        }
    )

    if op_kwargs.get('mean', None) is not None:
        op_args[-1]['mean'] = ['{'+','.join(str(m) for m in op_kwargs['mean'])+'}']
        op_args[-1]['std'] = ['{'+','.join(str(m) for m in op_kwargs['std'])+'}']
        op_args[-1]['reverse_channel'] = ['{1}'] if op_kwargs.get('reverse_channel', False) else ['{0}']

    include_path = f'eagleeye/engine/nano/op/{platform_engine.lower()}_op.hpp'

    return f'{platform_engine.capitalize()}Op', template_args, op_args, include_path


def package_build(output_folder, eagleeye_path, project_config, platform, abi=None, generate_demo_code=True):    
    project_name = project_config["name"]
    pipeline_name = project_name
    if '/' in project_name:
        project_name, pipeline_name = project_name.split('/')

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

        print(f'op_name {op_name}')
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
            op_name = op_name[12:]
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

            if op_name.startswith('inference_onnx_op'):
                # 算子转换为平台预测引擎算子
                engine_op_name, template_args, op_args, include_path = \
                    convert_onnx_to_platform_engine(op_name, op_index, op_args, op_kwargs, output_folder, platform, abi, project_name=project_name)

                deploy_graph_info[op_unique_name] = {
                    'type': engine_op_name,
                    'template': template_args,
                    'input': input_ctx,
                    'output': output_ctx,
                    'args': op_args,
                    'include': include_path
                }
                continue
            
            # 转换统一格式
            if op_name.endswith('_op'):
                op_name = op_name.capitalize()                     
                op_name = op_name.replace('_op', 'Op')

            # 检查是否在核心集中
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
        if 'template' in deploy_op_info:
            op_graph_code += f'dataflow::Node* {deploy_op_name} = op_graph->add("{deploy_op_name}", {deploy_op_info["type"]}{deploy_op_info["template"]}(), EagleeyeRuntime(EAGLEEYE_CPU));\n'
        else:
            op_graph_code += f'dataflow::Node* {deploy_op_name} = op_graph->add("{deploy_op_name}", {deploy_op_info["type"]}(), EagleeyeRuntime(EAGLEEYE_CPU));\n'

        deploy_op_args_tuple = deploy_op_info['args']
        if isinstance(deploy_op_args_tuple, dict):
            deploy_op_args_tuple = (deploy_op_args_tuple,)

        for deploy_op_args in deploy_op_args_tuple:
            arg_code = ''
            for deploy_arg_name, deploy_arg_list in deploy_op_args.items():
                if deploy_arg_name != 'c++_type':
                    if arg_code == '':
                        arg_code = '{"'+deploy_arg_name+'",{'+','.join([str(v) for v in deploy_arg_list])+'}}'
                    else:
                        arg_code += ',{"'+deploy_arg_name+'",{'+','.join([str(v) for v in deploy_arg_list])+'}}'

            if arg_code != '':
                args_init_code = deploy_op_args['c++_type']+'({'+arg_code+'})'
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
            project=pipeline_name,
            version=project_config.get('version', '1.0.0.0'),
            signature=project_config.get('signature', 'xxx'),
            include_list=include_list,
            in_port='{'+','.join([str(i) for i in range(len(project_config['input']))]) + '}',
            in_signal='{'+','.join(['"'+info[-1]+'"' for info in project_config['input']])+'}',
            out_port='{'+','.join([str(i) for i in range(len(project_config['output']))]) + '}',
            out_signal=','.join(['"'+info[-1]+'"' for info in project_config['output']]),
            graph_in_ops='{'+','.join(['"'+deploy_output_data_name_inv_link[info[0]][0]+'"' for info in project_config['input']])+'}',
            graph_out_ops='{'+','.join(['{"'+deploy_output_data_name_inv_link[info[0]][0]+'",'+str(deploy_output_data_name_inv_link[info[0]][1])+'}' for info in project_config['output']])+'}',
            op_graph=op_graph_code
        )

    with open(os.path.join(output_folder, f'{pipeline_name}_plugin.cpp'), 'w') as fp:
        fp.write(eagleeye_plugin_code_content)

    eagleeye_plugin_header_content = \
        gen_code('./templates/plugin_code.h')(project=pipeline_name)

    with open(os.path.join(output_folder, f'{pipeline_name}_plugin.h'), 'w') as fp:
        fp.write(eagleeye_plugin_header_content)

    os.makedirs(os.path.join(output_folder, 'data'), exist_ok=True)

    # 准备插件demo文件
    if generate_demo_code:
        plugin_input_size_list = []
        plugin_input_type_list = []
        for info_i, info in enumerate(project_config['input']):
            for graph_op_info in graph_config:
                if graph_op_info['op_index'][-1][0] == info[0]:
                    plugin_input_size_list.append(graph_op_info['op_kwargs']['shape'])
                    plugin_input_type_list.append(graph_op_info['op_kwargs']['data_type'])

        plugin_input_size_list = ['{'+','.join([str(v) for v in shape])+'}' for shape in plugin_input_size_list]
        plugin_input_type_list = ','.join([str(v) for v in plugin_input_type_list])
        demo_code_content = gen_code(f'./templates/demo_code.cpp')(
            project=project_name,
            input_name_list='{'+','.join([f'"placeholder_{i}"' for i in range(len(project_config['input']))])+'}',
            input_size_list='{'+','.join(plugin_input_size_list)+'}',
            input_type_list='{'+plugin_input_type_list+'}',
            output_name_list='{'+','.join(['"nnnode"' for _ in range(len(project_config['output']))])+'}',
            output_port_list='{'+','.join([f"{i}" for i in range(len(project_config['output']))])+'}'
        )
        with open(os.path.join(output_folder, f'{project_name}_demo.cpp'), 'w') as fp:
            fp.write(demo_code_content)

    # 准备额外依赖库（libc++_shared.so）
    if platform.lower() == 'android':
        ndk_path = os.environ['ANDROID_NDK_HOME']
        os.makedirs(os.path.join(output_folder, '3rd', abi), exist_ok=True)
        shutil.copy(os.path.join(ndk_path, "sources/cxx-stl/llvm-libc++/libs", abi, 'libc++_shared.so'), os.path.join(output_folder, '3rd', abi, 'libc++_shared.so'))

    # 更新CMakeLists.txt
    update_cmakelist(output_folder, project_name, pipeline_name,[s['src'] for s in deploy_graph_info.values() if 'src' in s], project_config.get('compile', []))

    # 更新插件工程编译脚本
    shell_code_content = gen_code('./templates/android_build.sh')(
        project=project_name,
        ANDROID_NDK_HOME=os.environ['ANDROID_NDK_HOME']
    )
    with open(os.path.join(output_folder, 'android_build.sh'), 'w') as fp:
        fp.write(shell_code_content)

    shell_code_content = gen_code('./templates/linux_build.sh')(
        project=project_name
    )
    with open(os.path.join(output_folder, 'linux_build.sh'), 'w') as fp:
        fp.write(shell_code_content)

    # 保存项目配置信息
    for item in graph_config:
        # check op_args
        op_args = []
        for value in item['op_args']:
            if isinstance(value, np.ndarray):
                value = value.tolist()
            op_args.append(value)
        item['op_args'] = op_args

        # check op_kwargs
        for key in item['op_kwargs']:
            if isinstance(item['op_kwargs'][key], np.ndarray):
                item['op_kwargs'][key] = item['op_kwargs'][key].tolist()

    project_config.update({
        'graph': graph_config,
        'platform': platform
    })
    with open(os.path.join(output_folder, '.project.json'), 'w') as fp:
        json.dump(project_config, fp)

    if 'python' in project_config.get('compile', []):
        pymodel_code_content = gen_code('./templates/pypipelinemodel_code.cpp')(
            project=project_name
        )
        with open(os.path.join(output_folder, 'PyPipelineModel.cpp'), 'w') as fp:
            fp.write(pymodel_code_content)

        os.system(f'cd {output_folder} ; bash {platform}_build.sh BUILD_PYTHON_MODULE')
    else:
        os.system(f'cd {output_folder} ; bash {platform}_build.sh')


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


def prepare_eagleeye_environment(system_platform, abi_platform):
    print('Check eagleeye environment')
    os.makedirs(ANTGO_DEPEND_ROOT, exist_ok=True)
    if not os.path.exists(os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye')) or len(os.listdir(os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye'))) == 0:
        print('Download eagleeye git')
        os.system(f'cd {ANTGO_DEPEND_ROOT} ; git clone https://github.com/jianzfb/eagleeye.git')

    p = subprocess.Popen("pip3 show eagleeye", shell=True, encoding="utf-8", stdout=subprocess.PIPE)
    if p.stdout.read() == '':
        print('Install eagleeye scafold')
        os.system(f'cd {ANTGO_DEPEND_ROOT}/eagleeye/scripts ; pip3 install -r requirements.txt ; python3 setup.py install')

    eagleeye_path = f'{ANTGO_DEPEND_ROOT}/eagleeye/{system_platform}-install'
    if not os.path.exists(eagleeye_path):
        print('Compile eagleeye core sdk')
        os.system(f'cd {ANTGO_DEPEND_ROOT}/eagleeye ; bash {system_platform.lower()}_build.sh ; mv install {system_platform}-install')
    eagleeye_path = os.path.abspath(eagleeye_path)
    return eagleeye_path


class DeployMixin:
    def build(self, platform='android/arm64-v8a', output_folder='./deploy', project_config=None):
        # android/arm64-v8a, linux/x86-64
        assert(platform in ['android/arm64-v8a', 'linux/x86-64'])
        system_platform, abi_platform = platform.split('/')

        # 准备eagleeye集成环境
        eagleeye_path = prepare_eagleeye_environment(system_platform, abi_platform)

        # 创建工程
        project_name = project_config["name"]
        pipeline_name = project_name
        if '/' in project_name:
            project_name, pipeline_name = project_name.split('/')
        os.makedirs(output_folder, exist_ok=True)
        if not os.path.exists(os.path.join(output_folder, f'{project_name}_plugin')):
            if project_config.get('git', None) is not None and project_config['git'] != '':
                os.system(f'cd {output_folder} ; git clone {project_config["git"]}')
            else:
                os.system(f'cd {output_folder} ; eagleeye-cli project --project={project_name} --version={project_config.get("version", "1.0.0.0")} --signature=xxxxx --build_type=Release --abi={abi_platform.capitalize() if system_platform != "android" else abi_platform} --eagleeye={eagleeye_path}')
        output_folder = os.path.join(output_folder, f'{project_name}_plugin')

        # 编译
        package_build(
            output_folder, 
            eagleeye_path, 
            project_config=project_config, platform=system_platform, abi=abi_platform, generate_demo_code=project_name==pipeline_name)

        return True
