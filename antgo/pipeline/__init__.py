# -*- coding: UTF-8 -*-
# @Time    : 2022/9/5 23:38
# @File    : __init__.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.pipeline.hub import *
import numpy as np
import os
import json

def run(project, folder='./deploy', **kwargs):
    project_folder = os.path.join(folder, f'{project}_plugin')
    if not os.path.exists(project_folder):
        print(f'Project {project} not exist.')
        return

    if not os.path.exists(os.path.join(project_folder, '.project.json')):
        print(f'Project {project} missing .project.json file.')
        return

    os.makedirs(os.path.join(project_folder, 'data', 'input'), exist_ok=True)
    os.makedirs(os.path.join(project_folder, 'data', 'output'), exist_ok=True)

    with open(os.path.join(project_folder, '.project.json'), 'r') as fp:
        project_info = json.load(fp)

    project_input_info = project_info['input']
    project_output_info = project_info['output']
    project_graph_info = project_info['graph']
    project_platform_info = project_info['platform']

    for input_i, input_info in enumerate(project_input_info):
        data_value = kwargs.get(input_info[0], None)
        if data_value is None:
            print(f'Missing {input_info[0]} data')

        # file format
        # input_name.input_port.type.shape.bin
        if isinstance(data_value, np.ndarray):
            data_type_code = -1
            if data_value.dtype == np.int8:
                data_type_code = 0
            elif data_value.dtype == np.uint8:
                data_type_code = 1
            elif data_value.dtype == np.int32:
                data_type_code = 4
            elif data_value.dtype == np.uint32:
                data_type_code = 5
            elif data_value.dtype == np.float32:
                data_type_code = 6
            elif data_value.dtype == np.float64:
                data_type_code = 7
            elif data_value.dtype == np.bool:
                data_type_code = 10
            
            data_shape_code = '-'.join([str(s) for s in data_value.shape])
            data_value.tofile(os.path.join(project_folder, 'data', 'input', f'placeholder_0.{0}.{data_type_code}.{data_shape_code}.bin'))

    os.system(f'cd {project_folder} && bash run.sh')

    out_list = [None for _ in range(len(project_output_info))]
    for file_name in os.listdir(os.path.join(project_folder, 'data', 'output')):
        file_prefix = file_name[:-4]
        _, data_port, data_type_code, data_shape = file_prefix.split('.')
        shape_list = [int(v) for v in data_shape.split('-')]
        data_port = (int)(data_port)
        data_type_code = (int)(data_type_code)

        if data_type_code == 0:
            # int8
            data = np.fromfile(os.path.join(project_folder, 'data', 'output', file_name), np.int8)
            data = data.reshape(shape_list)
            out_list[data_port] = data
        elif data_type_code == 1:
            # uint8
            data = np.fromfile(os.path.join(project_folder, 'data', 'output', file_name), np.uint8)
            data = data.reshape(shape_list)
            out_list[data_port] = data
        elif data_type_code == 4:
            # int32
            data = np.fromfile(os.path.join(project_folder, 'data', 'output', file_name), np.int32)
            data = data.reshape(shape_list)
            out_list[data_port] = data
        elif data_type_code == 5:
            # uint32
            data = np.fromfile(os.path.join(project_folder, 'data', 'output', file_name), np.uint32)
            data = data.reshape(shape_list)
            out_list[data_port] = data
        elif data_type_code == 6:
            # float32
            data = np.fromfile(os.path.join(project_folder, 'data', 'output', file_name), np.float32)
            data = data.reshape(shape_list)
            out_list[data_port] = data
        elif data_type_code == 7:
            # double
            data = np.fromfile(os.path.join(project_folder, 'data', 'output', file_name), np.float64)
            data = data.reshape(shape_list)
            out_list[data_port] = data
        elif data_type_code == 10:
            # bool
            data = np.fromfile(os.path.join(project_folder, 'data', 'output', file_name), np.bool)
            data = data.reshape(shape_list)
            out_list[data_port] = data

    return out_list


def package(project, folder='./deploy', **kwargs):
    # project_folder = os.path.join(folder, f'{project}_plugin')
    # if not os.path.exists(project_folder):
    #     print(f'Project {project} not exist.')
    #     return

    # os.system(f'cd {project_folder} && bash package.sh')
    pass


def service(project, folder='./deploy', **kwargs):
    pass