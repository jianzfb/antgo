import os
import sys
import copy
from typing import Any
import numpy as np
import json
import shutil


class Exe(object):
    def __init__(self, func_op_name=None, folder=None, **kwargs):
        self.plugin_name = func_op_name
        self.folder = "./"
        if folder is not None:
            self.folder = folder
        self.project_folder = os.path.join(self.folder, 'deploy', f'{self.plugin_name}_plugin')
        self.data_folder = self.project_folder
        if not os.path.exists(self.project_folder):
            print(f'Project {project} not exist.')
            return

        if not os.path.exists(os.path.join(self.project_folder, '.project.json')):
            print(f'Project {project} missing .project.json file.')
            return

        with open(os.path.join(self.project_folder, '.project.json'), 'r') as fp:
            project_info = json.load(fp)
        
        if project_info['platform'] == 'linux':
            self.data_folder = os.path.join(self.folder, 'deploy', f'{self.plugin_name}_plugin', 'bin', 'X86-64')
        self.project_input_info = project_info['input']
        self.project_output_info = project_info['output']
        self.project_graph_info = project_info['graph']
        self.project_platform_info = project_info['platform']

        os.makedirs(os.path.join(self.data_folder, 'data', 'input'), exist_ok=True)
        os.makedirs(os.path.join(self.data_folder, 'data', 'output'), exist_ok=True)

        self.is_first_call = True

    def __call__(self, *args):
        # 清空数据
        if os.path.exists(os.path.join(self.data_folder, 'data', 'input')):
            shutil.rmtree(os.path.join(self.data_folder, 'data', 'input')) 
            os.makedirs(os.path.join(self.data_folder, 'data', 'input'), exist_ok=True)

        if os.path.exists(os.path.join(self.data_folder, 'data', 'output')):
            shutil.rmtree(os.path.join(self.data_folder, 'data', 'output')) 
            os.makedirs(os.path.join(self.data_folder, 'data', 'output'), exist_ok=True)

        # 准备输入数据
        run_args = []
        for arg_i, (data_value, data_info) in enumerate(zip(args, self.project_input_info)):
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

                data_shape_code = ','.join([str(s) for s in data_value.shape])
                data_value.tofile(os.path.join(self.data_folder, 'data', 'input', f'placeholder_0.{0}.{data_type_code}.{data_shape_code}.bin'))
                run_args.append(f'placeholder_{arg_i}/{data_shape_code}/{data_type_code}')

        run_args = ' '.join(run_args)

        # 运行
        if self.is_first_call:
            os.system(f'cd {self.project_folder} && bash run.sh reload {run_args}')
            self.is_first_call = False
        else:
            os.system(f'cd {self.project_folder} && bash run.sh normal {run_args}')

        # 解析输出数据
        out_list = [None for _ in range(len(self.project_output_info))]
        for file_name in os.listdir(os.path.join(self.data_folder, 'data', 'output')):
            file_prefix = file_name[:-4]
            _, data_port, data_type_code, data_shape = file_prefix.split('.')
            shape_list = [int(v) for v in data_shape.split(',')]
            data_port = (int)(data_port)
            data_type_code = (int)(data_type_code)

            if data_type_code == 0:
                # int8
                data = np.fromfile(os.path.join(self.data_folder, 'data', 'output', file_name), np.int8)
                data = data.reshape(shape_list)
                out_list[data_port] = data
            elif data_type_code == 1 or data_type_code == 8 or data_type_code == 9:
                # uint8
                data = np.fromfile(os.path.join(self.data_folder, 'data', 'output', file_name), np.uint8)
                data = data.reshape(shape_list)
                out_list[data_port] = data
            elif data_type_code == 4:
                # int32
                data = np.fromfile(os.path.join(self.data_folder, 'data', 'output', file_name), np.int32)
                data = data.reshape(shape_list)
                out_list[data_port] = data
            elif data_type_code == 5:
                # uint32
                data = np.fromfile(os.path.join(self.data_folder, 'data', 'output', file_name), np.uint32)
                data = data.reshape(shape_list)
                out_list[data_port] = data
            elif data_type_code == 6:
                # float32
                data = np.fromfile(os.path.join(self.data_folder, 'data', 'output', file_name), np.float32)
                data = data.reshape(shape_list)
                out_list[data_port] = data
            elif data_type_code == 7:
                # double
                data = np.fromfile(os.path.join(self.data_folder, 'data', 'output', file_name), np.float64)
                data = data.reshape(shape_list)
                out_list[data_port] = data
            elif data_type_code == 10:
                # bool
                data = np.fromfile(os.path.join(self.data_folder, 'data', 'output', file_name), np.bool)
                data = data.reshape(shape_list)
                out_list[data_port] = data

        return out_list if len(out_list) > 1 else out_list[0]