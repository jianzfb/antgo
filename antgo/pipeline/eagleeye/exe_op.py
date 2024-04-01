import os
import sys
import copy
from typing import Any
import numpy as np
import json
import shutil
import fcntl
import select
import struct
from subprocess import run, Popen, PIPE, DEVNULL, STDOUT


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
        
        # if project_info['platform'] == 'linux':
        #     self.data_folder = os.path.join(self.folder, 'deploy', f'{self.plugin_name}_plugin', 'bin', 'X86-64')
        self.project_input_info = project_info['input']
        self.project_output_info = project_info['output']
        self.project_graph_info = project_info['graph']
        self.project_platform_info = project_info['platform']

        # 准备运行环境
        # step 1: 依赖库文件
        os.system(f'cd {self.project_folder} && bash setup.sh')

        # step 2: 创建输入输出目录
        os.makedirs(os.path.join(self.data_folder, 'data', 'input'), exist_ok=True)
        os.makedirs(os.path.join(self.data_folder, 'data', 'output'), exist_ok=True)

        self.proc = None
        self.readable_fds = None
        self.stdout_fd = None

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
                run_args.append(f'placeholder_{arg_i}/{data_shape_code}/{data_type_code}')

        if self.proc is None:
            if self.project_platform_info == "linux":
                command = f'{self.project_folder}/bin/X86-64/{self.plugin_name}_demo'
                self.proc = Popen([command] + ['stdinout'] + run_args, stdin=PIPE, stderr=PIPE, stdout=PIPE, text=False)
            elif self.project_platform_info == "android":
                command = f'adb shell "cd /data/local/tmp/{self.plugin_name}; export LD_LIBRARY_PATH=.; ./{self.plugin_name}_demo stdinout '+' '.join(run_args)+'"'
                self.proc = Popen([command], stdin=PIPE, stderr=PIPE, stdout=PIPE, text=False, shell=True)

            # 获取stdout的文件描述符
            self.stdout_fd = self.proc.stdout.fileno()

            # 设置stdout为非阻塞模式
            flags = fcntl.fcntl(self.stdout_fd, fcntl.F_GETFL)
            fcntl.fcntl(self.stdout_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

            # 定义select()等待的文件描述符列表
            self.readable_fds = [self.stdout_fd]

        for arg_i, arg_value in enumerate(args):
            self.proc.stdin.write(arg_value.tobytes())

        # 解析输出数据
        out_list = [None for _ in range(len(self.project_output_info))]

        out_i = 0
        while out_i < len(out_list):
            is_finish = False
            is_first_packet = True
            packet_size = 0
            packet_data = b''
            receive_size = 0
            data_type_code = -1
            data_dim_code = -1
            shape_list = []
            while True:
                ready_fds, _, _ = select.select(self.readable_fds, [], [])

                for fd in ready_fds:
                    if fd == self.stdout_fd:
                        value = self.proc.stdout.read()
                        if value:
                            if is_first_packet:
                                # 数据类型标记
                                data_type_code, = struct.unpack('<Q', value[0:8])
                                # 数据维度
                                data_dim_code, = struct.unpack('<Q', value[8:16])
                                # 数据shape
                                packet_size = 1
                                for dim_i in range(data_dim_code):
                                    t, = struct.unpack('<Q', value[16+8*dim_i:16+8*(dim_i+1)])
                                    shape_list.append(t)
                                    packet_size *= t

                                packet_size += 8*(2+data_dim_code)
                                is_first_packet = False
                            
                            receive_size += len(value)
                            packet_data += value
                            if packet_size == receive_size:
                                is_finish = True
                    if is_finish:
                        break
                if is_finish:
                    break 

            if data_type_code == 0:
                # int8
                data = np.frombuffer(packet_data[8*(2+data_dim_code):], np.int8)
                data = data.reshape(shape_list)
                out_list[out_i] = data
            elif data_type_code == 1 or data_type_code == 8 or data_type_code == 9:
                # uint8
                data = np.frombuffer(packet_data[8*(2+data_dim_code):], np.uint8)
                data = data.reshape(shape_list)
                out_list[out_i] = data
            elif data_type_code == 4:
                # int32
                data = np.frombuffer(packet_data[8*(2+data_dim_code):], np.int32)
                data = data.reshape(shape_list)
                out_list[out_i] = data
            elif data_type_code == 5:
                # uint32
                data = np.frombuffer(packet_data[8*(2+data_dim_code):], np.uint32)
                data = data.reshape(shape_list)
                out_list[out_i] = data
            elif data_type_code == 6:
                # float32
                data = np.frombuffer(packet_data[8*(2+data_dim_code):], np.float32)
                data = data.reshape(shape_list)
                out_list[out_i] = data
            elif data_type_code == 7:
                # double
                data = np.frombuffer(packet_data[8*(2+data_dim_code):], np.float64)
                data = data.reshape(shape_list)
                out_list[out_i] = data
            elif data_type_code == 10:
                # bool
                data = np.frombuffer(packet_data[8*(2+data_dim_code):], np.bool)
                data = data.reshape(shape_list)
                out_list[out_i] = data

            out_i += 1
            if out_i >= len(out_list):
                break

        return out_list if len(out_list) > 1 else out_list[0]