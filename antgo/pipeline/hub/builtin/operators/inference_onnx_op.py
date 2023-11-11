# -*- coding: UTF-8 -*-
# @Time    : 2022/9/21 22:42
# @File    : inference_onnx_op.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.pipeline.engine import *
from antgo.pipeline.models.utils.utils import *
from antgo.interactcontext import InteractContext
import torch
import numpy as np
import onnx
import onnxruntime as ort
from multiprocessing import Process
from multiprocessing import queues
import logging
import os
import shutil


@register
class inference_onnx_op(object):
    def __init__(self, onnx_path, input_fields=None, device_id=-1, **kwargs):
        privider = []
        if device_id >= 0:
            privider.append(('CUDAExecutionProvider', {'device_id': device_id}))
        privider.append('CPUExecutionProvider')

        self.onnx_path = onnx_path
        self.sess = ort.InferenceSession(onnx_path, providers=privider)
        print(self.sess.get_providers())

        self.input_names = []
        self.input_shapes = []
        for input_tensor in self.sess.get_inputs():
            self.input_names.append(input_tensor.name)
            self.input_shapes.append(input_tensor.shape)

        self.output_names = []
        self.output_shapes = []
        for output_tensor in self.sess.get_outputs():
            self.output_names.append(output_tensor.name)
            self.output_shapes.append(output_tensor.shape)

        self.input_fields = self.input_names
        self.mean_val = kwargs.get('mean', None)   # 均值
        self.std_val = kwargs.get('std', None)     # 方差
        self.reverse_channel = kwargs.get('reverse_channel', False)
        self.engine = kwargs.get('engine', None)
        self.engine_args = kwargs.get('engine_args', {}) 

    def __call__(self, *args):        
        input_map = None
        for field, data, expected_shape in zip(self.input_fields, args, self.input_shapes):
            if data.shape[0] == 0:
                continue

            if self.mean_val is not None:
                if len(data.shape) == 4:
                    # NxHxWx3
                    if self.reverse_channel:
                        data = data[:,:,:,::-1]
                    data = data - np.reshape(np.array(self.mean_val), (1,1,1,3))
                    data = data / np.reshape(np.array(self.std_val), (1,1,1,3))

                    # Nx3xHxW
                    data = np.transpose(data, (0,3,1,2))
                else:
                    # HxWx3
                    if self.reverse_channel:
                        data = data[:,:,::-1]
                    data = data - np.reshape(np.array(self.mean_val), (1,1,3))
                    data = data / np.reshape(np.array(self.std_val), (1,1,3))

                    # -> 1xHxWx3
                    data = np.expand_dims(data, 0)
                    # -> 1x3xHxW
                    data = np.transpose(data, (0,3,1,2))

            if data.dtype != np.float32:
                data = data.astype(np.float32)

            if not isinstance(expected_shape[0], str):
                group_num = data.shape[0] // expected_shape[0]
                if input_map is None:
                    input_map = [{} for _ in range(group_num)]

                for group_i in range(group_num):
                    input_map[group_i][field] = data[group_i*expected_shape[0]:(group_i+1)*expected_shape[0]]
            else:
                if input_map is None:
                    input_map = [{}]
                input_map[0][field] = data

        if input_map is None:
            if len(self.output_shapes) == 1:
                return np.empty([0]*len(self.output_shapes[0]), dtype=np.float32)
            else:
                oo = []
                for i in range(len(self.output_shapes)):
                    oo.append(
                        np.empty([0]*len(self.output_shapes[i]), dtype=np.float32)
                    )
                return oo

        group_output = []
        for group_input_map in input_map:
            result = self.sess.run(None, group_input_map)
            group_output.append(result)

        output = None
        if len(group_output) == 1:
            output = group_output[0]
        else:
            if isinstance(group_output[0], tuple) or isinstance(group_output[0], list):
                output = []
                for elem_i in range(len(group_output[0])):
                    rr = []
                    for group_i in range(len(group_output)):
                        rr.append(group_output[group_i][elem_i])
                    
                    rr = np.concatenate(rr, 0)
                    output.append(rr)
            else:
                output = np.concatenate(group_output, 0)

        if isinstance(output, list) or isinstance(output, tuple):
            if len(output) == 1:
                return output[0]

            return tuple(output)
        
        return output
    
    def export(self):
        if self.engine is None:
            logging.error(f'engine must be set.')
            return

        platform_device = self.engine_args.get('device', None)
        assert(platform_device is not None) 
        if self.engine == 'rknn':
            if self.engine_args.get('quantize', False):
                # 转量化模型
                os.system(f'mkdir /tmp/onnx ; mkdir /tmp/onnx/rknn ; cp {self.onnx_path} /tmp/onnx/')
                # 确保存在校正数据集
                assert(os.path.exists(self.engine_args.get('calibration-images')))
                shutil.copytree(self.engine_args.get('calibration-images'), '/tmp/onnx/calibration-images')

                prefix = os.path.basename(self.onnx_path)[:-5]
                onnx_dir_path = os.path.dirname(self.onnx_path)
                mean_values = ','.join([str(v) for v in self.mean_val])
                std_values = ','.join([str(v) for v in self.std_val])
                os.system(f'cd /tmp/onnx ; docker run --rm -v $(pwd):/workspace rknnconvert bash convert.sh --i={prefix}.onnx --quantize --image-folder=./calibration-images --o=./rknn/{prefix} --device={platform_device} --mean-values={mean_values} --std-values={std_values}')
                converted_model_file = ''
                for file_name in os.listdir('/tmp/onnx/rknn/'):
                    if file_name[0] != '.':
                        converted_model_file = file_name
                        break

                os.system(f'cp -r /tmp/onnx/rknn/* {onnx_dir_path} ; rm -rf /tmp/onnx/')
            else:
                # 转浮点模型
                os.system(f'mkdir /tmp/onnx ; mkdir /tmp/onnx/rknn ; cp {self.onnx_path} /tmp/onnx/')
                
                prefix = os.path.basename(self.onnx_path)[:-5]
                onnx_dir_path = os.path.dirname(self.onnx_path)
                mean_values = ','.join([str(v) for v in  self.mean_val])
                std_values = ','.join([str(v) for v in  self.std_val])
                os.system(f'cd /tmp/onnx ; docker run --rm -v $(pwd):/workspace rknnconvert bash convert.sh --i={prefix}.onnx --o=./rknn/{prefix} --device={platform_device} --mean-values={mean_values} --std-values={std_values}')
                converted_model_file = ''
                for file_name in os.listdir('/tmp/onnx/rknn/'):
                    if file_name[0] != '.':
                        converted_model_file = file_name
                        break
                
                os.system(f'cp -r /tmp/onnx/rknn/* {onnx_dir_path} ; rm -rf /tmp/onnx/')
        elif self.engine == 'snpe':
            if self.engine_args.get('quantize', False):
                # 转量化模型
                os.system(f'mkdir /tmp/onnx ; mkdir /tmp/onnx/snpe ; cp {self.onnx_path} /tmp/onnx/')
                # 确保存在校正数据集
                assert(os.path.exists(platform_engine_args.get('calibration-images')))
                shutil.copytree(platform_engine_args.get('calibration-images'), '/tmp/onnx/calibration-images')

                prefix = os.path.basename(self.onnx_path)[:-5]
                onnx_dir_path = os.path.dirname(self.onnx_path)
                os.system(f'cd /tmp/onnx ; docker run --rm -v $(pwd):/workspace snpeconvert bash convert.sh --i={prefix}.onnx --o=./snpe/{prefix} --quantize --npu --data-folder=calibration-images')
                converted_model_file = ''
                for file_name in os.listdir('/tmp/onnx/snpe/'):
                    if file_name[0] != '.':
                        converted_model_file = file_name
                        break

                os.system(f'cp -r /tmp/onnx/snpe/* {onnx_dir_path} ; rm -rf /tmp/onnx/')
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
        elif self.engine == 'tnn':
            os.system(f'mkdir /tmp/onnx ; mkdir /tmp/onnx/tnn ; cp {self.onnx_path} /tmp/onnx/')                 
            prefix = os.path.basename(self.onnx_path)[:-5]
            onnx_dir_path = os.path.dirname(self.onnx_path)
            os.system(f'cd /tmp/onnx/ ; docker run --rm -v $(pwd):/workspace tnnconvert bash convert.sh --i={prefix}.onnx --o=./tnn/{prefix}')
            converted_model_file = []
            for file_name in os.listdir('/tmp/onnx/tnn/'):
                if file_name[0] != '.' and '.tnnproto' in file_name:
                    converted_model_file = file_name
                    break

            os.system(f'cp -r /tmp/onnx/tnn/* {onnx_dir_path} ; rm -rf /tmp/onnx/')
        else:
            logging.error(f'Dont support engine {self.engine}')


def __session_run_in_process(onnx_path, device_id, input_fields, input_queue, output_queue):
    sess = ort.InferenceSession(onnx_path)
    sess.set_providers(['CUDAExecutionProvider'], [{'device_id': device_id}])

    while True:
        data = input_queue.get()
        if data is None:
            break

        input_map = {}
        for field, data in zip(input_fields, data):
            data = np.transpose(data, [2,0,1])
            input_map[field] = data[np.newaxis,:,:,:].astype(np.float32)
            
        output = sess.run(None, input_map)
        if isinstance(output, list) or isinstance(output, tuple):
            output_queue.put(tuple(output))
            continue
        
        output_queue.put(tuple(output))


@register
class ensemble_onnx_op(object):
    def __init__(self, onnx_path_list, input_fields, device_ids) -> None:
        self.input_fields = input_fields
        assert(self.backend in ['process', 'thread'])
        
        self.input_queues =[queues.Queue() for _ in range(len(onnx_path_list))]
        self.output_queues = [queues.Queue() for _ in range(len(onnx_path_list))]
        self.processes = []
        self.multi_process(self.onnx_path_list, device_ids, self.input_queues, self.output_queues)

    def multi_process(self, onnx_path_list, device_ids, input_queues, output_queues):
        self.processes = []
        for onnx_path, device_id in zip(onnx_path_list, device_ids):
            process = Process(target=__session_run_in_process, args=(onnx_path, device_id, self.input_fields, input_queues, output_queues))
            process.daemon = True
            process.start()
            self.processes.append(process)

    def __call__(self, *args):
        # pass
        for input_queue in self.input_queues:
            input_queue.put(args)
        
        ensemble_result = []
        for output_queue in self.output_queues:
            result = output_queue.get()
            if len(ensemble_result) == 0:
                ensemble_result = [[] for _ in range(len(result))]
            for i, v in enumerate(result):
                ensemble_result[i].append(v)

        out = []
        for i, v in enumerate(ensemble_result):
            out.append(np.mean(np.stack(v, 0), axis=0, keepdims=False))
        return tuple(out)


@register
class ensemble_shell_op(object):
    def __init__(self, shell_script_list, parse_fields=[], device_ids=None):
        self.parse_fields = parse_fields
        self.device_ids = device_ids
        self.ctx = InteractContext()
        self.ctx.ensemble.start("A", config = {
                'weight': 1.0,              # ignore
                'role': 'master',
                'worker': len(shell_script_list),
                'stage': 'merge',
                'feedback': True,
                'background': False})        

        for shell_script in shell_script_list:
            logging.info(f'Shell script {shell_script}')
            process = Process(target=os.system, args=(shell_script,))
            process.daemon = True
            process.start()
    
    def __call__(self, *args):
        # 推送到聚合服务
        if len(args) == 1:
            self.ctx.recorder.put(args[0])
        else:
            self.ctx.recorder.put(args)
        
        # 获得聚合结果
        result = self.ctx.recorder.avg()
        # k,v
        ss = []
        for field in self.parse_fields:
            ss.append(result[field])
        return ss