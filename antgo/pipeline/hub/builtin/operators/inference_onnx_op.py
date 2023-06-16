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


@register
class inference_onnx_op(object):
    def __init__(self, onnx_path, input_fields=None, device_id=-1, **kwargs):
        self.sess = ort.InferenceSession(onnx_path)
        self.input_names = []
        self.input_shapes = []
        for input_tensor in self.sess.get_inputs():
            self.input_names.append(input_tensor.name)
            self.input_shapes.append(input_tensor.shape)

        if device_id >= 0:
            self.sess.set_providers(['CUDAExecutionProvider'], [{'device_id': device_id}])
        self.input_fields = self.input_names
        self.mean_val = kwargs.get('mean', None)   # 均值
        self.std_val = kwargs.get('std', None)     # 方差
        self.rgb2bgr = kwargs.get('rgb2bgr', False)

    def __call__(self, *args):        
        input_map = None
        for field, data, expected_shape in zip(self.input_fields, args, self.input_shapes):
            if self.mean_val is not None:
                if len(data.shape) == 4:
                    # NxHxWx3
                    if self.rgb2bgr:
                        data = data[:,:,:,::-1]
                    data = data - np.reshape(np.array(self.mean_val), (1,1,1,3))
                    data = data / np.reshape(np.array(self.std_val), (1,1,1,3))

                    # Nx3xHxW
                    data = np.transpose(data, (0,3,1,2))
                else:
                    # HxWx3
                    if self.rgb2bgr:
                        data = data[:,:,::-1]
                    data = data - np.reshape(np.array(self.mean_val), (1,1,3))
                    data = data / np.reshape(np.array(self.std_val), (1,1,3))

                    # -> 1xHxWx3
                    data = np.expand_dims(data, 0)
                    # -> 1x3xHxW
                    data = np.transpose(data, (0,3,1,2))

            if data.dtype != np.float32:
                data = data.astype(np.float32)

            group_num = data.shape[0] // expected_shape[0]
            if input_map is None:
                input_map = [{} for _ in range(group_num)]

            for group_i in range(group_num):
                input_map[group_i][field] = data[group_i*expected_shape[0]:(group_i+1)*expected_shape[0]]

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
            return tuple(output)
        
        return (output,)
    

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