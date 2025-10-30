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
from antgo.utils import *
from .util import *
import torch
import numpy as np
import onnx
import onnxruntime as ort
from multiprocessing import Process
from multiprocessing import queues
import logging
import os
import shutil
import threading
import time
import queue
import traceback


@register
class inference_onnx_op(object):
    def __init__(self, onnx_path, input_fields=None, device_id=0, max_size=5, is_dyn_batch=False, dyn_max_batch_size=16, dyn_max_cache_time=0.03, **kwargs):
        self.onnx_path = onnx_path

        self.sess_config = []
        if not isinstance(device_id, list):
            device_id = [device_id]

        # 配置推理服务引擎
        self.max_size = max_size
        for index, d_id in enumerate(device_id):
            privider = []
            if d_id >= 0 and torch.cuda.is_available():
                privider.append(('CUDAExecutionProvider', {'device_id': d_id}))
            privider.append('CPUExecutionProvider')

            sess = ort.InferenceSession(onnx_path, providers=privider)
            sess_config = {
                'sess': sess,
                'in_queue': queue.Queue(maxsize=self.max_size),
                'out_queue': queue.Queue(maxsize=self.max_size)
            }
            self.sess_config.append(sess_config)

        # 初始化推理引擎服务
        for index in range(len(self.sess_config)):
            thread = threading.Thread(
                target=self.infer,
                args=(index,),
                daemon=True
            )
            thread.start()

        # 服务轮询相关参数
        self.server_current_index = 0
        self.server_list = list(range(len(self.sess_config)))
        self.cache_out_data = {}
        self.server_robin_lock = threading.Lock()
        self.cache_out_lock = threading.Lock()

        # 动态batch构建相关参数
        self.dyn_batch_cache_size = 1
        self.dyn_batch_cache = [{'main': -1,'other': [],'start_time': 0,'data': []} for _ in range(self.dyn_batch_cache_size)]
        self.dyn_batch_cache_lock = [threading.Lock() for _ in range(self.dyn_batch_cache_size)] 

        self.dyn_max_batch_size = dyn_max_batch_size    # 最大batch=4
        self.dyn_max_cache_time = dyn_max_cache_time    # 最大等待时间 0.01秒（10毫秒）
        self.dyn_batch_dispatch_lock = threading.Lock()
        self.dyn_batch_dispatch = {}
        self.dyn_est_batch_size = dyn_max_batch_size

        # 模型基本信息
        self.input_names = []
        self.input_shapes = []
        for input_tensor in self.sess_config[0]['sess'].get_inputs():
            self.input_names.append(input_tensor.name)
            self.input_shapes.append(input_tensor.shape)

        self.output_names = []
        self.output_shapes = []
        for output_tensor in self.sess_config[0]['sess'].get_outputs():
            self.output_names.append(output_tensor.name)
            self.output_shapes.append(output_tensor.shape)

        self.is_dyn_batch = False
        if is_dyn_batch:
            if not isinstance(self.input_shapes[0][0], str):
                print('onnx model not support dynamic batch')
                self.is_dyn_batch = False
            else:
                self.is_dyn_batch = True

        self.input_fields = self.input_names
        self.mean_val = kwargs.get('mean', None)   # 均值
        self.std_val = kwargs.get('std', None)     # 方差
        self.reverse_channel = kwargs.get('reverse_channel', False)
        self.engine = kwargs.get('engine', None)
        self.engine_args = kwargs.get('engine_args', {}) 

    def _inner_infer(self, *args, sess=None):
        input_map = None
        for field, data, expected_shape in zip(self.input_fields, args, self.input_shapes):
            if data is None or data.shape[0] == 0:
                continue

            if self.mean_val is not None and self.std_val is not None and data.dtype==np.uint8 and data.shape[-1] == 3:
                # 均值方差内部处理，仅对图像类型输入有效
                if len(data.shape) == 4:
                    # NxHxWx3
                    if self.reverse_channel:
                        data = data[:,:,:,::-1]
                    data = data - np.reshape(np.array(self.mean_val), (1,1,1,3))
                    data = data / np.reshape(np.array(self.std_val), (1,1,1,3))

                    # Nx3xHxW
                    data = np.transpose(data, (0,3,1,2))
                elif len(data.shape) == 3:
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
            result = sess.run(None, group_input_map)
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

    def infer(self, index=0):
        while True:
            try:
                # print(f'server {index} queue size {self.sess_config[index]["in_queue"].qsize()}')
                # 从队列中获取数据
                in_data, thread_id = self.sess_config[index]['in_queue'].get(block=True)

                # 在设备 index 执行推理
                out_data = self._inner_infer(*in_data, sess=self.sess_config[index]['sess'])

                # 推送到输出队列
                self.sess_config[index]['out_queue'].put((out_data, thread_id))
            except Exception as e:
                traceback.print_exc()

    @batchdyn
    def __call__(self, *args):
        # 轮询服务,获得空闲服务
        selected_server = 0
        with self.server_robin_lock:
            selected_server = self.server_list[self.server_current_index]
            self.server_current_index = (self.server_current_index + 1) % len(self.server_list)

        # 送入队列 (data, thread_id)
        self.sess_config[selected_server]['in_queue'].put((args, threading.current_thread().ident))

        # 等待输出
        # !输出数据，可能非本线程等待的数据
        out_data, thread_id = self.sess_config[selected_server]['out_queue'].get()
        if thread_id != threading.current_thread().ident:
            with self.cache_out_lock:
                self.cache_out_data[thread_id] = out_data

            while True:
                # 等待1ms
                time.sleep(0.001 + float(np.random.random() * 0.002))
                # 检查是否已经处理完当前线程的数据
                with self.cache_out_lock:
                    if threading.current_thread().ident in self.cache_out_data:
                        out_data = self.cache_out_data.pop(threading.current_thread().ident)
                        break

        return out_data

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
                os.system(f'cd /tmp/onnx ; {"docker" if not is_in_colab() else "udocker --allow-root"} run --rm -v $(pwd):/workspace rknnconvert bash convert.sh --i={prefix}.onnx --quantize --image-folder=./calibration-images --o=./rknn/{prefix} --device={platform_device} --mean-values={mean_values} --std-values={std_values}')
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
                os.system(f'cd /tmp/onnx ; {"docker" if not is_in_colab() else "udocker --allow-root"} run --rm -v $(pwd):/workspace rknnconvert bash convert.sh --i={prefix}.onnx --o=./rknn/{prefix} --device={platform_device} --mean-values={mean_values} --std-values={std_values}')
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
                os.system(f'cd /tmp/onnx ; {"docker" if not is_in_colab() else "udocker --allow-root"} run --rm -v $(pwd):/workspace snpeconvert bash convert.sh --i={prefix}.onnx --o=./snpe/{prefix} --quantize --npu --data-folder=calibration-images')
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
                os.system(f'cd /tmp/onnx ; {"docker" if not is_in_colab() else "udocker --allow-root"} run --rm -v $(pwd):/workspace snpeconvert bash convert.sh --i={prefix}.onnx --o=./snpe/{prefix}')
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
            os.system(f'cd /tmp/onnx/ ; {"docker" if not is_in_colab() else "udocker --allow-root"} run --rm -v $(pwd):/workspace tnnconvert bash convert.sh --i={prefix}.onnx --o=./tnn/{prefix}')
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
    if device_id >= 0 and torch.cuda.is_available():
        sess.set_providers(['CUDAExecutionProvider'], [{'device_id': device_id}])
    else:
        sess.set_providers(['CPUExecutionProvider'])

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