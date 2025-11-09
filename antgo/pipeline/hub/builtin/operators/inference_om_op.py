# -*- coding: UTF-8 -*-
# @Time    : 2022/9/21 22:42
# @File    : inference_om_op.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.pipeline.engine import *
from antgo.pipeline.models.utils.utils import *
from antgo.utils import *
from .util import *
import acl
import numpy as np
import os
import threading
import logging
import cv2
import queue
import traceback
import copy

# 常量定义
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_ERROR_NONE = 0

@register
class inference_om_op(object):
    is_init = False
    def __init__(self, om_path, device_id=0, max_size=5, is_dyn_batch=False, dyn_max_batch_size=[1], dyn_max_cache_time=0.03, **kwargs):
        self.device_id = device_id
        self.model_path = om_path

        self.is_dyn_batch = is_dyn_batch
        self.mean_val = kwargs.get('mean', None)   # 均值
        self.std_val = kwargs.get('std', None)     # 方差
        self.reverse_channel = kwargs.get('reverse_channel', False)
        if not isinstance(device_id, list):
            device_id = [device_id]

        # 配置推理服务引擎
        if not inference_om_op.is_init:
            # 全局（当前进程）仅进行一次
            ret = acl.init()
            if ret != ACL_ERROR_NONE:
                raise Exception(f"acl init failed, error code: {ret}")
            inference_om_op.is_init = True

        self.max_size = max_size
        # 初始化设备上下文，并启动推理线程
        self.sess_config = []
        for index, d_id in enumerate(device_id):
            self.sess_config.append({
                'in_queue': queue.Queue(maxsize=self.max_size),
                'out_queue': queue.Queue(maxsize=self.max_size)
            })
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

        if not isinstance(dyn_max_batch_size, list) and not isinstance(dyn_max_batch_size, tuple):
            dyn_max_batch_size = [dyn_max_batch_size]
        self.dyn_max_batch_size = max(dyn_max_batch_size)    # 最大batch=4
        self.dyn_allow_batch_sizes = dyn_max_batch_size
        self.dyn_max_cache_time = dyn_max_cache_time    # 最大等待时间 0.01秒（10毫秒）
        self.dyn_batch_dispatch_lock = threading.Lock()
        self.dyn_batch_dispatch = {}

    def infer(self, device_id):
        # --------------------------------  模型初始化 -------------------------------------- #
        logging.info(f'init ascend device {device_id}')
        ret = acl.rt.set_device(device_id)
        if ret != ACL_ERROR_NONE:
            raise Exception(f"set device {device_id} failed, error code: {ret}")

        context, ret = acl.rt.create_context(device_id)
        if ret != ACL_ERROR_NONE:
            raise Exception(f"create context failed, error code: {ret}")

        stream, ret = acl.rt.create_stream()
        if ret != ACL_ERROR_NONE:
            raise Exception(f"create stream failed, error code: {ret}")

        logging.info(f'load model {self.model_path} for device {device_id}')
        model_id, ret = acl.mdl.load_from_file(self.model_path)
        if ret != ACL_ERROR_NONE:
            raise Exception(f"load model {self.model_path} failed, error code: {ret}")

        # 获取模型输入输出描述
        model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(model_desc, model_id)
        if ret != ACL_ERROR_NONE:
            raise Exception(f"get model desc failed, error code: {ret}")

        # 准备输入
        input_shapes = []
        input_sizes = []
        input_buffers = []
        input_num = acl.mdl.get_num_inputs(model_desc)
        for i in range(input_num):
            dims = acl.mdl.get_input_dims(model_desc, i)
            input_shapes.append(dims[0]['dims'])
            size = acl.mdl.get_input_size_by_index(model_desc, i)
            input_sizes.append(size)

            # 创建输入缓冲区
            buffer, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            if ret != ACL_ERROR_NONE:
                raise Exception(f"malloc input buffer {i} failed, error code: {ret}")

            data_buffer = acl.create_data_buffer(buffer, size)
            if data_buffer is None:
                raise Exception(f"create input data buffer {i} failed")

            input_buffers.append(data_buffer)

        logging.info(f'input shapes {input_shapes} sizes {input_sizes}')

        # 准备输出
        output_shapes = []
        output_sizes = []
        output_buffers = []
        output_num = acl.mdl.get_num_outputs(model_desc)
        for i in range(output_num):
            dims = acl.mdl.get_output_dims(model_desc, i)
            output_shapes.append(dims[0]['dims'])
            size = acl.mdl.get_output_size_by_index(model_desc, i)
            output_sizes.append(size)

            # 创建输出缓冲区
            buffer, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            if ret != ACL_ERROR_NONE:
                raise Exception(f"malloc output buffer {i} failed, error code: {ret}")

            data_buffer = acl.create_data_buffer(buffer, size)
            if data_buffer is None:
                raise Exception(f"create output data buffer {i} failed")

            output_buffers.append(data_buffer)

        logging.info(f'output shapes {output_shapes} sizes {output_sizes}')

        # 创建数据集
        input_dataset = acl.mdl.create_dataset()
        if input_dataset is None:
            raise Exception("create input dataset failed")

        for buffer in input_buffers:
            _, ret = acl.mdl.add_dataset_buffer(input_dataset, buffer)
            if ret != ACL_ERROR_NONE:
                raise Exception(f"add input buffer to dataset failed, error code: {ret}")

        output_dataset = acl.mdl.create_dataset()
        if output_dataset is None:
            raise Exception("create output dataset failed")

        for buffer in output_buffers:
            _, ret = acl.mdl.add_dataset_buffer(output_dataset, buffer)
            if ret != ACL_ERROR_NONE:
                raise Exception(f"add output buffer to dataset failed, error code: {ret}")

        # 获得动态维度索引
        dynamic_idx = 0
        if self.is_dyn_batch:
            dynamic_idx, ret = acl.mdl.get_input_index_by_name(model_desc, "ascend_mbatch_shape_data")
            if ret != ACL_ERROR_NONE:
                raise Exception(f"fail to set dynamic batchsize {current_batch_size}, error code: {ret}")

        acl.mdl.destroy_desc(model_desc)
        logging.info(f"Ascend Model {self.model_path} loaded successfully")

        # -------------------------------- 模型推理 ----------------------------------- #
        # 等待数据，进行推理
        logging.info(f'Ascend device {device_id}, waiting process data.')
        while True:
            try:
                # 输入队列拾取数据
                in_data, thread_id = self.sess_config[device_id]['in_queue'].get(block=True)

                input_map = None
                for data, expected_shape in zip(in_data, input_shapes):
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

                    if len(data.shape) != 4:
                        raise Exception("input data shape not NCHW")

                    if not self.is_dyn_batch:
                        # 静态batch
                        group_num = data.shape[0] // expected_shape[0]
                        if input_map is None:
                            input_map = [[] for _ in range(group_num)]

                        for group_i in range(group_num):
                            input_map[group_i].append(data[group_i*expected_shape[0]:(group_i+1)*expected_shape[0]])
                    else:
                        # 动态batch
                        if input_map is None:
                            input_map = [[]]
                        input_map[0].append(data)

                if input_map is None:
                    if len(output_shapes) == 1:
                        return np.empty([0]*len(output_shapes[0]), dtype=np.float32)
                    else:
                        oo = []
                        for i in range(len(output_shapes)):
                            oo.append(
                                np.empty([0]*len(output_shapes[i]), dtype=np.float32)
                            )
                        return oo

                # 模型运行
                group_output = []
                for group_input_map in input_map:
                    for data in group_input_map:
                        print(data.shape)

                    # 准备数据
                    if self.is_dyn_batch:
                        current_batch_size = group_input_map[0].shape[0]
                        ret = acl.mdl.set_dynamic_batch_size(model_id, input_dataset, dynamic_idx, current_batch_size)
                        if ret != ACL_ERROR_NONE:
                            raise Exception(f"fail to set dynamic batchsize {current_batch_size}, error code: {ret}")

                    for i, input_data in enumerate(group_input_map):
                        # 将数据从主机内存拷贝到设备内存
                        input_buffer = acl.get_data_buffer_addr(input_buffers[i])
                        ret = acl.rt.memcpy(input_buffer, input_data.nbytes,
                                        input_data.ctypes.data, input_data.nbytes,
                                        ACL_MEMCPY_HOST_TO_DEVICE)
                        if ret != ACL_ERROR_NONE:
                            raise Exception(f"Copy input {i} to device failed, error code: {ret}")

                    # 执行模型
                    ret = acl.mdl.execute(model_id, input_dataset, output_dataset)
                    if ret != ACL_ERROR_NONE:
                        raise Exception(f"Model inference failed, error code: {ret}")

                    ret = acl.rt.synchronize_stream(stream)
                    if ret != ACL_ERROR_NONE:
                        raise Exception(f"Synchronize stream failed, error code: {ret}")

                    # 获得输出
                    output_list = []
                    for i in range(len(output_buffers)):
                        output_buffer = acl.get_data_buffer_addr(output_buffers[i])
                        output_size = copy.deepcopy(output_sizes[i])
                        output_shape = copy.deepcopy(output_shapes[i])
                        if self.is_dyn_batch:
                            output_size = int(output_size / output_shape[0] * current_batch_size)
                            output_shape[0] = current_batch_size

                        # 从设备内存拷贝结果到主机内存
                        host_buffer = np.zeros(output_size, dtype=np.byte)
                        ret = acl.rt.memcpy(host_buffer.ctypes.data, output_size,
                                        output_buffer, output_size,
                                        ACL_MEMCPY_DEVICE_TO_HOST)
                        if ret != ACL_ERROR_NONE:
                            raise Exception(f"Copy output {i} to host failed, error code: {ret}")

                        # 重置形状为目标性状
                        data = np.frombuffer(host_buffer.data, dtype=np.float32).reshape(output_shape)
                        output_list.append(data)

                    group_output.append(output_list)

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

                # 输出队列写入数据
                out_data = output
                if isinstance(output, list) or isinstance(output, tuple):
                    if len(output) == 1:
                        out_data = output[0]
                        self.sess_config[device_id]['out_queue'].put((out_data, thread_id))
                        continue

                    out_data = tuple(output)            
                self.sess_config[device_id]['out_queue'].put((out_data, thread_id))
            except Exception as e:
                traceback.print_exc()

        # ---------------------------   资源销毁   -------------------------------- #
        logging.info("relase ascend resource")
        # 释放数据集和缓冲区
        if input_dataset:
            for buffer in input_buffers:
                if buffer:
                    acl.destroy_data_buffer(buffer)
            acl.mdl.destroy_dataset(input_dataset)

        if output_dataset:
            for buffer in output_buffers:
                if buffer:
                    acl.destroy_data_buffer(buffer)
            acl.mdl.destroy_dataset(output_dataset)

        # 卸载模型
        if model_id:
            acl.mdl.unload(model_id)

        # 释放流和上下文
        if stream:
            acl.rt.destroy_stream(stream)

        if context:
            acl.rt.destroy_context(context)

        # 重置设备
        acl.rt.reset_device(device_id)

        # 最终释放ACL资源
        # acl.finalize()
        logging.info(f"All resources {self.model_path} released")

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


# image = cv2.imread('./aa.jpeg')
# image = cv2.resize(image, (640,640))
# aabb = inference_om_op(om_path='./om_last_static.om', device_id=0, max_size=5, is_dyn_batch=False, mean=[128,128,128], std=[128,128,128])
# ccdd = aabb(image)
# print(ccdd.shape)