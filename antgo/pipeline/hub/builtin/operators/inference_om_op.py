# -*- coding: UTF-8 -*-
# @Time    : 2022/9/21 22:42
# @File    : inference_om_op.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.pipeline.engine import *
from antgo.pipeline.models.utils.utils import *
import acl
import numpy as np
import os


# 常量定义
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_ERROR_NONE = 0

@register
class inference_om_op(object):
    is_init = False
    context = None
    stream = None
    def __init__(self, om_path, device_id=0, **kwargs):
        self.device_id = device_id
        self.model_path = om_path
        # self.context = None
        # self.stream = None
        self.model_id = None
        self.input_dataset = None
        self.output_dataset = None
        self.input_buffers = []
        self.output_buffers = []
        self.input_sizes = []
        self.output_sizes = []

        self.input_shapes = []
        self.output_shapes = []

        self.mean_val = kwargs.get('mean', None)   # 均值
        self.std_val = kwargs.get('std', None)     # 方差
        self.reverse_channel = kwargs.get('reverse_channel', False)

        self.init_acl()
        self.load_model()

    def init_acl(self):
        if not inference_om_op.is_init:
            """初始化AscendCL环境"""
            ret = acl.init()
            if ret != ACL_ERROR_NONE:
                raise Exception(f"acl init failed, error code: {ret}")

            ret = acl.rt.set_device(self.device_id)
            if ret != ACL_ERROR_NONE:
                raise Exception(f"set device {self.device_id} failed, error code: {ret}")

            inference_om_op.context, ret = acl.rt.create_context(self.device_id)
            if ret != ACL_ERROR_NONE:
                raise Exception(f"create context failed, error code: {ret}")

            inference_om_op.stream, ret = acl.rt.create_stream()
            if ret != ACL_ERROR_NONE:
                raise Exception(f"create stream failed, error code: {ret}")

            print("ACL environment initialized successfully")
            inference_om_op.is_init = True

    def load_model(self):
        """加载.om模型"""
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        if ret != ACL_ERROR_NONE:
            raise Exception(f"load model {self.model_path} failed, error code: {ret}")

        # 获取模型输入输出描述
        model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(model_desc, self.model_id)
        if ret != ACL_ERROR_NONE:
            raise Exception(f"get model desc failed, error code: {ret}")

        # 准备输入
        input_num = acl.mdl.get_num_inputs(model_desc)
        for i in range(input_num):
            dims = acl.mdl.get_input_dims(model_desc, i)
            self.input_shapes.append(dims[0]['dims'])
            size = acl.mdl.get_input_size_by_index(model_desc, i)
            self.input_sizes.append(size)

            # 创建输入缓冲区
            buffer, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            if ret != ACL_ERROR_NONE:
                raise Exception(f"malloc input buffer {i} failed, error code: {ret}")

            data_buffer = acl.create_data_buffer(buffer, size)
            if data_buffer is None:
                raise Exception(f"create input data buffer {i} failed")

            self.input_buffers.append(data_buffer)

        # 准备输出
        output_num = acl.mdl.get_num_outputs(model_desc)
        for i in range(output_num):
            dims = acl.mdl.get_output_dims(model_desc, i)
            self.output_shapes.append(dims[0]['dims'])
            size = acl.mdl.get_output_size_by_index(model_desc, i)
            self.output_sizes.append(size)

            # 创建输出缓冲区
            buffer, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
            if ret != ACL_ERROR_NONE:
                raise Exception(f"malloc output buffer {i} failed, error code: {ret}")

            data_buffer = acl.create_data_buffer(buffer, size)
            if data_buffer is None:
                raise Exception(f"create output data buffer {i} failed")

            self.output_buffers.append(data_buffer)

        # 创建数据集
        self.input_dataset = acl.mdl.create_dataset()
        if self.input_dataset is None:
            raise Exception("create input dataset failed")

        for buffer in self.input_buffers:
            _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, buffer)
            if ret != ACL_ERROR_NONE:
                raise Exception(f"add input buffer to dataset failed, error code: {ret}")

        self.output_dataset = acl.mdl.create_dataset()
        if self.output_dataset is None:
            raise Exception("create output dataset failed")

        for buffer in self.output_buffers:
            _, ret = acl.mdl.add_dataset_buffer(self.output_dataset, buffer)
            if ret != ACL_ERROR_NONE:
                raise Exception(f"add output buffer to dataset failed, error code: {ret}")

        acl.mdl.destroy_desc(model_desc)
        print(f"Model {self.model_path} loaded successfully")

    def process_input(self, input_data_list):
        """处理输入数据并拷贝到设备"""
        if len(input_data_list) != len(self.input_buffers):
            raise Exception(f"Input data count {len(input_data_list)} does not match "
                           f"model input count {len(self.input_buffers)}")

        for i, input_data in enumerate(input_data_list):
            # 确保输入数据大小与模型要求一致
            if input_data.nbytes != self.input_sizes[i]:
                raise Exception(f"Input {i} size {input_data.nbytes} does not match "
                               f"required size {self.input_sizes[i]}")

            # 将数据从主机内存拷贝到设备内存
            input_buffer = acl.get_data_buffer_addr(self.input_buffers[i])
            ret = acl.rt.memcpy(input_buffer, self.input_sizes[i],
                              input_data.ctypes.data, input_data.nbytes,
                              ACL_MEMCPY_HOST_TO_DEVICE)
            if ret != ACL_ERROR_NONE:
                raise Exception(f"Copy input {i} to device failed, error code: {ret}")

    def execute_inference(self):
        """执行模型推理"""
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        if ret != ACL_ERROR_NONE:
            raise Exception(f"Model inference failed, error code: {ret}")

        # 等待推理完成
        ret = acl.rt.synchronize_stream(self.stream)
        if ret != ACL_ERROR_NONE:
            raise Exception(f"Synchronize stream failed, error code: {ret}")

        print("Inference executed successfully")

    def get_output(self):
        """获取并处理输出结果"""
        output_list = []
        for i in range(len(self.output_buffers)):
            output_buffer = acl.get_data_buffer_addr(self.output_buffers[i])
            output_size = self.output_sizes[i]

            # 从设备内存拷贝结果到主机内存
            host_buffer = np.zeros(output_size, dtype=np.byte)
            ret = acl.rt.memcpy(host_buffer.ctypes.data, output_size,
                              output_buffer, output_size,
                              ACL_MEMCPY_DEVICE_TO_HOST)
            if ret != ACL_ERROR_NONE:
                raise Exception(f"Copy output {i} to host failed, error code: {ret}")

            data = np.frombuffer(host_buffer.data, dtype=np.float32).reshape(self.output_shapes[i])
            output_list.append(data)

        return output_list

    def release_resource(self):
        """释放所有资源"""
        # 释放数据集和缓冲区
        if self.input_dataset:
            for buffer in self.input_buffers:
                if buffer:
                    acl.destroy_data_buffer(buffer)
            acl.mdl.destroy_dataset(self.input_dataset)

        if self.output_dataset:
            for buffer in self.output_buffers:
                if buffer:
                    acl.destroy_data_buffer(buffer)
            acl.mdl.destroy_dataset(self.output_dataset)

        # 卸载模型
        if self.model_id:
            acl.mdl.unload(self.model_id)

        # 释放流和上下文
        if inference_om_op.stream:
            acl.rt.destroy_stream(inference_om_op.stream)

        if inference_om_op.context:
            acl.rt.destroy_context(inference_om_op.context)

        # 重置设备
        acl.rt.reset_device(self.device_id)

        # 最终释放ACL资源
        acl.finalize()
        print(f"All resources {self.model_path} released")

    def __call__(self, *args):
        input_map = None
        for data, expected_shape in zip(args, self.input_shapes):
            if data.shape[0] == 0:
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
                    input_map = [[] for _ in range(group_num)]

                for group_i in range(group_num):
                    input_map[group_i].append(data[group_i*expected_shape[0]:(group_i+1)*expected_shape[0]])
            else:
                if input_map is None:
                    input_map = [[]]
                input_map[0].append(data)

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

        # run
        group_output = []
        for group_input_map in input_map:
            for data in group_input_map:
                print(data.shape)

            # 设置输入
            self.process_input(group_input_map)
            # 执行模型
            self.execute_inference()
            # 获得输出
            result = self.get_output()
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
