# -*- coding: UTF-8 -*-
# @Time    : 2025/10/26 22:42
# @File    : util.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import threading
import time
import functools


def batchdyn(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # 空数据直接返回
        if args[0] is None or args[0].shape[0] == 0:
            if len(self.output_shapes) == 1:
                return np.empty([0]*len(self.output_shapes[0]), dtype=np.float32)
            else:
                oo = []
                for i in range(len(self.output_shapes)):
                    oo.append(
                        np.empty([0]*len(self.output_shapes[i]), dtype=np.float32)
                    )
                return oo

        if not self.is_dyn_batch:
            # 非动态batch模式
            return func(self, *args)

        # 动态batch模式
        dyn_batch_cache_i = 0
        main_thread_id = -1
        self.dyn_batch_cache_lock[dyn_batch_cache_i].acquire()
        while True:
            if len(self.dyn_batch_cache[dyn_batch_cache_i]['data']) >= self.dyn_max_batch_size:
                # 超过最大batch_size, 需要等待
                # 等待10ms
                self.dyn_batch_cache_lock[dyn_batch_cache_i].release()
                time.sleep(0.01 + float(np.random.random() * 0.003))
                self.dyn_batch_cache_lock[dyn_batch_cache_i].acquire()
                continue

            if len(self.dyn_batch_cache[dyn_batch_cache_i]['data']) == 0:
                main_thread_id = threading.current_thread().ident
                self.dyn_batch_cache[dyn_batch_cache_i]['start_time'] = time.time()
                self.dyn_batch_cache[dyn_batch_cache_i]['main'] = main_thread_id
            else:
                main_thread_id = self.dyn_batch_cache[dyn_batch_cache_i]['main']

            self.dyn_batch_cache[dyn_batch_cache_i]['other'].append(threading.current_thread().ident)
            self.dyn_batch_cache[dyn_batch_cache_i]['data'].append(args)
            self.dyn_batch_cache_lock[dyn_batch_cache_i].release()
            break

        # 非batch 主线程，等待处理结果
        if threading.current_thread().ident != main_thread_id:
            # 当前线程非处理的主线程，仅需等待处理结果
            while True:
                is_ready = False
                with self.dyn_batch_dispatch_lock:
                    is_ready = threading.current_thread().ident in self.dyn_batch_dispatch
                if not is_ready:
                    # 等待2ms
                    time.sleep(0.002 + float(np.random.random() * 0.003))
                    continue
                break

            with self.dyn_batch_dispatch_lock:
                out_data = self.dyn_batch_dispatch.pop(threading.current_thread().ident)
            return out_data

        # batch 主线程，等待batch数据
        dyn_batch_cache = None
        while True:
            self.dyn_batch_cache_lock[dyn_batch_cache_i].acquire()
            if (len(self.dyn_batch_cache[dyn_batch_cache_i]['data']) < self.dyn_max_batch_size) and \
                ((time.time() - self.dyn_batch_cache[dyn_batch_cache_i]['start_time']) < self.dyn_max_cache_time):
                self.dyn_batch_cache_lock[dyn_batch_cache_i].release()
                # 等待 2ms
                time.sleep(0.002 + float(np.random.random() * 0.003))
                continue
            if len(self.dyn_allow_batch_sizes) > 0:
                if len(self.dyn_batch_cache[dyn_batch_cache_i]['data']) not in self.dyn_allow_batch_sizes:
                    self.dyn_batch_cache_lock[dyn_batch_cache_i].release()
                    # 等待 2ms
                    time.sleep(0.002 + float(np.random.random() * 0.003))                
                    continue

            # 组织完成一组新batch
            dyn_batch_cache = self.dyn_batch_cache[dyn_batch_cache_i]
            # 重置
            self.dyn_batch_cache[dyn_batch_cache_i] = {
                'main': -1,
                'other': [],
                'start_time': 0,
                'data': []
            }
            self.dyn_batch_cache_lock[dyn_batch_cache_i].release()
            break

        # 主处理线程，执行这里
        # print(f'batch size {len(dyn_batch_cache["data"])}')
        batch_args = [[] for _ in range(len(dyn_batch_cache['data'][0]))]
        for data in dyn_batch_cache['data']:
            for part_i, part_data in enumerate(data):
                batch_args[part_i].append(part_data)

        split_num_list = [1] * len(dyn_batch_cache['data'])
        if len(batch_args[0][0]) == 4 or len(batch_args[0][0]) == 2:
            split_num_list = [batch_args[0][0].shape[0]] * len(dyn_batch_cache['data'])
        for part_i in range(len(batch_args)):
            if len(batch_args[part_i][0]) == 4 or len(batch_args[part_i][0]) == 2:
                # NHWC/NCHW/NC -> NHWC/NCHW/NC
                batch_args[part_i] = np.concatenate(batch_args[part_i], 0)
            else:
                # HWC/C -> NHWC/NC
                batch_args[part_i] = np.stack(batch_args[part_i], 0)

        # 推送到后端服务，同步调用
        batch_out_data = func(self, *batch_args)

        # 拆分结果数据
        out_data = None
        with self.dyn_batch_dispatch_lock:
            if isinstance(batch_out_data, list) or isinstance(batch_out_data, tuple):
                out_data = tuple()
                for part_data in batch_out_data:
                    offset_start = 0
                    for thread_index, thread_id in enumerate(dyn_batch_cache['other']):
                        offset_end = offset_start + split_num_list[thread_index]
                        if thread_id == dyn_batch_cache['main']:
                            out_data += (part_data[offset_start:offset_end],)
                        else:
                            if thread_id not in self.dyn_batch_dispatch:
                                self.dyn_batch_dispatch[thread_id] = tuple()
                            self.dyn_batch_dispatch[thread_id] += (part_data[offset_start:offset_end],)

                        offset_start = offset_end
            else:
                offset_start = 0
                for thread_index, thread_id in enumerate(dyn_batch_cache['other']):
                    offset_end = offset_start + split_num_list[thread_index]
                    if thread_id == dyn_batch_cache['main']:
                        out_data = batch_out_data[offset_start:offset_end]
                    else:
                        if thread_id not in self.dyn_batch_dispatch:
                            self.dyn_batch_dispatch[thread_id] = tuple()
                        self.dyn_batch_dispatch[thread_id] = batch_out_data[offset_start:offset_end]
                    
                    offset_start = offset_end

        return out_data

    return wrapper
