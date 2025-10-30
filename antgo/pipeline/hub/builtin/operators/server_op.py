# -*- coding: UTF-8 -*-
# @Time    : 2025/10/26 22:42
# @File    : server_op.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.pipeline.engine import *
from antgo.pipeline.models.utils.utils import *
from antgo.interactcontext import InteractContext
from antgo.utils import *
import threading
import queue
import traceback


@register
class server_op(object):
    def __init__(self, init_func, proc_func, destroy_func=None, max_size=5, server_num=1):
        # 服务轮询相关参数
        self.server_current_index = 0
        self.server_list = list(range(server_num))
        self.cache_out_data = {}
        self.server_robin_lock = threading.Lock()
        self.cache_out_lock = threading.Lock()

        self.max_size = max_size
        self.init_func = init_func
        self.proc_func = proc_func
        self.destroy_func = destroy_func
        self.sess_config = []
        for index, d_id in enumerate(self.server_list):
            self.sess_config.append({
                'in_queue': queue.Queue(maxsize=self.max_size),
                'out_queue': queue.Queue(maxsize=self.max_size)
            })

        # 初始化推理引擎服务
        for index in range(len(self.sess_config)):
            thread = threading.Thread(
                target=self.infer,
                args=(index,),
                daemon=True
            )
            thread.start()

    def _inner_infer(self, *args, sess=None):
        return self.proc_func(*args, sess=sess)

    def infer(self, index):
        # 初始化
        sess = self.init_func()

        while True:
            try:
                # 从队列中获取数据
                in_data, thread_id = self.sess_config[index]['in_queue'].get(block=True)

                # 在设备 index 执行推理
                out_data = self._inner_infer(*in_data, sess=sess)

                # 推送到输出队列
                self.sess_config[index]['out_queue'].put((out_data, thread_id))
            except Exception as e:
                traceback.print_exc()

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