import os
import sys
import copy
import numpy as np

# usage
# control.DetectOrTracking.xxx[]()

class DetectOrTracking(object):
    def __init__(self, det_func, tracking_func=None, interval=1, only_once=False):
        self.det_func = det_func
        self.tracking_func = tracking_func
        self.count = 0
        self.interval = interval
        self.only_once = only_once
        
        self.det_func_arg_num = -1
        self.tracking_func_update_arg_num = -1

        self.det_func_res_num = 0
        self.det_func_res_type = None

    def __call__(self, *args):
        if self.det_func_arg_num < 0:
            self.det_func_arg_num = len([_ for _ in args if _ is not None])
            self.tracking_func_update_arg_num = len(args) - self.det_func_arg_num

        # 运行func条件
        # 1: 首次调用
        # 2: 间隔interval调用
        # 3: args[...] 无效数据
        is_call_det = False
        if self.count == 0:
            is_call_det = True
        elif self.interval > 0 and self.count % self.interval == 0:
            is_call_det = True
        elif self.tracking_func_update_arg_num > 0 and args[self.det_func_arg_num].shape[0] == 0:
            is_call_det = True
        elif self.interval == 0:
            is_call_det = True

        if self.only_once:
            is_call_det = False

        # 计数+1
        self.count += 1
    
        if is_call_det:
            # 使用det_func计算（一般情况下，det_func计算代价高）
            det_args = args[:self.det_func_arg_num]
            det_res = self.det_func(*det_args)
            self.det_func_res_num = 1
            if isinstance(det_res, list) or isinstance(det_res, tuple):
                self.det_func_res_num = len(det_res)
            if self.det_func_res_type is None:
                if isinstance(det_res, list) or isinstance(det_res, tuple):
                    self.det_func_res_type = [d.dtype for d in det_res]
                else:
                    self.det_func_res_type = [det_res.dtype]

            # 更新tracking_func状态
            tracking_args = args[:self.det_func_arg_num]
            if isinstance(det_res, list) or isinstance(det_res, tuple):
                tracking_args += det_res
            else:
                tracking_args += (det_res,)
            if len(args) > self.det_func_arg_num:
                for i in range(self.det_func_res_num):
                    tracking_args += (np.empty([0], dtype=self.det_func_res_type[i]),)

            self.tracking_func(*tracking_args)
            return det_res

        if self.tracking_func is not None:
            # 使用tracking_func计算（一般情况下，tracking_func计算代价小）
            # tracking_args: det_func_args + det_func_res(lastest) + update_res
            tracking_args = args[:self.det_func_arg_num]
            for i in range(self.det_func_res_num):
                tracking_args += (np.empty([0], dtype=self.det_func_res_type[i]),)
            if len(args) > self.det_func_arg_num:
                tracking_args += args[self.det_func_arg_num:]

            # 使用tracking_func计算
            tracking_res = self.tracking_func(*tracking_args)
            return tracking_res

        # update_res
        return args[self.det_func_arg_num:]