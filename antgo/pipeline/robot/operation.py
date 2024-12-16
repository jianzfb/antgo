# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : operation.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.pipeline.engine import *


class StepOp(object):
    def __init__(self, **kwargs):
        self.max_steps = kwargs.get('max_steps', 100)
        self.cur_step = 0

    def __call__(self, *args):
        env_info, action = args
        env = env_info['env']
        name = env_info['name']
        if name == 'maniskill':
            obs, reward, terminated, truncated, info = env.step(action)

        self.cur_step += 1
        done = terminated or truncated
        if self.cur_step > self.max_steps:
            done = True
        if done:
            env_info['status'] = 'done'
            self.cur_step = 0
        
        return env_info, obs, reward, info
