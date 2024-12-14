# -*- coding: UTF-8 -*-
# @Time    : 2022/9/11 23:01
# @File    : env_collection.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from .data_collection import DataCollection, DataFrame
from .entity import Entity
from .image import *
from .common import *
from antgo.pipeline.hparam import HyperParameter as State

from antgo.pipeline.hparam import param_scope
from antgo.pipeline.hparam import dynamic_dispatch
from antgo.pipeline.functional.common.config import *
import numpy as np
import json
import os
import cv2
try:
    import gymnasium as gym
    import mani_skill.envs
except:
    gym = None
    print('no maniskill package import')



@dynamic_dispatch
def maniskill_env(*args, **kwargs):
    index = param_scope()._index

    # 场景定义
    scene_name = kwargs.get('scene_name', None)
    seed = kwargs.get('seed', 0)
    assert(scene_name is not None)

    if not isinstance(scene_name, list):
        scene_name = [scene_name]

    def inner():
        for name in scene_name:
            env = gym.make(
                name, 
                render_mode='rgb_array',
                obs_mode='rgbd',
                control_mode="pd_joint_pos",
                reward_mode="dense",
                sensor_configs=dict(shader_pack='default'),
            )
            obs, _ = env.reset(seed=seed)
            env_entity = Entity()(**{index[0]: {'env': env, 'name': 'maniskill', 'status': 'running'}, index[1]: obs})
            while env_entity.env['status'] == 'running':
                yield env_entity
            env_entity.env['env'].close()

    return DataFrame(inner())


class _env(object):
    def __getattr__(self, name):
        if name not in ['maniskill']:
            return None

        return globals()[f'{name}_env']


env = _env()