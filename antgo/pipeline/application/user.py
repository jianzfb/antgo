# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : user.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antvis.client.httprpc import *
from antgo import config
import os
import cv2
import base64
import numpy as np


# 单点认证，认证通过执行，不通过暂停管线执行退出
class CasAuth(object):
    def __init__(self):
        pass

    def __call__(self, *args):
        # input
        # ticket, token
        # 

        # output
        # user
        pass