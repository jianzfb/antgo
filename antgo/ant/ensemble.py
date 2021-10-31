# -*- coding: UTF-8 -*-
# @Time    : 2019/1/22 1:16 PM
# @File    : ensemble.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.resource.html import *
from antgo.ant.base import *
from antgo.dataflow.common import *
from antgo.dataflow.recorder import *
from antvis.client.httprpc import *
from multiprocessing import Process, Queue
from antgo.task.task import *
import traceback
import subprocess
import os
import socket
import requests
import json
import zipfile


class AntEnsemble(AntBase):
    def __init__(self, ant_name, ant_context, ant_token, **kwargs):
        super().__init__(ant_name, ant_context=ant_context, ant_token=ant_token, **kwargs)
        # master, slave
        # bagging, stacking

    def start(self):
        pass
