# -*- coding: UTF-8 -*-
# @Time    : 2022/9/5 23:38
# @File    : __init__.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.pipeline.hub import *
from antgo.pipeline.functional.data_collection import *
from antgo.pipeline.functional import *
from antgo.pipeline.extent import op
from antgo.pipeline.extent.glue.common import *
from antgo.pipeline.engine import *
import numpy as np
import os
import json


def package(project, folder='./deploy', **kwargs):
    # project_folder = os.path.join(folder, f'{project}_plugin')
    # if not os.path.exists(project_folder):
    #     print(f'Project {project} not exist.')
    #     return

    # os.system(f'cd {project_folder} && bash package.sh')
    pass


def service(project, folder='./deploy', **kwargs):
    pass