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
from antgo.pipeline.eagleeye import *
import numpy as np
import shutil

import os
import json


def package(project, folder='./deploy', **kwargs):
    # 打包编译好的管线
    project_folder = os.path.join(folder, f'{project}_plugin')
    if not os.path.exists(project_folder):
        print(f'Project {project} not exist.')
        return

    project_file = os.path.join(project_folder, '.project.json')
    if not os.path.exists(project_file):
        print(f'Project {project} file not exist.')
        return

    with open(project_file, 'r') as fp:
        project_info = json.load(fp)

    package_folder = os.path.join(folder, 'package')
    os.makedirs(package_folder, exist_ok=True)
    plugin_folder = os.path.join(package_folder, 'plugins')
    os.makedirs(plugin_folder, exist_ok=True)
    project_plugin_folder = os.path.join(plugin_folder, project)
    os.makedirs(project_plugin_folder, exist_ok=True)

    depedent_folder = os.path.join(package_folder, 'dependents')
    os.makedirs(depedent_folder, exist_ok=True)

    os.makedirs(os.path.join(package_folder, 'config'), exist_ok=True)
    os.makedirs(os.path.join(package_folder, 'model'), exist_ok=True)

    project_compile_folder = ''
    project_platform = project_info['platform']
    if project_info['platform'] == 'linux':
        project_compile_folder = os.path.join(project_folder, 'bin', 'X86-64')
    elif project_info['platform'] == 'android':
        project_compile_folder = os.path.join(project_folder, 'bin', 'arm64-v8a')

    if project_compile_folder == '':
        print('Dont found project compile folder')
        return

    if not os.path.exists(project_compile_folder):
        print(f'Dont found project compile folder: {project_compile_folder}')
        return

    # 复制库文件
    print('install .so files')
    os.system(f'cp -r {project_compile_folder}/*.so {project_plugin_folder}/')

    # 复制主程序文件
    print('install main file')
    os.system(f'cp {project_compile_folder}/{project}_demo {package_folder}/')

    # 复制配置文件
    if os.path.exists(os.path.join(project_folder, 'config')):
        print('install config files')
        os.system(f'cp -r {os.path.join(project_folder, "config")}/* {os.path.join(package_folder, "config")}/')

    # 复制模型文件
    if os.path.exists(os.path.join(project_folder, 'model')):
        print('install model files')
        os.system(f'cp -r {os.path.join(project_folder, "model")}/* {os.path.join(package_folder, "model")}/')

    # 检查是否需要ffmpeg依赖
    if 'ffmpeg' in project_info['eagleeye']:
        os.system(f'cp -r {project_info["eagleeye"]["ffmpeg"]}/install/lib/*.so {depedent_folder}')

    # 检查是否需要rk依赖
    if 'rk' in project_info['eagleeye']:
        if project_platform == 'linux':
            os.system(f'cp -r {project_info["eagleeye"]["rk"]}/librga/libs/Linux/gcc-aarch64/librga.so {depedent_folder}')
            os.system(f'cp -r {project_info["eagleeye"]["rk"]}/mpp/build/linux/x86_64/librockchip_mpp.so {depedent_folder}')
        else:
            os.system(f'cp -r {project_info["eagleeye"]["rk"]}/librga/libs/AndroidNdk/arm64-v8a/librga.so {depedent_folder}')
            os.system(f'cp -r {project_info["eagleeye"]["rk"]}/mpp/build/android/mpp/librockchip_mpp.so {depedent_folder}')

    # 检查是否需要minio依赖

    # 检查是否需要grpc依赖

    # 检查是否需要opencv依赖


def release(project, folder='./deploy', **kwargs):
    # 发布打包好的管线
    pass