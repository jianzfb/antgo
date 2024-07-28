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
from jinja2 import Environment, FileSystemLoader
import numpy as np
import shutil
import pathlib
import os
import json
import logging
import time


ANTGO_DEPEND_ROOT = os.environ.get('ANTGO_DEPEND_ROOT', f'{str(pathlib.Path.home())}/.3rd')
def pipeline_cplusplus_package(project, folder='./deploy', **kwargs):
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

    # depedent_folder = os.path.join(package_folder, 'dependents')
    # os.makedirs(depedent_folder, exist_ok=True)

    os.makedirs(os.path.join(package_folder, 'config'), exist_ok=True)
    os.makedirs(os.path.join(package_folder, 'models'), exist_ok=True)

    project_compile_folder = ''
    project_platform = project_info['platform']
    if project_info['platform'] == 'linux':
        # 以镜像形式打包
        project_compile_folder = os.path.join(project_folder, 'bin', project_info['abi'])
    elif project_info['platform'] == 'android':
        # 以sdk形式打包
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

    # 复制依赖文件
    print('install 3rd .so')
    os.system(f'cp -r {ANTGO_DEPEND_ROOT}/eagleeye/{project_info["platform"]}-{project_info["abi"]}-install/libs/{project_info["abi"]}/* {package_folder}')
    os.system(f'cp -r ./3rd/{project_info["abi"]}/* bin/{project_info["abi"]}/')

    # 复制配置文件
    if os.path.exists(os.path.join(project_folder, 'config')):
        print('install config files')
        try:
            os.system(f'cp -r {os.path.join(project_folder, "config")}/* {os.path.join(package_folder, "config")}/')
        except:
            pass

    # 复制模型文件
    if os.path.exists(os.path.join(project_folder, 'models')):
        print('install model files')
        try:
            os.system(f'cp -r {os.path.join(project_folder, "models")}/* {os.path.join(package_folder, "models")}/')
        except:
            pass

    # 检查是否需要ffmpeg依赖
    if ('ffmpeg' in project_info['eagleeye']) and (project_platform != 'linux'):
        # 仅对android/windows处理（linux平台下，镜像中默认已经安装）
        os.system(f'cp -r {project_info["eagleeye"]["ffmpeg"]}/install/lib/*.so {depedent_folder}')

    # 检查是否需要rk依赖
    if ('rk' in project_info['eagleeye']) and (project_platform != 'linux'):
        os.system(f'cp -r {project_info["eagleeye"]["rk"]}/librga/libs/AndroidNdk/arm64-v8a/librga.so {depedent_folder}')
        os.system(f'cp -r {project_info["eagleeye"]["rk"]}/mpp/build/android/mpp/librockchip_mpp.so {depedent_folder}')

    # 检查是否需要minio依赖

    # 检查是否需要grpc依赖

    # 检查是否需要opencv依赖

    if project_platform == 'linux':
        # 对于linux平台，创建启动脚本
        with open(f'{package_folder}/server-launch.sh', 'w') as fp:
            fp.write(f'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./;./leshipro_demo $1') 


def pipeline_build_image(project, folder='./deploy', **kwargs):
    eagleeye_version = kwargs.get('version', 'master')
    dockerfile_data = {
        'version': eagleeye_version
    }

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

    # 解析项目平台信息
    project_platform = project_info['platform']
    project_abi = project_info['abi']

    if project_platform != 'linux':
        print('Only support linux')
        return

    # 解析第三方依赖信息（ffmpeg, rk, minio）
    eagleeye_config = project_info['eagleeye']
    compile_suffix = project_abi.replace('-', '_')+'_build_with'
    for env_key in eagleeye_config.keys():
        compile_suffix += f'_{env_key}'
    dockerfile_data.update(
        {
            'eagleeye_compile_suffix': compile_suffix,
            "project_compile_suffix": project_abi.replace('-', '_'),
            'abi': project_abi,
            'project': project,
            'server_port': kwargs.get('port', 9002)
        }
    )

    env = Environment(loader=FileSystemLoader('/'.join(os.path.realpath(__file__).split('/')[0:-2])))
    dockerfile_template = env.get_template('script/DockerfileX86-64')
    if project_abi == 'arm64-v8a':
        dockerfile_template = env.get_template('script/DockerfileArm64-V8a')

    dockerfile_content = dockerfile_template.render(**dockerfile_data)

    with open(f'Dockerfile', 'w') as fp:
        fp.write(dockerfile_content)

    # 构建镜像
    # os.system(f'docker build -t {project} .')

    # 发布镜像
    image_pro = kwargs.get('image_repo', None)
    image_version = kwargs.get('image_version', None)
    user = kwargs.get('user', None)
    password = kwargs.get('password', None)
    server_port = kwargs.get('port', 9002)
    if image_pro is None or user is None or password is None:
        # logging.warn("No set image repo and user name, If need to deploy, must set --image-repo=xxxx --user=xxx --password=xxx.")
        logging.warn("No image_repo, only use local image file")
        image_time = time.strftime(f"%Y-%m-%d.%H-%M-%S", time.localtime(time.time()))
        server_config_info = {
            'image_repo': '',
            'create_time': image_time,
            'update_time': image_time,
            'server_port': server_port,
            'name': project,
            'mode': 'grpc'
        }

        if os.path.exists('./server_config.json'):
            with open('./server_config.json', 'r') as fp:
                info = json.load(fp)
                server_config_info['create_time'] = info['create_time']

        # 更新服务配置
        with open('./server_config.json', 'w') as fp:
            json.dump(server_config_info, fp)
        return

    logging.info(f'Push image {project} to image repo {image_pro}:{image_version}')
    # 需要手动添加密码
    os.system(f'docker login --username={user} --password={password} {image_repo.split("/")[0]}')
    os.system(f'docker tag {project}:latest {image_repo}:{image_version}')
    os.system(f'docker push {image_repo}:{image_version}')

    image_time = time.strftime(f"%Y-%m-%d.%H-%M-%S", time.localtime(time.time()))
    server_config_info = {
        'image_repo': f'{image_repo}:{image_version}',
        'create_time': image_time,
        'update_time': image_time,
        'server_port': server_port,
        'name': project,
        'mode': 'grpc'
    }

    if os.path.exists('./server_config.json'):
        with open('./server_config.json', 'r') as fp:
            info = json.load(fp)
            server_config_info['create_time'] = info['create_time']

    # 更新服务配置
    with open('./server_config.json', 'w') as fp:
        json.dump(server_config_info, fp)
