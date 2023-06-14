import logging
import os
import time
from queue import PriorityQueue
import shutil
import yaml 
import json
from antgo import config
from antgo.script.base import *


# 提交任务运行
def ssh_submit_process_func(project_name, sys_argv, gpu_num, cpu_num, memory_size, task_name=None):   
    # 前提假设，调用此函数前当前目录下需要存在项目代码
    # step1: 加载ssh配置文件
    ssh_submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', 'ssh-submit-config.yaml')
    if not os.path.exists(ssh_submit_config_file):
        logging.error('No ssh submit config.')
        return False

    with open(ssh_submit_config_file, encoding='utf-8', mode='r') as fp:
        config_content = yaml.safe_load(fp)

    username = config_content['config']['username']
    password = config_content['config']['password']
    ip = config_content['config']['ip']
    submit_script = os.path.join(os.path.dirname(__file__), 'ssh-submit.sh')

    project_info = {}
    if project_name != '':
        with open(os.path.join(config.AntConfig.task_factory,f'{project_name}.json'), 'r') as fp:
            project_info = json.load(fp)

    image_name = 'antgo-env:latest' # 基础镜像
    if 'image' in project_info and project_info['image'] != '':
        image_name = project_info['image']

    # 添加扩展配置：保存到当前目录下并一同提交
    if task_name is not None and len(project_info) > 0:
        extra_config = prepare_extra_config(task_name, project_info)
        if extra_config is None:
            return False

        with open('./extra-config.py', 'w') as fp:
            json.dump(extra_config, fp)    
        sys_argv += " --extra-config=./extra-config.py"

    # 执行提交命令
    if password == '':
        password = 'default'

    remote_local_folder_name = time.strftime(f"%Y-%m-%d.%H-%M-%S", time.localtime(time.time()))
    submit_cmd = f'bash {submit_script} {username} {password} {ip} {gpu_num} {cpu_num} {memory_size}M "{sys_argv}" {image_name} {remote_local_folder_name}'
    os.system(submit_cmd)

    # 删除临时配置：
    if os.path.exists('./extra-config.py'):
        os.remove('./extra-config.py')
    return True

# 检查任务资源是否满足
def ssh_submit_resource_check_func(gpu_num, cpu_num, memory_size):
    # TODO，支持资源检查
    return True