import logging
import os
from queue import PriorityQueue
import shutil
import yaml 
import json
import logging
from antgo import config
from antgo.script.base import *


# 提交任务运行
def custom_submit_process_func(project_name, sys_argv, gpu_num, cpu_num, memory_size, task_name=None):   
    # 前提假设，调用此函数前当前目录下需要存在项目代码
    # step1: 加载配置文件
    submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', 'submit-config.yaml')    
    if not os.path.exists(submit_config_file):
        logging.error('No custom submit script config')
        return
    
    with open(submit_config_file, encoding='utf-8', mode='r') as fp:
        config_content = yaml.safe_load(fp)

    script_folder = config_content['folder']
    script_file = config_content['script']
    if not os.path.exists(os.path.join(script_folder, script_file)):
        logging.error('Custom submit scrip launch file not exist.')
        return

    project_info = {}
    if project_name != '':
        with open(os.path.join(config.AntConfig.task_factory,f'{project_name}.json'), 'r') as fp:
            project_info = json.load(fp)

    image_name = 'antgo-env:latest' # 基础镜像
    if project_info['image'] != '':
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
    script_file = os.path.join(script_folder, script_file)
    submit_cmd = f'bash {script_file} {image_name} {sys_argv} {gpu_num} {cpu_num} {memory_size}M'
    os.system(submit_cmd)

    # 删除临时配置：
    if os.path.exists('./extra-config.py'):
        os.remove('./extra-config.py')
    return True

# 检查任务资源是否满足
def custom_submit_resource_check_func(gpu_num, cpu_num, memory_size):
    # TODO，支持资源检查
    return True