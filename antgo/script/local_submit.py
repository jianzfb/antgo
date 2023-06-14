import logging
import os
import shutil
import yaml 
import json
import subprocess
from antgo import config
from antgo.script.base import *


# 提交任务运行 (仅作为debug时使用，在本地机器上运行任务)
def local_submit_process_func(project_name, sys_argv, gpu_num, cpu_num, memory_size, task_name=None):   
    # 前提假设，调用此函数前当前目录下需要存在项目代码
    if project_name != '':
        with open(os.path.join(config.AntConfig.task_factory,f'{project_name}.json'), 'r') as fp:
            project_info = json.load(fp)

        if task_name is not None:
            extra_config = prepare_extra_config(task_name, project_info)
            if extra_config is None:
                return False

            with open('./extra-config.json', 'w') as fp:
                json.dump(extra_config, fp)      
            sys_argv += " --extra-config=./extra-config.json"

    # 后台运行
    process = subprocess.Popen(sys_argv, shell=True)
    with open(os.path.join(config.AntConfig.factory, '.local_submit_process.pid'), 'w') as fp:
        fp.write(f'{process.pid}')

    # 删除临时配置：
    if os.path.exists('./extra-config.py'):
        os.remove('./extra-config.py')
    return True


# 检查任务资源是否满足
def local_submit_resource_check_func(gpu_num, cpu_num, memory_size):
    # # 单位时间内仅可以一个任务在执行
    # if not os.path.exists(os.path.join(config.AntConfig.factory, '.local_submit_process.pid')):
    #     return True
    
    # with open(os.path.join(config.AntConfig.factory, '.local_submit_process.pid'), 'r') as fp:
    #     process_pid = fp.read().strip()
        
    # # 检查process_pid是否在运行
    # process_dir = os.path.join('/proc', str(process_pid))
    # if os.path.exists(process_dir):
    #     return False
    
    return True