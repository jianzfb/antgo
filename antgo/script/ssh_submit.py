import os
import yaml 
from antgo import config
import json

# 提交任务运行
def ssh_submit_process_func(project_name, sys_argv, gpu_num, cpu_num, memory_size):
    # step1: 加载ssh配置文件
    ssh_submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', 'ssh-submit-config.yaml')
    assert(os.path.exists(ssh_submit_config_file))
    
    with open(ssh_submit_config_file, encoding='utf-8', mode='r') as fp:
        config_content = yaml.safe_load(fp)

    username = config_content['config']['username']
    password = config_content['config']['password']
    ip = config_content['config']['ip']
    submit_script = os.path.join(os.path.dirname(__file__), 'ssh-submit.sh')
    
    with open(os.path.join(config.AntConfig.task_factory,f'{project_name}.json'), 'r') as fp:
        project_info = json.load(fp)

    image_name = '' # 基础镜像
    if project_info['image'] != '':
        image_name = project_info['image']
    if image_name == '':
        image_name = 'antgo-env:latest'
    
    if password == '':
        password = 'default'
    submit_cmd = f'bash {submit_script} {username} {password} {ip} {gpu_num} {cpu_num} {memory_size}M "{sys_argv}" {image_name} {project_name}'
    os.system(submit_cmd)
    
    
# 检查任务资源是否满足
def ssh_submit_resource_check_func(gpu_num, cpu_num, memory_size):
    return True