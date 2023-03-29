import logging
import os
import yaml 
from antgo import config
import json

# 提交任务运行
def ssh_submit_process_func(project_name, sys_argv, gpu_num, cpu_num, memory_size, task_name=None):
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

    # 添加临时配置：将当前工程信息保存到当前目录下并一同提交
    if task_name is None:
        # 复合任务标记
        extra_config = {}
        if task_name == "activelearning":
            pass
        elif task_name == "supervised":
            # 扩展数据源
            # label, pseudo-label, unlabel
            extra_config['source'] = {
                "label": project_info["dataset"]["train"]["label"],
                "pseudo-label": project_info["dataset"]["train"]["pseudo-label"],
                "unlabel": project_info["dataset"]["train"]["unlabel"]
            }
        elif task_name == "semi-supervised":
            if project_info['tool']['semi']['method'] == '' or len(project_info['tool']['semi']['config']) == 0:
                logging.error(f"Missing {task_name} config, couldnt launch task")
                return False
            
            # 扩展数据源
            # label, pseudo-label, unlabel
            extra_config['source'] = {
                "label": project_info["dataset"]["train"]["label"],
                "pseudo-label": project_info["dataset"]["train"]["pseudo-label"] if "pseudo-label" in project_info["dataset"]["train"] else [],
                "unlabel": project_info["dataset"]["train"]["unlabel"] if "unlabel" in project_info["dataset"]["train"] else []
            }
            
            # 扩展模型配置/优化器/学习率等
            extra_config.update( project_info['tool']['semi']['config'])
        elif task_name == "distillation":
            if project_info['tool']['semi']['method'] == '' or len(project_info['tool']['semi']['config']) == 0:
                logging.error(f"Missing {task_name} config, couldnt launch task")
                return False
                    
            # 扩展数据源
            # label, pseudo-label, unlabel
            extra_config['source'] = {
                "label": project_info["dataset"]["train"]["label"],
                "pseudo-label": project_info["dataset"]["train"]["pseudo-label"],
                "unlabel": project_info["dataset"]["train"]["unlabel"]
            }
                    
            # 扩展模型配置/优化器/学习率等
            extra_config.update( project_info['tool']['distillation']['config'])

        with open('./extra-config.py', 'w') as fp:
            json.dump(extra_config, fp)    
        sys_argv += " --extra-config=./extra-config.py"

    # 执行提交命令
    if password == '':
        password = 'default'
    submit_cmd = f'bash {submit_script} {username} {password} {ip} {gpu_num} {cpu_num} {memory_size}M "{sys_argv}" {image_name} {project_name}'
    os.system(submit_cmd)

    # 删除临时配置：
    if os.path.exists('./extra-config.py'):
        os.remove('./extra-config.py')

    return True

# 检查任务资源是否满足
def ssh_submit_resource_check_func(gpu_num, cpu_num, memory_size):
    # TODO，支持资源检查
    return True