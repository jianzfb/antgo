import sys
import logging
import os
import time
from queue import PriorityQueue
import shutil
import yaml 
import json
from antgo import config
from antgo.framework.helper.utils import Config
from antgo.script.base import *
import subprocess
import re


# 提交任务运行
def remote_gpu_running_info(username, ip):
    ret = subprocess.Popen(f'ssh {username}@{ip} nvidia-smi', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    content = ret.stdout.read()
    content = content.decode('utf-8')

    driver_version = re.findall('(?<=Driver Version: )[\d.]+', content)[0]
    gpu_basic_info = re.findall('(?<=\|)[ ]+\d+%[ ]+\d+C[]+\w+\d+W / /d+W[ ]+(?=\|)', content)
    is_plan_b = False
    if len(gpu_basic_info) == 0:
        gpu_basic_info = re.findall('(?<=\|)[ ]+N/A[ ]+\d+C[]+\w+\d+W / /d+W[ ]+(?=\|)', content)
        is_plan_b = True
    
    gpu_num = len(gpu_basic_info)
    gpus=[]

    for gpu_index in range(gpu_num):
        result = re.split('\s+', gpu_basic_info[gpu_index].strip())
        gpus.append(result[2])

    gpu_pwr_info = re.findall('\d+W +/ +\d+W',content)
    gpu_pwr_usage=[]
    gpu_pwr_cap=[]
    for gpu_index in range(gpu_num):
        result = re.split('/',gpu_pwr_info[gpu_index])
        pwr_usage = re.findall('\d+',result[0])[0]
        pwr_cap = re.findall('\d+',result[1])[0]
        gpu_pwr_usage.append(int(pwr_usage))
        gpu_pwr_cap.append(int(pwr_cap))

    gpu_mem_info = re.findall('\d+MiB / +\d+MiB',content)
    gpu_mem_usage=[]
    gpu_mem_max=[]
    for gpu_index in range(gpu_num):
        result = re.split('/',gpu_mem_info[gpu_index])
        mem_usage = re.findall('\d+',result[0])[0]
        mem_max = re.findall('\d+',result[1])[0]
        gpu_mem_usage.append(int(mem_usage))
        gpu_mem_max.append(int(mem_max))

    gpu_util = re.findall('\d+(?=%)',content)
    gpu_util = [int(util) for util in gpu_util]
    if not is_plan_b:
        gpu_util = [int(util) for id, util in enumerate(gpu_util) if id % 2 == 1]

    occupy_gpus = list(range(len(gpus)))

    free_gpus = []
    count = 0
    for line in content.split('\n'):
        if line.startswith("|======="):
            count += 1
            continue
        
        if line == '':
            continue
        if 'No running processes found' in line:
            continue

        if count == 2 and not line.startswith('+-------'):
            line = re.sub(r'\s+', ' ', line)            
            free_gpus.append(int(line.split(' ')[1]))
    
    free_gpus = [gpu_i for gpu_i in range(gpu_num) if gpu_i not in free_gpus]

    return {'gpus': gpus,
            'driver-version': driver_version,
            'gpu_pwr_usage': gpu_pwr_usage,
            'gpu_pwr_cap': gpu_pwr_cap,
            'gpu_mem_usage': gpu_mem_usage,
            'gpu_mem_max': gpu_mem_max,
            'gpu_util': gpu_util,
            'occupy_gpus': occupy_gpus,
            'free_gpus': free_gpus}


def check_data_is_exist(username, ip, data_list):
    ret = subprocess.Popen(f'ssh {username}@{ip} "ls -lh /data/"', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    content = ret.stdout.read()
    content = content.decode('utf-8')
    remote_existed_data_list = content.split('\n')

    is_all_exist = True
    for check_data_name in data_list:
        if check_data_name not in remote_existed_data_list:
            is_all_exist = False
            break
        pass
    
    return is_all_exist


def _iterator_analyze_data_list(data):
    if isinstance(data, list) or isinstance(data, tuple):
        all_data_list = []
        for sub_data in data:
            all_data_list.extend(_iterator_analyze_data_list(sub_data))

        return all_data_list
    
    all_data_list = []
    if 'dir' in data:
        data_name = data.dir.split('/')[-1]
        all_data_list.append(data_name)
    elif 'data_folder' in data:
        data_name = data.data_folder.split('/')[-1]
        all_data_list.append(data_name)
    return all_data_list


def analyze_all_dependent_data(config_file_path):
    cfg = Config.fromfile(config_file_path)
    
    dependent_data_list = []
    # train, val, test
    if 'train' in cfg.data:
        dependent_data_list.extend(_iterator_analyze_data_list(cfg.data.train))
    if 'val' in cfg.data:
        dependent_data_list.extend(_iterator_analyze_data_list(cfg.data.val))
    if 'test' in cfg.data:
        dependent_data_list.extend(_iterator_analyze_data_list(cfg.data.test))
    
    return dependent_data_list


def ssh_submit_process_func(create_time, sys_argv, gpu_ids, cpu_num, memory_size, task_name=None, ip='', exp='', check_data=False, env='master', data_folder='/data', project_folder='', is_resource_check=False):   
    # 前提假设，调用此函数前当前目录下需要存在项目代码
    # 遍历所有注册的设备，找到每个设备的空闲GPU
    with open('./.project.json', 'r') as fp:
        project_info = json.load(fp)

    gpu_num = len(gpu_ids)
    dependent_data_list= []
    if check_data:
        exp_info = project_info['exp'][exp][-1]
        exp_config_file = exp_info['config']
        exp_config_path = exp_config_file
        if not os.path.exists(exp_config_path):
            exp_config_path = os.path.join(exp, 'configs', exp_config_file)
        
        dependent_data_list = analyze_all_dependent_data(exp_config_path)

    username = ''
    password = ''
    ssh_config_info = None
    logging.info("Analyze cluster environment.")
    if ip == '':
        # 自动搜索可用远程机器
        for file_name in os.listdir(os.path.join(os.environ['HOME'], '.config', 'antgo')):
            register_ip = ''
            if file_name.endswith('.yaml') and file_name.startswith('ssh'):
                terms = file_name.split('-')
                if len(terms) == 4:
                    register_ip = terms[1]
            else:
                continue

            if register_ip == '':
                continue

            ssh_submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', file_name)
            with open(ssh_submit_config_file, encoding='utf-8', mode='r') as fp:
                ssh_config_info = yaml.safe_load(fp)

            # 检查GPU占用情况
            logging.info(f'Analyze IP: {register_ip}')
            info = remote_gpu_running_info(ssh_config_info["config"]["username"], ssh_config_info["config"]["ip"])
            if len(info['free_gpus']) >= gpu_num:
                if check_data:
                    has_local_data = check_data_is_exist(ssh_config_info["config"]["username"], ssh_config_info["config"]["ip"], dependent_data_list)
                    if not has_local_data:
                        continue

                ip = ssh_config_info["config"]["ip"]
                username = ssh_config_info["config"]["username"]
                password = ssh_config_info["config"]["password"]
                free_gpus = info['free_gpus']
                break

    target_ip_list = ip.split(',')
    target_machine_info_list = []
    if is_resource_check:
        for target_ip in target_ip_list:
            ssh_submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', f'ssh-{target_ip}-submit-config.yaml')
            with open(ssh_submit_config_file, encoding='utf-8', mode='r') as fp:
                ssh_config_info = yaml.safe_load(fp)

            # 检查GPU占用情况
            info = remote_gpu_running_info(ssh_config_info["config"]["username"], ssh_config_info["config"]["ip"])
            if len(info['free_gpus']) < gpu_num:
                logging.error(f"No enough gpu in {target_ip}.")
                return

            has_local_data = True
            if check_data:
                has_local_data = check_data_is_exist(ssh_config_info["config"]["username"], ssh_config_info["config"]["ip"], dependent_data_list)
                if not has_local_data:
                    logging.error(f'Dont exist data in remote {target_ip}')

            if has_local_data:
                username = ssh_config_info["config"]["username"]
                password = ssh_config_info["config"]["password"]
                target_machine_info_list.append({
                    'ip': target_ip,
                    'username': username,
                    'password': password,
                    'gpus': info['free_gpus']
                })

        if len(target_machine_info_list) != len(target_ip_list):
            logging.error("No enough machine resource")
            return
    else:
        for target_ip in target_ip_list:
            ssh_submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', f'ssh-{target_ip}-submit-config.yaml')
            with open(ssh_submit_config_file, encoding='utf-8', mode='r') as fp:
                ssh_config_info = yaml.safe_load(fp)

            username = ssh_config_info["config"]["username"]
            password = ssh_config_info["config"]["password"]
            target_machine_info_list.append({
                'ip': target_ip,
                'username': username,
                'password': password,
                'gpus': [int(g) for g in gpu_ids]
            })

    logging.info(f"Apply target machine resource {target_machine_info_list}")

    # 对于多机多卡模式，不可自定义指定GPU编号
    # 对于单机多卡模式，可以自定义指定GPU编号
    apply_gpu_id = [str(i) for i in range(gpu_num)]
    if len(target_machine_info_list) == 1:
        candidate_gpu_ids = [int(gpu_id) for gpu_id in target_machine_info_list[0]['gpus']]
        apply_gpu_id = []
        for gd in gpu_ids:
            if gd in candidate_gpu_ids:
                apply_gpu_id.append(str(gd))
        
        if len(apply_gpu_id) != gpu_num:
            logging.error(f"gpus {gpu_ids} not all free")
            return

    apply_gpu_id = ','.join(apply_gpu_id)
    sys_argv = f'{sys_argv} --gpu-id={apply_gpu_id}'

    # 添加多机多卡配置参数
    sys_argv = f'{sys_argv} --nodes={len(target_machine_info_list)}'
    if '--master-addr' not in sys_argv:
        if len(target_machine_info_list) == 1:
            sys_argv = f'{sys_argv} --master-addr=127.0.0.1'
        else:
            # [0]作为master节点
            sys_argv = f'{sys_argv} --master-addr={target_machine_info_list[0]["ip"]}'

    # 添加扩展配置:保存到当前目录下并一同提交
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

    launch_script = ssh_config_info['script'] if 'script' in ssh_config_info else ''
    if launch_script != '':
        sys_argv = launch_script

    image_name = 'registry.cn-hangzhou.aliyuncs.com/vibstring/antgo-env:latest' # 基础镜像
    launch_image = ssh_config_info['image'] if 'image' in ssh_config_info else ''
    if launch_image != '':
        image_name = launch_image
    if 'image' in project_info and project_info['image'] != '':
        image_name = project_info['image']

    print(f'Use image {image_name}')
    project_name = os.path.abspath(os.path.curdir).split("/")[-1]
    submit_time = create_time

    target_machine_ips = ','.join([v['ip'] for v in target_machine_info_list])
    submit_script = os.path.join(os.path.dirname(__file__), 'ssh-submit.sh')
    submit_cmd = f'bash {submit_script} {username} {password} {target_machine_ips} {gpu_num} {cpu_num} {memory_size}M "{sys_argv}" {image_name} {project_name} {env} {submit_time} {data_folder} {project_folder}'

    # 解析提交后的输出，并解析出container id
    print('submit command')
    print(submit_cmd)

    # 记录提交机器地址
    with open('./address', 'w') as fp:
        fp.write(f'{username}@{target_machine_info_list[0]["ip"]}')

    print('\n\n')
    print('remote execute process')
    ret = subprocess.Popen(submit_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    content = ret.stdout.read()
    content = content.decode('utf-8')

    print('\n\n')
    print('remote execute response')
    print(content)

    # 检查返回的容器ID和机器IP对应关系
    master_machine_info = target_machine_info_list[0]
    container_id_list = content.split('\n')[(-1-len(target_machine_info_list)):-1]

    master_container_id = ''
    for container_id in container_id_list:
        ssh_submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', f'ssh-{master_machine_info["ip"]}-submit-config.yaml')
        with open(ssh_submit_config_file, encoding='utf-8', mode='r') as fp:
            ssh_config_info = yaml.safe_load(fp)

        cmd = f'ssh {ssh_config_info["config"]["username"]}@{ssh_config_info["config"]["ip"]} docker ps'
        ret = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ret.returncode:
            logging.error("Couldnt get running info")
            continue

        running_info = ret.stdout.read()
        running_info = running_info.decode('utf-8')
        running_info = running_info.split('\n')
        if len(running_info) <= 1:
            logging.error(f"Couldnt parse container info on {master_machine_info['ip']}")
            continue

        is_found = False
        for i in range(1, len(running_info)):
            if running_info[i] == '':
                continue

            container_info = running_info[i].split(' ')
            abs_container_id = container_info[0]
            if container_id.startswith(abs_container_id):
                is_found = True
                break
        
        if is_found:
            master_container_id = container_id
            break

    if master_container_id == '':
        logging.error('Couldnt find task container id.')
        return False

    # 获得container id
    with open('./.project.json', 'r') as fp:
        project_info = json.load(fp)
    project_info['exp'][exp][-1]['id'] = master_container_id
    project_info['exp'][exp][-1]['ip'] = target_machine_info_list[0]['ip']
    with open('./.project.json', 'w') as fp:
        json.dump(project_info,fp)

    # 删除临时配置:
    if os.path.exists('./extra-config.py'):
        os.remove('./extra-config.py')
    if os.path.exists('./address'):
        os.remove('./address')
    return True


def ssh_submit_3rd_process_func(create_time, exe_script, base_image, gpu_ids, cpu_num, memory_size, task_name=None, ip='', exp='', env='master', is_inner_launch=False, data_folder='/data', project_folder='', is_resource_check=False):
    # 前提假设，调用此函数前当前目录下需要存在项目代码
    # 遍历所有注册的设备，找到每个设备的空闲GPU
    with open('./.project.json', 'r') as fp:
        project_info = json.load(fp)

    gpu_num = len(gpu_ids)
    username = ''
    password = ''
    ssh_config_info = None
    logging.info("Analyze cluster environment.")
    if ip == '':
        # 自动搜索可用远程机器
        for file_name in os.listdir(os.path.join(os.environ['HOME'], '.config', 'antgo')):
            register_ip = ''
            if file_name.endswith('.yaml') and file_name.startswith('ssh'):
                terms = file_name.split('-')
                if len(terms) == 4:
                    register_ip = terms[1]
            else:
                continue

            if register_ip == '':
                continue

            ssh_submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', file_name)
            with open(ssh_submit_config_file, encoding='utf-8', mode='r') as fp:
                ssh_config_info = yaml.safe_load(fp)

            # 检查GPU占用情况
            logging.info(f'Analyze IP: {register_ip}')
            info = remote_gpu_running_info(ssh_config_info["config"]["username"], ssh_config_info["config"]["ip"])
            if len(info['free_gpus']) >= gpu_num:
                ip = ssh_config_info["config"]["ip"]
                username = ssh_config_info["config"]["username"]
                password = ssh_config_info["config"]["password"]
                free_gpus = info['free_gpus']
                break

    target_ip_list = ip.split(',')
    target_machine_info_list = []
    if is_resource_check:
        target_machine_info_list = []
        for target_ip in target_ip_list:
            ssh_submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', f'ssh-{target_ip}-submit-config.yaml')
            with open(ssh_submit_config_file, encoding='utf-8', mode='r') as fp:
                ssh_config_info = yaml.safe_load(fp)

            # 检查GPU占用情况
            info = remote_gpu_running_info(ssh_config_info["config"]["username"], ssh_config_info["config"]["ip"])
            if len(info['free_gpus']) < gpu_num:
                logging.error(f"No enough gpu in {target_ip}.")
                return

            username = ssh_config_info["config"]["username"]
            password = ssh_config_info["config"]["password"]
            target_machine_info_list.append({
                'ip': target_ip,
                'username': username,
                'password': password,
                'gpus': info['free_gpus']
            })

        if len(target_machine_info_list) != len(target_ip_list):
            logging.error("No enough machine resource")
            return
    else:
        for target_ip in target_ip_list:
            ssh_submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', f'ssh-{target_ip}-submit-config.yaml')
            with open(ssh_submit_config_file, encoding='utf-8', mode='r') as fp:
                ssh_config_info = yaml.safe_load(fp)

            username = ssh_config_info["config"]["username"]
            password = ssh_config_info["config"]["password"]
            target_machine_info_list.append({
                'ip': target_ip,
                'username': username,
                'password': password,
                'gpus': [int(g) for g in gpu_ids]
            })

    logging.info(f"Apply target machine resource {target_machine_info_list}")

    # 对于多机多卡模式，不可自定义指定GPU编号
    # 对于单机多卡模式，可以自定义指定GPU编号
    apply_gpu_id = [str(i) for i in range(gpu_num)]
    if len(target_machine_info_list) == 1:
        candidate_gpu_ids = [int(gpu_id) for gpu_id in target_machine_info_list[0]['gpus']]
        print(f'Candidate GPU-ID {candidate_gpu_ids}')
        apply_gpu_id = []
        for gd in gpu_ids:
            if gd in candidate_gpu_ids:
                apply_gpu_id.append(str(gd))

        if len(apply_gpu_id) != gpu_num:
            logging.error(f"gpus {gpu_ids} not all free")
            return
    apply_gpu_id = ','.join(apply_gpu_id)

    # 申请GPU-ID
    print(f'Apply GPU-ID {apply_gpu_id}')
    image_name = 'registry.cn-hangzhou.aliyuncs.com/vibstring/antgo-env:latest' # 基础镜像
    if base_image is not None and base_image != '':
        image_name = base_image

    if password == '':
        password = 'default'


    print(f'Use image {image_name}')
    project_name = os.path.abspath(os.path.curdir).split("/")[-1]
    submit_time = create_time

    target_machine_ips = ','.join([v['ip'] for v in target_machine_info_list])
    print(f'project_name {project_name}')
    print(f'target_machine_ips {target_machine_ips}')

    submit_script = os.path.join(os.path.dirname(__file__), 'ssh-submit.sh')
    if not is_inner_launch:
        exe_script = f'{exe_script} --device-num={gpu_num} --nnodes={len(target_machine_info_list)} --master-port=8990 --master-addr={target_machine_info_list[0]["ip"]}'
    # 设置CUDA可见性，第三方yolo框架，缺少精确控制设备能力
    exe_script = f'export CUDA_VISIBLE_DEVICES={apply_gpu_id}; {exe_script}'
    submit_cmd = f'bash {submit_script} {username} {password} {target_machine_ips} {gpu_num} {cpu_num} {memory_size}M "{exe_script}" {image_name} {project_name} {env} {submit_time} {data_folder} {project_folder}'

    # 解析提交后的输出，并解析出container id
    print('submit command')
    print(submit_cmd)

    # 记录提交机器地址
    with open('./address', 'w') as fp:
        fp.write(f'{username}@{target_machine_info_list[0]["ip"]}')

    ret = subprocess.Popen(submit_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    content = ret.stdout.read()
    content = content.decode('utf-8')
    print(content)

    # 检查返回的容器ID和机器IP对应关系
    master_machine_info = target_machine_info_list[0]
    container_id_list = content.split('\n')[(-1-len(target_machine_info_list)):-1]

    master_container_id = ''
    for container_id in container_id_list:
        ssh_submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', f'ssh-{master_machine_info["ip"]}-submit-config.yaml')
        with open(ssh_submit_config_file, encoding='utf-8', mode='r') as fp:
            ssh_config_info = yaml.safe_load(fp)

        cmd = f'ssh {ssh_config_info["config"]["username"]}@{ssh_config_info["config"]["ip"]} docker ps'
        ret = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ret.returncode:
            logging.error("Couldnt get running info")
            continue

        running_info = ret.stdout.read()
        running_info = running_info.decode('utf-8')
        running_info = running_info.split('\n')
        if len(running_info) <= 1:
            logging.error(f"Couldnt parse container info on {master_machine_info['ip']}")
            continue

        is_found = False
        for i in range(1, len(running_info)):
            if running_info[i] == '':
                continue

            container_info = running_info[i].split(' ')
            abs_container_id = container_info[0]
            if container_id.startswith(abs_container_id):
                is_found = True
                break
        
        if is_found:
            master_container_id = container_id
            break

    if master_container_id == '':
        logging.error('Couldnt find task container id.')
        return False

    # 获得container id
    with open('./.project.json', 'r') as fp:
        project_info = json.load(fp)
    project_info['exp'][exp][-1]['id'] = master_container_id
    project_info['exp'][exp][-1]['ip'] = target_machine_info_list[0]['ip']
    with open('./.project.json', 'w') as fp:
        json.dump(project_info,fp)

    if os.path.exists('./address'):
        os.remove('./address')
    return True


def ssh_submit_yolo_process_func(create_time, mode_name, dataset_name, pretrained_model, device_ids):
    # 训练和评估过程
    
    pass

# 检查任务资源是否满足
def ssh_submit_resource_check_func(gpu_num, cpu_num, memory_size):
    # TODO，支持资源检查
    return True


def ssh_data_server_func(data_name, data_server_ip, data_server_port, consumer_size, worker_num, epoch_num, loader_ip):
    # 生成数据服务代码
    pass