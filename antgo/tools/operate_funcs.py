import os
import logging
import subprocess
import yaml
import json
import psutil
import pandas as pd


def stop_ssh_running(action_name, id):
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

        cmd = f'ssh {ssh_config_info["config"]["username"]}@{ssh_config_info["config"]["ip"]} docker ps'
        ret = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ret.returncode:
            continue

        running_info = ret.stdout.read()
        running_info = running_info.decode('utf-8')
        running_info = running_info.split('\n')
        if len(running_info) <= 1:
            return

        for i in range(1, len(running_info)):
            if running_info[i] == '':
                continue

            container_info = running_info[i].split(' ')
            container_id = container_info[0]
            if id == container_id:
                os.system(f'ssh {ssh_config_info["config"]["username"]}@{ssh_config_info["config"]["ip"]} docker stop {id}')
                return

    print(f'Not found running id {id}')


def ls_ssh_running(exp=None, is_a=False):
    if not os.path.exists('.project.json'):
        print('Not found .project.json')

    exp_run_info = []
    if os.path.exists('.project.json'):
        with open('.project.json', 'r') as fp:
            project_info = json.load(fp)

        for exp_name, exp_info_list in project_info['exp'].items():
            exp_run_info.extend([(info['id'], exp_name, info['ip'], info['root'], info['create_time'], info['config']) for info in exp_info_list])

    display_info = {}
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

        cmd = f'ssh {ssh_config_info["config"]["username"]}@{ssh_config_info["config"]["ip"]} docker ps'
        ret = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ret.returncode:
            logging.error("Couldnt get running info")
            continue
        
        display_info[register_ip] = []

        running_info = ret.stdout.read()
        running_info = running_info.decode('utf-8')
        running_info = running_info.split('\n')
        if len(running_info) <= 1:
            print('No running info')
            return

        for i in range(1, len(running_info)):
            if running_info[i] == '':
                continue

            container_info = running_info[i].split(' ')
            container_id = container_info[0]
            container_name = container_info[-1]

            is_found = False
            exp_name = ''
            exp_root = ''
            exp_create_time = ''
            exp_config = ''
            for exp_run_id_and_name in exp_run_info:
                if (exp_run_id_and_name[2] == register_ip) and (exp_run_id_and_name[0].startswith(container_id)):
                    is_found = True
                    exp_name = exp_run_id_and_name[1]
                    exp_root = exp_run_id_and_name[3]
                    exp_create_time = exp_run_id_and_name[4]
                    exp_config = exp_run_id_and_name[5]
                    break

            if is_found:
                display_info[register_ip].append(
                    {
                        'status': 'running',
                        'container_id': container_id,
                        'name': exp_name,
                        'create_time': exp_create_time,
                        'root': exp_root,
                        'config': exp_config
                    }
                )

        if is_a:
            # 发现所有已经停止的信息(在 当前的设备内)
            for exp_run_id_and_name in exp_run_info:
                if exp_run_id_and_name[2] != register_ip:
                    continue
                
                is_found = False
                for i in range(1, len(running_info)):
                    if running_info[i] == '':
                        continue
                    container_id = container_info[0]
                    if exp_run_id_and_name[0].startswith(container_id):
                        is_found = True
                        break

                if not is_found:
                    display_info[register_ip].append(
                        {
                            'status': 'stop',
                            'container_id': exp_run_id_and_name[0],
                            'name': exp_run_id_and_name[1],
                            'create_time': exp_run_id_and_name[4],
                            'root': exp_run_id_and_name[3],
                            'config': exp_run_id_and_name[5]
                        }
                    )

    for record_ip, record_info in display_info.items():
        print('')
        print(f'IP: {record_ip}')

        # 时间排序
        sorted(record_info, key=lambda ff: ff['create_time']) 
        
        if len(record_info) == 0:
            continue

        # 显示
        display_info_format = {
            "status": [],
            "id": [],
            "name": [],
            "config": [],
            "create_time": [],
            "root": []
        }

        for exp_info in record_info:
            display_info_format['id'].append(exp_info["container_id"])
            display_info_format['name'].append(exp_info["name"])
            display_info_format['config'].append(exp_info["config"])
            display_info_format['create_time'].append(exp_info["create_time"])
            display_info_format['root'].append(exp_info["root"])            
            if exp_info['status'] == 'running':
                display_info_format['status'].append('*')
            else:
                display_info_format['status'].append('-')

        df = pd.DataFrame(display_info_format)
        print(df.to_string(index=False))


def log_ssh_running(action_name, id):
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

        cmd = f'ssh {ssh_config_info["config"]["username"]}@{ssh_config_info["config"]["ip"]} docker ps'
        ret = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ret.returncode:
            continue

        running_info = ret.stdout.read()
        running_info = running_info.decode('utf-8')
        running_info = running_info.split('\n')
        if len(running_info) <= 1:
            return

        for i in range(1, len(running_info)):
            if running_info[i] == '':
                continue

            container_info = running_info[i].split(' ')
            container_id = container_info[0]
            if id.startswith(container_id):
                os.system(f'ssh {ssh_config_info["config"]["username"]}@{ssh_config_info["config"]["ip"]} docker logs -f {id}')
                return

    print(f'Not found running id {id}')


def operate_on_running_status(action_name, args):
    if action_name not in ['stop', 'ls', 'log']:
        logging.error("Only support stop/ls/log")
        return

    if args.k8s:
        pass
    else:
        # ssh 远程
        if action_name == 'stop':
            stop_ssh_running(action_name, args.id)
        elif action_name == 'ls':
            ls_ssh_running(args.exp, args.a)
        elif action_name == 'log':
            log_ssh_running(action_name, args.id)
