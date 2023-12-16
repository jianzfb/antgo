import os
import logging
import subprocess
import yaml
import json


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


def ls_ssh_running(exp=None):
    exp_run_info = []
    if os.path.exists('.project.json'):
        with open('.project.json', 'r') as fp:
            project_info = json.load(fp)

        for exp_name, exp_info_list in project_info['exp'].items():
            exp_run_info.extend([(info['id'], exp_name) for info in exp_info_list])

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

        print('')
        print(f'IP: {register_ip}')
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
            for exp_run_id_and_name in exp_run_info:
                if exp_run_id_and_name[0].startswith(container_id):
                    is_found = True
                    exp_name = exp_run_id_and_name[1]
                    break

            if is_found:
                print(f'  (*)id {container_id}, name {container_name}, exp {exp_name}')
            else:
                print(f'  id {container_id}, name {container_name}, exp UNKOWN')


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
    if args.ssh:
        if action_name == 'stop':
            stop_ssh_running(action_name, args.id)
        elif action_name == 'ls':
            ls_ssh_running()
        elif action_name == 'log':
            log_ssh_running(action_name, args.id)
        else:
            logging.error("Only support stop/ls/log")
    else:
        logging.error("Only support operate for ssh task")
