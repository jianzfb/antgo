import os
import logging
import subprocess
import yaml


def stop_ssh_running(action_name, id):
    ssh_submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', 'ssh-submit-config.yaml')
    if not os.path.exists(ssh_submit_config_file):
        logging.error('No ssh config.')
        return False

    with open(ssh_submit_config_file, encoding='utf-8', mode='r') as fp:
        ssh_config_info = yaml.safe_load(fp)

    username = ssh_config_info['config']['username']
    password = ssh_config_info['config']['password']
    ip = ssh_config_info['config']['ip']

    print(f'ACTIVE(REMOTE BY SSH): {ssh_config_info["config"]["ip"]}')
    os.system(f'ssh {ssh_config_info["config"]["username"]}@{ssh_config_info["config"]["ip"]} docker stop {id}')

def ls_ssh_running():
    ssh_submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', 'ssh-submit-config.yaml')
    if not os.path.exists(ssh_submit_config_file):
        logging.error('No ssh config.')
        return False

    with open(ssh_submit_config_file, encoding='utf-8', mode='r') as fp:
        ssh_config_info = yaml.safe_load(fp)
    
    username = ssh_config_info['config']['username']
    password = ssh_config_info['config']['password']
    ip = ssh_config_info['config']['ip']

    print(f'ACTIVE(REMOTE BY SSH): {ssh_config_info["config"]["ip"]}')
    cmd = f'ssh {ssh_config_info["config"]["username"]}@{ssh_config_info["config"]["ip"]} docker ps'
    ret = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if ret.returncode:
        logging.error("Couldnt get running info")

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

        print(f'running id {container_id}, name {container_name}')


def log_ssh_running(action_name, id):
    ssh_submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', 'ssh-submit-config.yaml')
    if not os.path.exists(ssh_submit_config_file):
        logging.error('No ssh config.')
        return False

    with open(ssh_submit_config_file, encoding='utf-8', mode='r') as fp:
        ssh_config_info = yaml.safe_load(fp)

    username = ssh_config_info['config']['username']
    password = ssh_config_info['config']['password']
    ip = ssh_config_info['config']['ip']

    print(f'ACTIVE(REMOTE BY SSH): {ssh_config_info["config"]["ip"]}')
    os.system(f'ssh {ssh_config_info["config"]["username"]}@{ssh_config_info["config"]["ip"]} docker logs -f {id}')


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
