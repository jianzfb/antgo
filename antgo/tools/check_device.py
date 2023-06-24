import os
import json
import yaml


def check_device_info(args):
    if args.ssh:
        # 远程执行命令
        ssh_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', 'ssh-submit-config.yaml')
        with open(ssh_config_file, 'r') as fp:
          ssh_config_info = yaml.safe_load(fp)

        print(f'ACTIVE(REMOTE BY SSH): {ssh_config_info["config"]["ip"]}')
        os.system(f'ssh {ssh_config_info["config"]["username"]}@{ssh_config_info["config"]["ip"]} nvidia-smi')
    elif args.local:
        os.system('nvidia-smi')
    elif args.k8s:
        print(f'k8s not support now')
    else:
        os.system('nvidia-smi')
