import os
import json
import yaml


def check_device_info(args):
    if args.ssh:
        # 远程执行命令
        if args.ip == "":
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
                
                print()
                print()
                print(f'IP: {register_ip}')
                with open(os.path.join(os.environ['HOME'], '.config', 'antgo', file_name), 'r') as fp:
                    ssh_config_info = yaml.safe_load(fp)
                os.system(f'ssh {ssh_config_info["config"]["username"]}@{ssh_config_info["config"]["ip"]} nvidia-smi')
            return

        ssh_submit_config_file = os.path.join(os.environ['HOME'], '.config', 'antgo', f'ssh-{args.ip}-submit-config.yaml')
        if not os.path.exists(ssh_submit_config_file):
            logging.error('No ssh submit config.')
            logging.error('Please run antgo submitter update --config= --ssh')
            return
        with open(ssh_submit_config_file, 'r') as fp:
            ssh_config_info = yaml.safe_load(fp)
        print()
        print()
        print(f'IP: {args.ip}')
        os.system(f'ssh {ssh_config_info["config"]["username"]}@{ssh_config_info["config"]["ip"]} nvidia-smi')
    elif args.k8s:
        print(f'k8s not support now')
    else:
        os.system('nvidia-smi')
