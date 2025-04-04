import os
import sys
import argparse
import subprocess
import re

def get_cuda_version():
    try:
        content = subprocess.check_output('nvidia-smi').decode('utf-8')
    except:
        return None

    driver_version = re.findall('(?<=CUDA Version: )[\d.]+', content)[0]
    return f'{driver_version}.0'

def main():
    parser = argparse.ArgumentParser(description=f'Dockerfile')

    parser.add_argument(
        '--with-vscode-server',
        action='store_true', help="")
    parser.add_argument(
        '--with-android-ndk',
        action='store_true', help="")
    parser.add_argument(
        '--with-dev',
        action='store_true', help="")
    parser.add_argument(
        '--with-runtime',
        action='store_true', help="")        
    args = parser.parse_args()

    if args.with_vscode_server:
        cmd_list = []
        with open(os.path.join(os.path.dirname(__file__), 'Dockerfile'), 'r') as fp:
            content = fp.readline()
            content = content.strip()
            
            while True:
                if content == '# COPY code-server-4.92.2-linux-amd64  /opt/':
                    cmd_list.append('COPY code-server-4.92.2-linux-amd64  /opt/')
                if content == 'COPY code-server-4.92.2-linux-amd64  /opt/':
                    content = fp.readline()
                    content = content.strip()                    
                    continue

                cmd_list.append(content)
                if content == 'WORKDIR /root/workspace':
                    break

                content = fp.readline()
                content = content.strip()

        with open(os.path.join(os.path.dirname(__file__), 'Dockerfile'), 'w') as fp:
            for cmd_str in cmd_list:
                fp.write(f'{cmd_str}\n')

    if args.with_android_ndk:
        cmd_list = []
        with open(os.path.join(os.path.dirname(__file__), 'Dockerfile'), 'r') as fp:
            content = fp.readline()
            content = content.strip()

            while True:
                if content == '# RUN mkdir /android-ndk-r20b':
                    cmd_list.append('RUN mkdir /android-ndk-r20b')
                    cmd_list.append('COPY android-ndk-r20b /android-ndk-r20b')

                    content = fp.readline()
                    content = content.strip() 
                    continue

                cmd_list.append(content)
                if content == 'WORKDIR /root/workspace':
                    break

                content = fp.readline()
                content = content.strip()

        with open(os.path.join(os.path.dirname(__file__), 'Dockerfile'), 'w') as fp:
            for cmd_str in cmd_list:
                fp.write(f'{cmd_str}\n')

    if args.with_dev:
        cmd_list = []
        is_found_pos = False
        is_replace_base_image = False
        cuda_version = get_cuda_version()
        with open(os.path.join(os.path.dirname(__file__), 'Dockerfile'), 'r') as fp:
            content = fp.readline()
            content = content.strip()

            while True:
                if is_found_pos and not is_replace_base_image:
                    content = f'FROM nvidia/cuda:{cuda_version}-devel-ubuntu22.04'
                    is_replace_base_image = True

                if content == '# builder stage':
                    is_found_pos = True

                cmd_list.append(content)

                if content == 'WORKDIR /root/workspace':
                    break

                content = fp.readline()
                content = content.strip()

        with open(os.path.join(os.path.dirname(__file__), 'Dockerfile'), 'w') as fp:
            for cmd_str in cmd_list:
                fp.write(f'{cmd_str}\n')

    if args.with_runtime:
        cmd_list = []
        is_found_pos = False
        is_replace_base_image = False
        cuda_version = get_cuda_version()
        with open(os.path.join(os.path.dirname(__file__), 'Dockerfile'), 'r') as fp:
            content = fp.readline()
            content = content.strip()

            while True:
                if is_found_pos and not is_replace_base_image:
                    content = f'FROM nvidia/cuda:{cuda_version}-runtime-ubuntu22.04'
                    is_replace_base_image = True

                if content == '# builder stage':
                    is_found_pos = True

                cmd_list.append(content)

                if content == 'WORKDIR /root/workspace':
                    break

                content = fp.readline()
                content = content.strip()

        with open(os.path.join(os.path.dirname(__file__), 'Dockerfile'), 'w') as fp:
            for cmd_str in cmd_list:
                fp.write(f'{cmd_str}\n')


if __name__ == '__main__':
  main()
