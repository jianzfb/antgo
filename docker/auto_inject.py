import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description=f'Dockerfile')

    parser.add_argument(
        '--with-vscode-server',
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
                if content == '# COPY code-server-4.0.2-linux-amd64  /opt/':
                    cmd_list.append('COPY code-server-4.0.2-linux-amd64  /opt/')
                if content == 'COPY code-server-4.0.2-linux-amd64  /opt/':
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
        with open(os.path.join(os.path.dirname(__file__), 'Dockerfile'), 'r') as fp:
            content = fp.readline()
            content = content.strip()

            while True:
                if is_found_pos and not is_replace_base_image:
                    content = 'FROM nvidia/cuda:12.1.0-devel-ubuntu18.04'
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
        with open(os.path.join(os.path.dirname(__file__), 'Dockerfile'), 'r') as fp:
            content = fp.readline()
            content = content.strip()

            while True:
                if is_found_pos and not is_replace_base_image:
                    content = 'FROM nvidia/cuda:12.1.0-runtime-ubuntu18.04'
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