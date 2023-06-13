#!/bin/sh

# download dependent files
if [ ! -f "./Miniconda3-py38_23.3.1-0-Linux-x86_64.sh" ];then
    wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.3.1-0-Linux-x86_64.sh
fi

# download android ndk
if [ "$1"x = "with-android-ndk"x -o "$2"x = "with-android-ndk"x ]; then
    if [ ! -d "./android-ndk-r20b" ];then
        if [ ! -f "./android-ndk-r20b-linux-x86_64.zip" ];then
            wget https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip
        fi
        unzip android-ndk-r20b-linux-x86_64.zip
    fi

    # 自动修改Dockerfile
    python3 docker/auto_inject.py --with-android-ndk
fi

# download vscode-server
if [ "$1"x = "with-vscode-server"x -o "$2"x = "with-vscode-server"x ]; then
    if [ ! -d "./code-server-4.0.2-linux-amd64" ];then
        if [ ! -f "./code-server-4.0.2-linux-amd64.tar.gz" ];then
            wget https://github.com/coder/code-server/releases/download/v4.0.2/code-server-4.0.2-linux-amd64.tar.gz
        fi
        tar -xf code-server-4.0.2-linux-amd64.tar.gz
    fi

    # 自动修改Dockerfile
    python3 docker/auto_inject.py --with-vscode-server
fi

# 自动修改Dockerfile
python3 docker/auto_inject.py --with-dev

# build docker
sudo docker build -t antgo-env-dev -f docker/Dockerfile ./

# example: run docker
# 1. 交互式启动容器
# sudo docker run -it --rm --name antgo-env-runtime --shm-size="20G" --gpus all antgo-env /bin/bash
# 2. vscode-server服务
# sudo docker run --rm -d --name antgo-env-ide --shm-size="20G" -p 8080:8080 -e PASSWORD=123 -v /tmp:/tmp -v $(pwd):/workspace -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker --gpus all antgo-env-dev /opt/code-server --host 0.0.0.0 --auth password