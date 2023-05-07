# 容器镜像构建主机需要连通公网

# 基础容器镜像, https://github.com/NVIDIA/nvidia-docker/wiki/CUDA
# 
# https://docs.docker.com/develop/develop-images/multistage-build/#use-multi-stage-builds
# require Docker Engine >= 17.05
#
# builder stage
FROM nvidia/cuda:12.1.0-runtime-ubuntu18.04

# 基础容器镜像的默认用户已经是 root
# USER root

# 使用国内开源镜像站提供的 pypi 配置
RUN mkdir -p /root/.pip/
COPY ./docker/pip.conf /root/.pip/pip.conf

# 拷贝待安装文件到基础容器镜像中的 /tmp 目录
COPY Miniconda3-py38_23.3.1-0-Linux-x86_64.sh /tmp
RUN mkdir -p /tmp/antgo
COPY ./ /tmp/antgo/

# 拷贝android ndk 到根目录
RUN mkdir /android-ndk-r20b
COPY android-ndk-r20b /android-ndk-r20b

# 拷贝vscode-server到基础容器镜像中的 /opt/
# COPY code-server-4.0.2-linux-amd64  /opt/

# https://conda.io/projects/conda/en/latest/user-guide/install/linux.html#installing-on-linux
# 安装 Miniconda3 到基础容器镜像的 /miniconda3 目录中
RUN bash /tmp/Miniconda3-py38_23.3.1-0-Linux-x86_64.sh -b -p /miniconda3

# 使用 Miniconda3 默认 python 环境
RUN cd /tmp/antgo && apt-get update -y && apt-get install -y gcc && /miniconda3/bin/pip install --no-cache-dir -r requirements.txt &&/miniconda3/bin/pip install opencv-python-headless && /miniconda3/bin/python setup.py build_ext install && rm -rf /tmp/*

# 安装工具（依然使用华为开源镜像站）
RUN cp -a /etc/apt/sources.list /etc/apt/sources.list.bak && \
    sed -i "s@http://.*archive.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list && \
    sed -i "s@http://.*security.ubuntu.com@http://repo.huaweicloud.com@g" /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y vim curl git && \
    apt-get install -y make gcc g++ libz-dev android-tools-adb && \
    apt-get clean && \
    mv /etc/apt/sources.list.bak /etc/apt/sources.list

# 设置容器镜像预置环境变量
# 请务必设置 PYTHONUNBUFFERED=1, 以免日志丢失
ENV PATH=$PATH:/android-ndk-r20b:/miniconda3/bin PYTHONUNBUFFERED=1 ANDROID_NDK_HOME=/android-ndk-r20b

# 初始化conda环境
RUN conda init bash

# 设置容器镜像默认用户与工作目录
RUN mkdir -p /root/workspace
USER root
WORKDIR /root/workspace
