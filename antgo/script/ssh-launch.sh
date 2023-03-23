#!/usr/bin/env bash
user=$1
cpu_num=$2
memory_size=$3
gpu_num=$4
command=$5
image=$6

# 拉取项目镜像
docker pull ${image}
# 执行
cd /${user}/;tar -xf project.tar
docker run -d -v ${user}:/tiger ${image} ${command}
