#!/usr/bin/env bash
username=$1
cpu_num=$2
memory=$3
gpu_num=$4
command=$5
image=$6
project=$7

# 执行
# TODO, 映射GPU,
sudo docker run -d -w /tiger -m ${memory} --cpus ${cpu_num} -v /home/${username}/${project}:/tiger ${image} sh -c "cd /tiger && $command"
