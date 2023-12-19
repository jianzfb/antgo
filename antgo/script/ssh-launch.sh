#!/usr/bin/env bash
username=$1
cpu_num=$2
memory=$3
gpu_num=$4
command=$5
image=$6
project=$7

# 执行
docker run --rm -d --shm-size="50G" -w /tiger -v /home/${username}/${project}:/tiger -v /data:/dataset --gpus all --privileged ${image} sh -c "cd /tiger && $command"
