#!/usr/bin/env bash
username=$1
cpu_num=$2
memory=$3
gpu_num=$4
command=$5
image=$6
project=$7
env=$8
submit_time=$9

# 执行
if [ "$env" = "-" ] ; then
docker run --rm -d --shm-size="50G" -w /tiger -v /home/${username}/${submit_time}:/tiger -v /data:/dataset --gpus all --ipc=host --net=host --privileged ${image} sh -c "cd /tiger/${project} && $command"
else
docker run --rm -d --shm-size="50G" -w /tiger -v /home/${username}/${submit_time}:/tiger -v /data:/dataset --gpus all --ipc=host --net=host --privileged ${image} sh -c "cd /tiger/${project} && pip3 install --upgrade --upgrade-strategy=only-if-needed git+https://github.com/jianzfb/antgo.git@$env && $command"
fi
# docker run --rm -d --shm-size="50G" -w /tiger -v /home/${username}/${project}:/tiger -v /data:/dataset --gpus all --ipc=host --net=host ${image} sh -c "cd /tiger && pip3 install --upgrade --upgrade-strategy=only-if-needed git+https://github.com/jianzfb/antgo.git@2.2 && $command"
