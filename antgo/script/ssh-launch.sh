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
echo docker run -d --shm-size="50G" -w /tiger/${project} -v /home/${username}/${submit_time}:/tiger -v /data:/dataset --gpus all --ipc=host --net=host --privileged ${image} sh -c "$command"
docker run -d --shm-size="50G" -w /tiger/${project} -v /home/${username}/${submit_time}:/tiger -v /data:/dataset --gpus all --ipc=host --net=host --privileged ${image} sh -c "$command"
else
echo docker run --rm -d --shm-size="50G" -w /tiger/${project} -v /home/${username}/${submit_time}:/tiger -v /data:/dataset --gpus all --ipc=host --net=host --privileged ${image} sh -c "pip3 install --upgrade --upgrade-strategy=only-if-needed git+https://github.com/jianzfb/antgo.git@$env && $command"
docker run --rm -d --shm-size="50G" -w /tiger/${project} -v /home/${username}/${submit_time}:/tiger -v /data:/dataset --gpus all --ipc=host --net=host --privileged ${image} sh -c "pip3 install --upgrade --upgrade-strategy=only-if-needed git+https://github.com/jianzfb/antgo.git@$env && $command"
fi
