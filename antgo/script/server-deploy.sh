#!/usr/bin/env bash
# 在目标机器需要登陆镜像中心
if [ {{image_registry}} != "" ]; then
    docker login --username={{user}} --password={{password}} {{image_registry}}
    docker pull {{image}}
else
    docker load -i {{image}}.tar
    rm {{image}}.tar
fi
# stop exist task container
docker stop {{name}}
# create container data folder
mkdir -p /data/{{image}}
# launch task container
docker run --name {{name}} --rm -d --shm-size="50G" -w {{workspace}} --gpus "device={{gpu_id}}" -p {{outer_port}}:{{inner_port}} -v /data/{{image}}:/data --privileged {{image}}
