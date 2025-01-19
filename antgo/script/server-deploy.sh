#!/usr/bin/env bash
if [ {{image_registry}} != "" ]; then
    # 使用镜像中心加载镜像
    docker login --username={{user}} --password={{password}} {{image_registry}}
    docker pull {{image}}
else
    # 加载本地镜像
    if [ -f "{{image}}.tar" ];then
        docker load -i {{image}}.tar
        rm {{image}}.tar
    fi
    # 解压项目压缩包
    if [ -f "{{project_name}}.tar" ];then
        tar -xf {{project_name}}.tar
        rm {{project_name}}.tar
    fi
fi

# stop old container
docker stop {{name}}

# create data folder in container
mkdir -p /data/{{project_name}}

# launch in container
if [ {{command}} != "" ]; then
    project_dir=$(pwd)
    docker run --name {{name}} --rm -d --shm-size="50G" -w {{workspace}} --gpus "device={{gpu_id}}" -p {{outer_port}}:{{inner_port}} -v {project_dir}/{{project_name}}:{{workspace}} -v /data/{{project_name}}:/data --privileged {{image}} sh -c "{{command}}"
else
    docker run --name {{name}} --rm -d --shm-size="50G" -w {{workspace}} --gpus "device={{gpu_id}}" -p {{outer_port}}:{{inner_port}} -v /data/{{project_name}}:/data --privileged {{image}}
fi