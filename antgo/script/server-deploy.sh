#!/usr/bin/env bash
if [ "{{image_registry}}" != "" ]; then
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
    if [ ! -d "{{project_folder}}/{{project_name}}" ]; then
        rm -rf {{project_folder}}/{{project_name}}
    fi
    if [ -f "{{project_folder}}/{{project_name}}.tar" ];then
        tar -xf {{project_folder}}/{{project_name}}.tar -C {{project_folder}}
        rm {{project_folder}}/{{project_name}}.tar
    fi
fi

# stop old container
docker stop {{name}}
docker rm {{name}}

# waiting 
sleep 2

# create data folder in container
server_data_folder=""
if [ "{{data_folder}}" != "" ]; then
    mkdir -p {{data_folder}}/{{project_name}}
    server_data_folder="{{data_folder}}/{{project_name}}"
else
    mkdir -p /data/{{project_name}}
    server_data_folder="/data/{{project_name}}"
fi

echo ${server_data_folder}
# launch in container
if [ "{{command}}" != "" ]; then
    docker run --name {{name}} --rm -d --shm-size="50G" -w {{workspace}} --gpus '"device={{gpu_id}}"' -p {{outer_port}}:{{inner_port}} -v {{project_folder}}/{{project_name}}:{{workspace}} -v ${server_data_folder}:/data --privileged {{image}} sh -c "{{command}}"
else
    docker run --name {{name}} --rm -d --shm-size="50G" -w {{workspace}} --gpus '"device={{gpu_id}}"' -p {{outer_port}}:{{inner_port}} -v ${server_data_folder}:/data --privileged {{image}}
fi