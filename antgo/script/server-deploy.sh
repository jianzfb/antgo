#!/usr/bin/env bash
# 在目标机器需要登陆镜像中心
# docker login --username={{user}} {{image_registry}}
docker pull {{image}}
docker run --rm -d --shm-size="50G" -w /workspace --gpus "device={{gpu_id}}" -p {{outer_port}}:{{inner_port}} --privileged {{image}}