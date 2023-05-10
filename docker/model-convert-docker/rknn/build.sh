#!/bin/sh

# 第一步：下载rknn-toolkit2压缩包到当前目录，从https://github.com/rockchip-linux/rknn-toolkit2

# 第二步：构建镜像
docker build -t rknnconvert .
