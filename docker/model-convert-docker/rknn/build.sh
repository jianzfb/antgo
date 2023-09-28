#!/bin/sh

# 第一步：下载rknn-toolkit2压缩包到当前目录，从https://github.com/rockchip-linux/rknn-toolkit2
if [ ! -f "./rknn-toolkit2-master.zip" ];then
    wget http://files.mltalker.com/rknn-toolkit2-master.zip
fi

# 第二步：构建镜像
docker build -t rknnconvert .
