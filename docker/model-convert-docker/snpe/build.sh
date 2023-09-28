#!/bin/sh

# 第一步：下载snpe工具包到当前路径，从https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk
echo "https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk"
wget http://files.mltalker.com/snpe-2.9.0.4462.zip

# 第二步：构建镜像
docker build -t snpeconvert .
docker build -t snpeconvert .