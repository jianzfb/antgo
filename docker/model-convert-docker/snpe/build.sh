#!/bin/sh

# 第一步：下载snpe工具包到当前路径，从https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk

# 第二步：构建镜像
docker build -t snpeconvert .