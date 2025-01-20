# 算法WEB服务发布及部署

## 创建算法管线
下面以图像尺寸变化功能举例
```
from antgo.pipeline import *
import numpy as np

with web['image'](name='demo') as handler:
    app = handler.resize_op['image', 'resized_image'](out_size=(384,256)). \
    demo(
        title="图像尺寸变换DEMO",
        description="图像尺寸变换DEMO",
		input=[
			{'data': 'image', 'type': 'image'},
		], 
		output=[
			{'data': 'resized_image', 'type': 'image'}
		]
    )
```

## 算法管线打包
将DEMO打包到镜像中
```
# name              自定义服务名字
# main              入口文件名
# port              服务端口
# version           antgo  版本分支
# image-repo        (OPTIONAL)镜像中心服务（推荐使用阿里云镜像中心）
# user              (OPTIONAL)镜像中心用户名（推荐使用阿里云镜像中心）
# password          (OPTIONAL)镜像中心密码（推荐使用阿里云镜像中心）
# mode 打包模式（http/api, http/demo, grpc, android/sdk, linux/sdk）
antgo package --name=resizefunc --main=server:app --port=8080 --version=semib --image-repo=xxx --user=xxx --password=xxx
```

## 管线服务发布1
将镜像部署到目标机器，并进行平台注册发布（进针对内部开发者使用）
```
# ip                目标机器IP地址
# port              对外服务端口
# user              (OPTIONAL)镜像中心用户名（推荐使用阿里云镜像中心）
# password          (OPTIONAL)镜像中心密码（推荐使用阿里云镜像中心）
antgo deploy --ip=xxx --port=xxx --user=xxx --password=xxx --release
```

## 管线服务发布2（无打包直接发布）
无需package进行镜像打包直接项目发布
```
# ip                目标机器IP地址
# port              对外服务端口
# mode				打包模式（http/api, http/demo, grpc, android/sdk, linux/sdk）
# image				基础镜像
# gpu-id			gpu资源
# main 				入口文件名
antgo deploy --ip=xxx --port=xxx --mode=http/api --image=xxx --gpu-id=0,1,2,3 --main=server:app
```