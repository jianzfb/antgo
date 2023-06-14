# 开发环境创建

## 构建镜像
依赖文件包括
（如果网络环境允许，运行docker/build.sh将自行下载对应文件）
* Miniconda3-py38_23.3.1-0-Linux-x86_64.sh
    可以从 https://repo.anaconda.com/miniconda/Miniconda3-py38_23.3.1-0-Linux-x86_64.sh下载
* android-ndk-r20b-linux-x86_64.zip
    可以从https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip下载
* code-server-4.0.2-linux-amd64.tar.gz
    可以从https://github.com/coder/code-server/releases/download/v4.0.2/code-server-4.0.2-linux-amd64.tar.gz下载


```
# 构建运行环境镜像
sudo bash docker/build_runtime.sh

# 构建ide开发环境镜像
sudo bash docker/build_dev.sh with-android-ndk with-vscode-server
```

## 创建容器
### 创建运行环境容器
可以直接以命令行方式，运行相关实验
```
# sudo docker run -it --rm --name antgo-env-runtime --shm-size="20G" --gpus all --privileged antgo-env /bin/bash

```

### 创建vscode-server服务
在线IDE环境，运行后你可以访问http://IP:8080，开始在线开发吧
```
sudo docker run --rm -d --name antgo-env-ide --shm-size="20G" -p 8080:8080 -e PASSWORD=123 -v /tmp:/tmp -v $(pwd):/workspace -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker --gpus all --privileged antgo-env-dev /opt/code-server --host 0.0.0.0 --auth password
```


## 备注

1. 如果nvidia环境有问题，尝试如下方法
```
    sudo apt-get install nvidia-container-runtime
    sudo systemctl restart docker
```
2. 在一段随机时间后，如果容器中报GPU不可用，并且nvidia-smi返回如下错误
   
```
    “Failed to initialize NVML: Unknown Error”. **A restart of all the containers fixes the issue and the GPUs return available**.
```
可以采用如下方法解决，在宿主机的/etc/nvidia-container-runtime/config.toml文件中修改参数
```
no-cgroups = false
```
然后，重启docker以及容器，
```
sudo systemctl restart docker
sudo docker run --rm -d --name antgo-env-ide -p 8080:8080 -e PASSWORD=123 -v /tmp:/tmp -v $(pwd):/workspace -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker --gpus all --privileged antgo-env-dev /opt/code-server --host 0.0.0.0 --auth password
```

3. 如果Code Server页面 对于图片和markdown无法正常预览，可以通过如下方式解决
   
   在 chrome浏览器地址栏中，键入chrome://flags/，进入实验页面，在“搜索标志”中输入Insecure origins treated as secure，启用该功能，
并在下面的编辑框中输入code-server的IP地址和端口。修改后点击下方弹出的重启按钮。

![](http://image.mltalker.com/Untitled.png)

