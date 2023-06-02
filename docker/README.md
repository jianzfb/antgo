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
sudo bash docker/build.sh with-vscode-server
```

## 创建容器
### 创建运行环境容器
可以直接以命令行方式，运行相关实验
```
sudo docker run -it --rm --name myantgoenv --gpus all antgo-env /bin/bash
```

### 创建vscode-server服务
在线IDE环境
```
sudo docker run --rm -d -p 8080:8080 -e PASSWORD=123 -v /tmp:/tmp -v $(pwd):/workspace -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker --gpus all antgo-env /opt/code-server --host 0.0.0.0 --auth password
```

> 备注
>
> 如果nvidia环境有问题，尝试如下方法
> 
> sudo apt-get install nvidia-container-runtime
>
> sudo systemctl restart docker
