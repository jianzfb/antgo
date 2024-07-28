FROM registry.cn-hangzhou.aliyuncs.com/vibstring/antgo-env:latest

COPY ./ /workspace/

# eagleeye 环境更新
RUN mkdir -p /root/.3rd && cd /root/.3rd && rm -rf eagleeye && git clone https://github.com/jianzfb/eagleeye.git

# antgo 环境
RUN pip3 install --upgrade --upgrade-strategy=only-if-needed git+https://github.com/jianzfb/antgo.git@{{version}}

# eagleeye python 编译
RUN antgo eagleeye

# 修正环境信息
RUN rm /miniconda3/bin/../lib/libstdc++.so.6;ln /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /miniconda3/bin/../lib/libstdc++.so.6

# 创建并设置工作目录
WORKDIR /workspace

# 设置环境变量
ENV PATH="/workspace:${PATH}"

# 设置启动命令
ENTRYPOINT ["bash", "launch.sh"]