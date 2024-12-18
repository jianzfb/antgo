FROM registry.cn-hangzhou.aliyuncs.com/vibstring/antgo-env:latest

COPY ./ /workspace/


# antgo 环境
ARG IS_ANTGO_UPGRADE="{{version}}"
RUN if [ "$IS_ANTGO_UPGRADE" != "master" ]; then \
        pip3 install --upgrade --upgrade-strategy=only-if-needed git+https://github.com/jianzfb/antgo.git@{{version}}; \
    fi

# eagleeye 环境更新
ARG IS_EAGLEEYE_UPGRADE="{{is_upgrade}}"
RUN if [ "$IS_EAGLEEYE_UPGRADE" = "upgrade" ]; then \
        mkdir -p /root/.3rd && cd /root/.3rd && rm -rf eagleeye && git clone https://github.com/jianzfb/eagleeye.git && antgo install eagleeye; \
    fi

# 修正环境信息
RUN rm /miniconda3/bin/../lib/libstdc++.so.6;ln /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /miniconda3/bin/../lib/libstdc++.so.6

# 安装自定义
RUN if [ -e /workspace/install.sh ]; then \
        cd /workspace && bash install.sh; \
    fi

# 创建并设置工作目录
WORKDIR /workspace

# 设置环境变量
ENV PATH="/workspace:${PATH}"

# 设置启动命令
ENTRYPOINT ["bash", "launch.sh"]