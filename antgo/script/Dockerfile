FROM registry.cn-hangzhou.aliyuncs.com/vibstring/antgo-env:latest

COPY ./ /workspace/
RUN pip3 install --upgrade --upgrade-strategy=only-if-needed git+https://github.com/jianzfb/antgo.git@{{version}}
WORKDIR /workspace
ENTRYPOINT ["launch.sh"]