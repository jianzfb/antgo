# builder stage
FROM ubuntu:18.04

RUN echo "dash dash/sh boolean false" | debconf-set-selections

RUN apt-get update && apt-get install -y python3-dev python3-pip zip
RUN pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN mkdir /workspace && mkdir /tools/
COPY requirements.txt /tools/requirements.txt
RUN python3 -m pip install --upgrade tensorrt -i https://pypi.tuna.tsinghua.edu.cn/simple && pip3 install -r /tools/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install nvidia-pyindex==1.0.4 polygraphy importlib-metadata -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN ln -s /usr/local/lib/python3.6/dist-packages/nvidia/cuda_runtime/lib/libcudart.so.12 libcudart.so

ENV PATH=$PATH:/tools/
COPY data_loader.py /tools/data_loader.py
COPY convert.py /tools/convert.py
COPY convert.sh /tools/convert.sh

USER root
WORKDIR /workspace
