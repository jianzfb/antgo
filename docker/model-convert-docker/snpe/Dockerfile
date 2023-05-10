# builder stage
FROM ubuntu:18.04

RUN echo "dash dash/sh boolean false" | debconf-set-selections
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-dev python3-matplotlib python3-numpy python3-protobuf python3-scipy python3-skimage python3-sphinx wget zip libc++-9-dev cmake libprotobuf-dev protobuf-compiler python3-pip

ENV alias python='python3'
RUN pip3 install --upgrade setuptools && pip3 install --upgrade pip
RUN pip3 install onnx==1.11.0 onnx-simplifier packaging onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple

ENV SNPE_ROOT=/tools/snpe-2.9.0.4462
ENV ONNX_DIR=/usr/local/lib/python3.6/dist-packages/onnx

COPY snpe-2.9.0.4462.zip /tools/
RUN cd /tools/ && unzip snpe-2.9.0.4462.zip && rm snpe-2.9.0.4462.zip
RUN echo "source $SNPE_ROOT/bin/envsetup.sh -o $ONNX_DIR" | cat >> ~/.bashrc

ENV PATH=$PATH:/tools/
RUN mkdir /workspace
COPY convert.py /tools/convert.py
COPY convert.sh /tools/convert.sh

USER root
WORKDIR /workspace
