# builder stage
FROM ubuntu:18.04

RUN echo "dash dash/sh boolean false" | debconf-set-selections

RUN apt-get update && \
    apt-get install -y libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 libsm6 libgl1-mesa-glx libprotobuf-dev gcc

RUN apt-get install -y python3-dev python3-pip zip
RUN pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY rknn-toolkit2-master.zip /
RUN unzip /rknn-toolkit2-master.zip && \
    cd /rknn-toolkit2-master && \
    pip3 install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install -r doc/requirements_cp36-1.5.0.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    cd packages && pip3 install rknn_toolkit2-1.50+1fa95b5c-cp36-cp36m-linux_x86_64.whl

RUN mkdir /workspace && mkdir /tools/
ENV PATH=$PATH:/tools/
COPY convert.py /tools/convert.py
COPY convert.sh /tools/convert.sh

USER root
WORKDIR /workspace
