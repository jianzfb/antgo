# Build as antgo/server

FROM jupyter/scipy-notebook:8f56e3c47fec
MAINTAINER Project Antgo <jian@mltalker.com>

# install antgo and its dependence
USER root
ADD install.sh install.sh
RUN bash install.sh

# set enviroment variable
ENV CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:/rocksdb/include
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/rocksdb
ENV LIBRARY_PATH=${LIBRARY_PATH}:/rocksdb