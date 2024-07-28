FROM arm64v8/ubuntu:22.04

# timezone
ENV TZ=Asia/Shanghai
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

RUN apt-get update && apt install -y git pip build-essential autoconf libtool pkg-config libsystemd-dev cmake  curl zip unzip tar ninja-build

# install grpc
RUN cd /tmp && git clone --recurse-submodules -b v1.62.0 --depth 1 --shallow-submodules http://github.com/grpc/grpc && \
    cd grpc && mkdir -p cmake/build && cd cmake/build && \
    cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF ../.. && \
    make -j 6 && make install && cd /tmp && rm grpc -rf

ENV VCPKG_FORCE_SYSTEM_BINARIES=1

# install vcpkg && minio-cpp
RUN cd /opt && git clone -b master --depth 1 https://github.com/microsoft/vcpkg.git && \
    ./vcpkg/bootstrap-vcpkg.sh && ./vcpkg/vcpkg integrate install && /opt/vcpkg/vcpkg install minio-cpp

# install antgo (not support)
# RUN mkdir /workspace && cd /workspace && git clone https://github.com/jianzfb/antgo.git && \
#     cd antgo && pip install -r requirements.txt && python setup.py build_ext install

# clone eagleeye
RUN cd /root/.3rd && git clone --recurse-submodules https://github.com/jianzfb/eagleeye.git  

# install ffmpeg
RUN mkdir /root/.3rd/ffmpeg && git clone -b release/7.0 https://git.ffmpeg.org/ffmpeg.git /root/.3rd/ffmpeg && cd /root/.3rd/ffmpeg && \
    cp /root/.3rd/eagleeye/eagleeye/3rd/ffmpeg/libavformat/* /root/.3rd/ffmpeg/libavformat && \
    ./configure --prefix=./linux-install --enable-neon --enable-hwaccels --enable-gpl --disable-postproc --disable-debug --enable-small \
    --enable-static --enable-shared --disable-doc --enable-ffmpeg --disable-ffplay --disable-ffprobe \
    --disable-avdevice --disable-doc --enable-symver --pkg-config="pkg-config --static" && make clean && make -j 6 && make install

# install opencv
RUN cd /root/.3rd && git clone https://github.com/opencv/opencv.git -b 3.4 && \
    git clone https://github.com/opencv/opencv_contrib.git -b 3.4 && \
    cd opencv && mkdir build && cd build &&  mkdir -p /root/.3rd/opencv-install && \
    cmake -DOPENCV_EXTRA_MODULES_PATH=/root/.3rd/opencv_contrib/modules -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/root/.3rd/opencv-install \
    -D BUILD_DOCS=OFF -D BUILD_EXAMPLES=OFF -D BUILD_opencv_apps=OFF -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=OFF -D BUILD_PERF_TESTS=OFF  \
    -D BUILD_JAVA=OFF -D BUILD_opencv_java=OFF -D BUILD_TESTS=OFF -D WITH_FFMPEG=OFF .. && \
    make -j 6 &&  make install && cd .. && rm build -rf

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.3rd/opencv-install/lib:/root/.3rd/ffmpeg/linux-install/lib

# install rga
RUN mkdir /root/.3rd/rk && git clone https://github.com/airockchip/librga.git /root/.3rd/rk/librga 

# install mpp
RUN cd /root/.3rd/rk && git clone https://github.com/rockchip-linux/mpp.git &&  cd mpp/build/linux/aarch64 && \
    sed -i 's/aarch64-linux-gnu-gcc/\/usr\/bin\/gcc/g' arm.linux.cross.cmake && \
    sed -i 's/aarch64-linux-gnu-g++/\/usr\/bin\/g++/g' arm.linux.cross.cmake && \
    bash make-Makefiles.bash && make -j 6 

# install rknpu2
RUN cd /root/.3rd/rk &&  git clone https://github.com/rockchip-linux/rknpu2.git