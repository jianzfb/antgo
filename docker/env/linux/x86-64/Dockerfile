FROM registry.cn-hangzhou.aliyuncs.com/vibstring/antgo-env-dev:latest

# timezone
ENV TZ=Asia/Shanghai
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

RUN apt-get update 
# update antgo
RUN mkdir /workspace && cd /workspace && git clone --recurse-submodules --depth 1 https://github.com/jianzfb/antgo.git && \
    cd antgo && pip install -r requirements.txt && python setup.py build_ext install

WORKDIR /opt

# update zlib to 1.3.1
RUN cd /opt && wget https://www.zlib.net/zlib-1.3.1.tar.gz && tar -zxvf zlib-1.3.1.tar.gz && cd zlib-1.3.1 && ./configure && make -j 4  && make install
   
# clone eagleeye
RUN cd /root/.3rd && git clone --recurse-submodules https://github.com/jianzfb/eagleeye.git

# install ffmpeg
RUN cd /root/.3rd && git clone --recurse-submodules -b sdk/12.0 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && cd nv-codec-headers && \
    make && make install && cd /root/.3rd && git clone --recurse-submodules -b release/7.0 https://git.ffmpeg.org/ffmpeg.git && cd ffmpeg && \
    cp /root/.3rd/eagleeye/eagleeye/3rd/ffmpeg/libavformat/* /root/.3rd/ffmpeg/libavformat && \
    apt-get install -y build-essential yasm cmake libtool libc6 libc6-dev unzip wget libnuma1 libnuma-dev && \
    ./configure --prefix=./linux-install --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda-12.1/lib64 --disable-static --enable-shared && \
    make -j 8 ; make install

# install opencv
RUN cd /root/.3rd && git clone https://github.com/opencv/opencv.git -b 3.4 && \
    git clone https://github.com/opencv/opencv_contrib.git -b 3.4 && \
    cd opencv && mkdir build && cd build &&  mkdir -p /root/.3rd/opencv-install && \
    cmake -DOPENCV_EXTRA_MODULES_PATH=/root/.3rd/opencv_contrib/modules -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/root/.3rd/opencv-install -D BUILD_DOCS=OFF -D BUILD_EXAMPLES=OFF -D BUILD_opencv_apps=OFF -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=OFF -D BUILD_PERF_TESTS=OFF  -D BUILD_JAVA=OFF -D BUILD_opencv_java=OFF -D BUILD_TESTS=OFF -D WITH_FFMPEG=OFF .. && \
    make -j 8 &&  make install && cd .. && rm build -rf

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.3rd/opencv-install/lib:/root/.3rd/ffmpeg/linux-install/lib

# install eagleeye
RUN cd /root/.3rd/eagleeye && bash linux_x86_64_build_with_ffmpeg_cuda.sh

RUN rm /miniconda3/bin/../lib/libstdc++.so.6 &&  ln /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /miniconda3/bin/../lib/libstdc++.so.6