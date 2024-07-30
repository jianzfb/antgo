import os
import torch
def install_ffmpeg():
    if torch.cuda.is_available():   
        os.system('cd /root/.3rd && git clone --recurse-submodules -b sdk/12.0 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && cd nv-codec-headers && make && make install && cd /root/.3rd && git clone --recurse-submodules -b release/7.0 https://git.ffmpeg.org/ffmpeg.git && cd ffmpeg && cp /root/.3rd/eagleeye/eagleeye/3rd/ffmpeg/libavformat/* /root/.3rd/ffmpeg/libavformat && apt-get install -y build-essential yasm cmake libtool libc6 libc6-dev unzip wget libnuma1 libnuma-dev && ./configure --prefix=./linux-install --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda-12.1/lib64 --disable-static --enable-shared && make -j 8 ; make install')
    else:
        os.system('cd /root/.3rd && git clone --recurse-submodules -b release/7.0 https://git.ffmpeg.org/ffmpeg.git && cp /root/.3rd/eagleeye/eagleeye/3rd/ffmpeg/libavformat/* /root/.3rd/ffmpeg/libavformat/ && ./configure --prefix=./linux-install --enable-neon --enable-hwaccels --enable-gpl --disable-postproc --disable-debug --enable-small --enable-static --enable-shared --disable-doc --enable-ffmpeg --disable-ffplay --disable-ffprobe --disable-avdevice --disable-doc --enable-symver --pkg-config="pkg-config --static" && make clean && make -j 6 && make install')
