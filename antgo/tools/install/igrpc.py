import os

def install_grpc():
    os.system('apt-get update && apt-get install -y build-essential autoconf libtool pkg-config libsystemd-dev cmake && cd /tmp && git clone --recurse-submodules -b v1.62.0 --depth 1 --shallow-submodules http://github.com/grpc/grpc && cd grpc && mkdir -p cmake/build && cd cmake/build && cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF ../.. && make -j 8 && make install && cd /tmp && rm grpc -rf')