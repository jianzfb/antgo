if [ -d "./build" ]; 
then
  rm -rf ./build
fi

# compile
echo "start compile ${abival}"
mkdir -p build
cd build

tool_chain_path="/opt/cross_build/linux-arm64/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu"
cmake -DCMAKE_BUILD_TYPE=Release \
  -D${abikey}=${abival}  \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64  \
  -DTOOLCHAIN_PATH=$tool_chain_path \
  -DCMAKE_C_COMPILER=$tool_chain_path/bin/aarch64-none-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=$tool_chain_path/bin/aarch64-none-linux-gnu-g++ \
  -DCMAKE_FIND_ROOT_PATH="$tool_chain_path/aarch64-linux-gnu;/opt/cross_build/linux-arm64/zlib-1.3.1" \
  -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
  -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
  -DZLIB_ROOT=/opt/cross_build/linux-arm64/zlib-1.3.1 \
  -DZLIB_INCLUDE_DIR=/opt/cross_build/linux-arm64/zlib-1.3.1 \
  -DZLIB_LIBRARY=/opt/cross_build/linux-arm64/zlib-1.3.1/libz.so \
  -DMINIO:BOOL=OFF \
  ..

make
cd ../

echo "finish compile ${abival}"
