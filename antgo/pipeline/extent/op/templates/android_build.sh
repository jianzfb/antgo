if [ -d "./build" ]; 
then
  rm -rf ./build
fi

# compile
echo "start compile arm64-v8a"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_STL=c++_shared -DANDROID_NATIVE_API_LEVEL=android-23 ..
make
cd ../


# install (package)
mkdir -p ./package/arm64-v8a/${project}
cp -r ./bin/arm64-v8a/*.so ./package/arm64-v8a/${project}/
echo "finish compile arm64-v8a"
echo "FINISH-arm64-v8a"
