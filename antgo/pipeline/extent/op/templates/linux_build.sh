if [ -d "./build" ]; 
then
  rm -rf ./build
fi

# compile
echo "start compile X86-64"
mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=Release -DX86_ABI=X86-64 ..
make
cd ../


# install (package)
mkdir -p ./package/X86-64/${project}
cp -r ./bin/X86-64/*.so ./package/X86-64/${project}/
echo "finish compile X86-64"
echo "FINISH-X86-64"
