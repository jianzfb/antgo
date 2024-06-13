if [ -d "./build" ]; 
then
  rm -rf ./build
fi

# compile
echo "start compile X86-64"
mkdir -p build
cd build
if [[ $1 == BUILD_PYTHON_MODULE ]];then
cmake -DCMAKE_BUILD_TYPE=Release -DX86_ABI=X86-64 -DBUILD_PYTHON_MODULE:BOOL=ON ..
else
cmake -DCMAKE_BUILD_TYPE=Release -DX86_ABI=X86-64 ..

fi
make
cd ../

echo "finish compile X86-64"
