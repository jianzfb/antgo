if [ -d "./build" ]; 
then
  rm -rf ./build
fi

# compile
echo "start compile ${abival}"
mkdir -p build
cd build
if [[ $1 == BUILD_PYTHON_MODULE ]];then
cmake -DCMAKE_BUILD_TYPE=Release -D${abikey}=${abival} -DBUILD_PYTHON_MODULE:BOOL=ON ..
else
cmake -DCMAKE_BUILD_TYPE=Release -D${abikey}=${abival} ..

fi
make
cd ../

echo "finish compile ${abival}"
