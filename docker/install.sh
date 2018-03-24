# 1.step install rocksdb
sudo apt-get update
sudo apt-get install -y build-essential libgflags-dev libsnappy-dev zlib1g-dev libbz2-dev liblz4-dev
git clone https://github.com/facebook/rocksdb.git
cd rocksdb/
make shared_lib
export CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH}:`pwd`/include
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:`pwd`
export LIBRARY_PATH=${LIBRARY_PATH}:`pwd`
cd ..

# 2.step install ipfs
wget -q https://raw.githubusercontent.com/ipfs/install-go-ipfs/master/install-ipfs.sh
chmod +x install-ipfs.sh
./install-ipfs.sh

# 3.step install graphviz
sudo apt-get install graphviz

# 4.step install antgo
git clone https://github.com/jianzfb/antgo.git
cd antgo
pip install -r requirements.txt
python setup.py build_ext install