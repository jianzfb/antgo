# 1.step install rocksdb
sudo apt-get install -y libgflags-dev libsnappy-dev zlib1g-dev libbz2-dev libzstd-dev
git clone https://github.com/facebook/rocksdb.git
cd rocksdb/
make all

# 2.step install ipfs

# 3.step install graphviz


# 4.step install antgo
git clone https://github.com/jianzfb/antgo.git
cd antgo
pip3 install -r requirements.txt
python3 setup.py build_ext install