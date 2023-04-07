# download dependent files
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# build docker
sudo docker build -t antgo-env -f docker/Dockerfile ./
rm Miniconda3-latest-Linux-x86_64.sh