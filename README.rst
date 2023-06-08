======================
Antgo
======================

.. image:: https://raw.githubusercontent.com/jianzfb/antgo/master/antgo/resource/static/card.png
    :alt: Antgo

Target
----------------------
Antgo is a machine learning experiment manage platform, which has been integrated deeply with MLTalker.
Antgo provides a one-stop model development,  deployment, analyze, auto-optimize and manage environment.

Installation
----------------------
1. (RECOMMENDED) use docker

    `docker environment <docker/README.md>`__.

2. install from pip

    pip install antgo


3. install from source

    1. git clone https://github.com/jianzfb/antgo.git
    2. cd antgo
    3. pip install -r requirements.txt
    4. python setup.py build_ext install


Register
-----------------------
Register in `MLTalker <http://www.mltalker.com/>`__.

.. image:: https://raw.githubusercontent.com/jianzfb/antgo/master/antgo/resource/static/register.png
    :alt: Antgo and MLTalker

All user experiment records would be managed by MLTalker in user's personal page.


Quick Example
-----------------------

1.step create mvp code(cifar10 classification task)

    antgo create mvp --name=cifar10

2.step start training process

    python3 ./cifar10/main.py --exp=cifar10 --gpu-id=0 --process=train

3.step check training log

    in ./output/cifar10/output/checkpoint

4.step export onnx model

    python3 ./cifar10/main.py --exp=cifar10 --checkpoint=./output/cifar10/output/checkpoint/epoch_1500.pth --process=export

