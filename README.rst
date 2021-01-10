======================
Antgo
======================

.. image:: https://raw.githubusercontent.com/jianzfb/antgo/master/antgo/resource/static/card.png
    :alt: Antgo

Target
----------------------
Antgo is a machine learning experiment manage platform, which has been integrated deeply with MLTalker.
Antgo provides some easy cli commands to help ML researchers to manage, analyze, and challenge all kinds
of ML tasks.

Based on amounts of statistical evaluation methods, Antgo could give a fruitful evaluation report, which
help researchers analyze and trade-off their model.

Antgo tutorial is at `MLTalker Blog <http://www.mltalker.com/blog/>`__.

Installation
----------------------
install Antgo::

    pip install antgo


or install Antgo from source::

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
1.step Build and Apply Task

Build and Apply Task in

.. image:: https://raw.githubusercontent.com/jianzfb/antgo/master/antgo/resource/static/apply.png
    :alt: Build Task in MLTalker


2.step Create Your Project

Create Your project

.. image:: https://raw.githubusercontent.com/jianzfb/antgo/master/antgo/resource/static/start-experiment.png
    :alt: Get Task ApiToken in MLTalker


antgo startproject --name=MNIST --author=xxx --token=Task API-TOKEN

after that, you will get in current folder

.. image:: https://raw.githubusercontent.com/jianzfb/antgo/master/antgo/resource/static/filetree.png
    :alt: file tree


3.step write your train and predict code

in MNIST_main.py, you should finish training_callback and infer_callback functions.

training_callback function::

    def training_callback(data_source, dump_dir):
        # warning: data_source include data and label
        try:
            # data_source.iterator_value() get generator
            for index, (data, label) in enumerate(data_source.iterator_value()):
                # data, label is data and its label
                pass
        except:
            pass

        # build logger to record important data
        mc = mlogger.Container()
        mc.loss = mlogger.metric.Simple('model loss')

        epochs = 100
        for epoch in range(epochs):
            for _ in range(500):
                # train model
                ...
                # loss value
                loss_val = ...
                mc.loss.update(loss_val)

            # save best model
            ...

infer_callback function::

    def infer_callback(data_source, dump_dir):
        # warning: dont include label data
        # get dataset size
        data_size = data_source.size
        # parse data
        try:
            # data_source.iterator_value() get generator
            for index, data in enumerate(data_source.iterator_value()):
                pass
        except:
            pass

        # load from model from ctx.from_experiment
        # ctx.from_experiment is experiment_uuid (shell script)

        # run predict
        ...

        # record every sample predict result
        for index in range(data_size):
            ctx.recorder.record({
              'RESULT': (int)(score[index])
            })


you can go `MLTalker Blog <http://www.mltalker.com/blog/>`__, to see more cases.


4.step Run Train Task

antgo train exp


5.step Run Challenge Task

antgo challenge exp experiment_uuid