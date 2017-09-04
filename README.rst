======================
antgo
======================
Target
----------------------
antgo is a machine learning experiment manage platform, which has been integrated deeply with MLTalker.
antgo provides some easy cli commands to help ML researchers to manage, analyze, and challenge all kinds
of ML tasks.

Based on amounts of statistical evaluation methods, antgo could give a fruitful evaluation report, which
help researchers analyze their model and users trade-off.

Installation
----------------------
install 3rd software or packages::

    1. leveldb
        Ubuntu
        apt-get install libleveldb1 libleveldb-dev
        Centos
        yum install epel-release
        yum install leveldb-devel
    2. graphviz
        http://www.graphviz.org/Download_linux_ubuntu.php

install antgo::

    1. git clone https://github.com/jianzfb/antgo.git
    2. cd antgo
    3. pip install -r requirements.txt
    4. python setup.py build_ext install


Example
-----------------------
1. Run Train Task::

    (1) build running main file (eg. training_task.py)
        from antgo.context import *
        from antgo.dataflow.common import *

        # 1.step ctx take control interaction with antgo
        ctx = Context()

        # 2.step build visualization channel
        # curve channel
        task12_loss_channel = ctx.job.create_channel("task12-loss","NUMERIC")
        task1_loss_channel = ctx.job.create_channel("task1-loss","NUMERIC")
        task2_loss_channel = ctx.job.create_channel("task2-loss","NUMERIC")

        # histogram channel
        histogram_channel = ctx.job.create_channel("Layer1-activation-histogram",'HISTOGRAM')

        # build chart (bind multi-channels)
        ctx.job.create_chart([task12_loss_channel, task1_loss_channel, task2_loss_channel],"Loss Curve","step","value")
        ctx.job.create_chart([histogram_channel],"Weight","value","frequence")

        # 3.step custom training process
        def training_callback(data_source,dump_dir):
            # data_source: data generator
            # dump_dir: save your training intermidiate data
            # 3.1 step stack batch
            stack_batch = BatchData(Node.inputs(data_source, batch_size=16)

            # 3.2 step running some epochs
            iter = 0
            for epoch in range(ctx.params.max_epochs):
                for data, label in stack_batch.iterator_value():
                    # run once iterator
                    loss, loss_1, loss_2, weight = your_training_model(data, label)

                    # send running information
                    task12_loss_channel.send(x=iter, y=loss)
                    task1_loss_channel.send(x=iter, y=loss_1)
                    task2_loss_channel.send(x=iter, y=loss_2)

                    histogram_channel.send(x=iter, y=weight)

        # 4.step custom infer process
        def infer_callback(data_source, dump_dir):
            # data_source: data generator
            # dump_dir: your training intermidiate data folder
            # 4.1 step load your custom model
            ...
            # 4.2 step traverse data and do forward process
            for data in data_source.iterator_value():
                # forward process
                ...
                # record result
                ctx.recorder.record(result)

        # 5.step bind training_callback and infer_callback
        ctx.training_process = training_callback
        ctx.infer_process = infer_callback

    (2) call antgo cli at terminal
        antgo run --main_file=challenge_task.py --task=yourtask.xml
        # --task=yourtask.xml config your challenge task


2. Run Challenge Task::

    (1) build running main file (eg. challenge_task.py)
        from antgo.context import *
        # 1.step ctx take control interaction with antgo
        ctx = Context()

        # 2.step custom infer process
        def infer_callback(data_source, dump_dir):
            # data_source: data generator
            # dump_dir : your training intermidiate data folder

            # 2.1 step load custom model
            ...
            # 2.2 step traverse data and do forward process
            for data in data_source.iterator_value():
                # forward process
                ...
                # record result
                ctx.recorder.record(result)

        # bind infer_callback
        ctx.infer_process = infer_callback
    (2) call antgo cli at terminal
    antgo challenge --main_file=challenge_task.py --task=yourtask.xml
    # --task=yourtask.xml config your challenge task


3. Custom Explore Task::

    (1) like 'Train' or 'Challenge' task, build running main file
        ...
    (2) build workflow configure file (.yaml)
        Bootstrap:
         name: 'DataSplit'
         dataset:
          name: 'portrait'
          train_or_test: 'train'
         method: 'bootstrap'
         params:
          bootstrap_counts: 2
         feedback-bind:
         - 'InferenceB'

        TrainingA:
         name: 'Training'
         cpu:
         - 1
         occupy: 'no share'
         dataset:
          name: 'portrait'
         model:
          hello: 'world'
         continue:
          key: 'iter_at'
          value: 10
          condition: 'mod'
         input-bind:
         - 'Bootstrap'

        InferenceB:
         name: 'Inference'
         cpu:
         - 2
         occupy: 'no share'
         input-bind:
         - 'TrainingA'

        EvaluationC:
         name: 'Evaluating'
         task:
          type: 'SEGMENTATION'
          class_label: [1]
         measure:
         - 'PixelAccuracy'
         - 'MeanAccuracy'
         input-bind:
         - 'InferenceB'

        ** implement bootstrap statistic evaluation process at training procedure
    (2) call antgo cli at terminal
    antgo compose --main_file=....py --main_params=...yaml
