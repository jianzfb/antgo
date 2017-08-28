# encoding=utf-8
# Time: 8/28/17
# File: linear_regression_example.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.common import *
from antgo.context import *
import numpy as np

##################################################
######## 1.step global interaction handle ########
##################################################
ctx = Context()


##################################################
######## 1.1 step define chart channel ###########
##################################################
# 模型回归代价
regression_loss = ctx.job.create_channel("regression-loss","NUMERIC")
# 正则项数值
regulation = ctx.job.create_channel("regulation","NUMERIC")

# 模型权重分布
histogram_w_channel = ctx.job.create_channel("model-w",'HISTOGRAM')

# 构建图表
# a_channel和b_channel绘制在同一图表中
ctx.job.create_chart([regression_loss,regulation],"loss","step","value")
# 模型权重直方图绘制在图表中
ctx.job.create_chart([histogram_w_channel],"histogram-weight","weight","frequence")


##################################################
######## 2.step define training process  #########
##################################################
def training_callback(data_source, dump_dir):
  batch_data = BatchData(Node.inputs(data_source), batch_size=64)
  for iter in range(100):
    data, label = batch_data.iterator_value()

    loss = np.random.random()
    l2_norm = np.random.random()
    weight = [np.random.random() for _ in range(200)]

    # regression loss
    regression_loss.send(iter, loss)
    # regulation
    regulation.send(iter, l2_norm)

    histogram_w_channel.send(iter, weight)


###################################################
######## 2.step define infer process     ##########
###################################################
def infer_callback(data_source, dump_dir):
  pass


###################################################
####### 5.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback