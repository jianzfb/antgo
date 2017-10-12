# encoding=utf-8
# Time: 8/28/17
# File: linear_regression_example.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.common import *
from antgo.context import *
from antgo.dataflow.dataset.heart import *
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

# 图像数据
image_channel = ctx.job.create_channel("sample", "IMAGE")

# 文本数据
text_channel = ctx.job.create_channel("record", "TEXT")

# 构建图表
# a_channel和b_channel绘制在同一图表中
ctx.job.create_chart([regression_loss,regulation],"loss","step","value")
# 模型权重直方图绘制在图表中
ctx.job.create_chart([histogram_w_channel],"histogram-weight","weight","frequence")
# 样本记录
ctx.job.create_chart([image_channel], "sample")
# 文本记录
ctx.job.create_chart([text_channel], "record")

##################################################
######## 2.step define training process  #########
##################################################
def training_callback(data_source, dump_dir):
  batch_data = BatchData(Node.inputs(data_source), batch_size=64)
  iter = 0
  for data, label in batch_data.iterator_value():
    loss = np.random.random()
    l2_norm = np.random.random()
    weight = np.random.random((200))

    # regression loss
    regression_loss.send(iter, loss)
    # regulation loss
    regulation.send(iter, l2_norm)

    # weight histogram
    histogram_w_channel.send(iter, weight)

    # currunt sample
    image = np.random.random((100,100,3))
    image_channel.send(iter, image)

    text_channel.send(iter, 'loss %f %f'%(loss, l2_norm))

    iter += 1
    print(iter)

  print('stop training process')

###################################################
######## 3.step define infer process     ##########
###################################################
def infer_callback(data_source, dump_dir):
  iter = 0
  for data in data_source.iterator_value():
    iter += 1

    ctx.recorder.record(random.random())
    # time.sleep(2)
    print('iterator %d'%iter)

  print('stop inference process')

###################################################
####### 4.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback
