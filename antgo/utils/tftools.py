# -*- coding: UTF-8 -*-
# @Time    : 17-12-6
# @File    : tftools.py
# @Author  : Jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.platform import gfile
slim = tf.contrib.slim
from antgo.utils import logger
from antgo.dataflow.dataset.dataset import *


def tftool_frozen_graph(dump_dir,
                        experiment_name,
                        input_node_names,
                        output_node_names):
  # 1.step experiment
  checkpoint_main_folder = os.path.join(dump_dir, experiment_name, 'inference')
  if not os.path.isdir(checkpoint_main_folder):
    logger.error('dont exist experiment')
    return
  
  # 2.step check infer_graph.pbtxt in experiment
  if not os.path.exists(os.path.join(checkpoint_main_folder, 'infer_graph.pbtxt')):
    logger.error('dont exist graph file')
    return
  
  # 3.step check input_node_names and output_node_names
  if input_node_names == '' or output_node_names == '':
    logger.error('must set input_nodes and output_nodes')
    return
  
  # 4.step build frozen graph
  # We retrieve our checkpoint fullpath
  checkpoint = tf.train.get_checkpoint_state(checkpoint_main_folder)
  input_checkpoint = checkpoint.model_checkpoint_path

  # We precise the file fullname of our freezed graph
  input_checkpoint_path = os.path.join(checkpoint_main_folder, input_checkpoint.split('/')[-1])

  input_graph_path = os.path.join(checkpoint_main_folder, 'infer_graph.pbtxt')
  output_graph_path = os.path.join(checkpoint_main_folder, "frozen_model.pb")

  input_saver_path = ""
  input_binary = False

  restore_op_name = 'save/restore_all'
  filename_tensor_name = 'save/Const:0'

  clear_devices = True
  initializer_nodes = ""

  freeze_graph.freeze_graph(input_graph_path,
                            input_saver_path,
                            input_binary,
                            input_checkpoint_path,
                            output_node_names,
                            restore_op_name,
                            filename_tensor_name,
                            output_graph_path,
                            clear_devices, initializer_nodes)
  
  # 5.step build optimized graph
  with tf.Graph().as_default():
    input_graph_def = tf.GraphDef()
    with open(output_graph_path, 'rb') as f:
      input_graph_def.ParseFromString(f.read())

  output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def,
                                                                       input_node_names.split(','),
                                                                       output_node_names.split(','),
                                                                       tf.float32.as_datatype_enum)

  optimized_path = os.path.join(checkpoint_main_folder, "optimized_model.pb")
  f = tf.gfile.FastGFile(optimized_path, "w")
  f.write(output_graph_def.SerializeToString())