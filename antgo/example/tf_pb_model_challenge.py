# -*- coding: UTF-8 -*-
# @Time    : 17-12-14
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.context import *
import tensorflow as tf
from antgo.utils import logger
from antgo.utils._resize import *
from antgo.utils.encode import *
#import cv2

##################################################
######## 1.step global interaction handle ########
##################################################
ctx = Context()


###################################################
######## 2.step define infer process     ##########
###################################################
def infer_callback(data_source, dump_dir):
  # We load the protobuf file from the disk and parse it to retrieve the
  # unserialized graph_def
  # 2.1.step parse tensorflow .pb
  pb_file = None
  if ctx.from_experiment is not None:
    pb_file = os.path.join(ctx.from_experiment, 'optimized_model.pb')
  else:
    pb_file = ctx.params.pb
  if pb_file is None:
    logger.error('must set tensorflow model (*.pb)')
    return
  
  with tf.gfile.GFile(pb_file, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  # Then, we can use again a convenient built-in function to import a graph_def into the
  # current default Graph
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def,
                        input_map=None,
                        return_elements=None,
                        name="prefix",
                        op_dict=None,
                        producer_op_list=None)
  
    # 2.2.step access input and output nodes
    # We access the input and output nodes
    input_nodes = ctx.params.input_nodes
    output_nodes = ctx.params.output_nodes
    
    tensor_input_nodes = []
    for input_node in input_nodes:
      tensor_input_nodes.append(graph.get_tensor_by_name('prefix/%s:0'%input_node))
    
    tensor_output_nodes = []
    for output_node in output_nodes:
      tensor_output_nodes.append(graph.get_tensor_by_name('prefix/%s:0'%output_node))

    # 2.3.step session
    with tf.Session(graph=graph) as sess:
      count = 0
      for data in data_source.iterator_value():
        if ctx.params.preprocess['type'] == 'MEAN-VAR':
          data, = data
          height, width = data.shape[0:2]
         
          data = data[:, :, 0:3]
          original_data = data.copy()

          ##########################################
          #####  preprocess data         ###########
          ##########################################
          if 'resize' in ctx.params.preprocess:
             data = resize(data, (ctx.params.preprocess['resize'][0], ctx.params.preprocess['resize'][1]))
          
          data = np.expand_dims(data, 0)
          data = data.astype(np.float32)
          data = data - np.reshape([ctx.params.preprocess['R_MEAN'],
                                     ctx.params.preprocess['G_MEAN'],
                                    ctx.params.preprocess['B_MEAN']], (1, 1, 1, 3))
          data = data / ctx.params.preprocess['VAR']
          
        if ctx.params.task_type == 'SEGMENTATION':
            start_time = time.time()
            y_val, = sess.run(tensor_output_nodes, feed_dict={tensor_input_nodes[0]: data})
            elapsed_time = (time.time() - start_time)
            logger.info('elapsed time %f'%elapsed_time)
            y_val = np.squeeze(y_val, 0)
            y_val = resize(y_val, (height, width))
            
            ddd = y_val.copy()
            channel_1 = np.exp(ddd[:,:,0])/(np.exp(ddd[:,:,0])+np.exp(ddd[:,:,1]))
            channel_2 = np.exp(ddd[:,:,1])/(np.exp(ddd[:,:,0])+np.exp(ddd[:,:,1]))
            yyy = channel_2
            y_val_max = np.max(yyy)
            y_val_min = np.min(yyy)
            fscore_data = ((yyy - y_val_min)/(y_val_max - y_val_min)) * 255
            fscore_data = fscore_data.astype(np.uint8)
            with open(os.path.join(dump_dir, 'fscore-%d.png'%count), 'wb') as fp:
                png_str = png_encode(fscore_data)
                fp.write(png_str)
            
            mask = np.argmax(y_val, 2)
            mask = mask.astype(np.uint8)
            # original rgb image
            with open(os.path.join(dump_dir, 'image-%d.png'%count), 'wb') as fp:
                png_str = png_encode(original_data)
                fp.write(png_str)
            
            # mask image
            with open(os.path.join(dump_dir, 'mask-%d.png'%count), 'wb') as fp:
                png_str = png_encode(mask*255)
                fp.write(png_str)
            
            # frontground rgb image
            front_img = original_data * np.expand_dims(mask, 2)
            front_img = front_img.astype(np.uint8)
            with open(os.path.join(dump_dir, 'frontground-%d.png'%count), 'wb') as fp:
                png_str = png_encode(front_img)
                fp.write(png_str)
            
            # background rgb image
            background_img = original_data * np.expand_dims(1-mask, 2)
            background_img = background_img.astype(np.uint8)
            with open(os.path.join(dump_dir, 'background-%d.png'%count), 'wb') as fp:
                png_str = png_encode(background_img)
                fp.write(png_str)
            
            ##########################################
            ####  record record         ##############
            ##########################################
            # record result
            ctx.recorder.record(mask)
    
        count += 1
      
###################################################
####### 3.step link training and infer ############
#######        process to context      ############
###################################################
ctx.infer_process = infer_callback
