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
from antgo.ant.utils import *
slim = tf.contrib.slim
from antgo.utils import logger
from antgo.dataflow.dataset.empty_dataset import *
from antgo.ant import flags
from datetime import datetime
from multiprocessing import Process
import yaml
FLAGS = flags.AntFLAGS


def tftool_frozen_graph(ctx,
                        dump_dir,
                        time_stamp,
                        input_node_names,
                        output_node_names):
  # 1.step build inference model
  now_time_stamp = datetime.fromtimestamp(time_stamp).strftime('%Y%m%d.%H%M%S.%f')
  dump_dir = os.path.join(dump_dir, now_time_stamp)
  if not os.path.exists(dump_dir):
    os.makedirs(dump_dir)

  ablation_blocks = getattr(ctx.params, 'ablation', [])
  for b in ablation_blocks:
    ctx.deactivate_block(b)

  # independent process
  # ctx.call_frozen_process(dump_dir)
  logger.info('deploy infer model')
  p = multiprocessing.Process(target=ctx.call_infer_process, args=(None, dump_dir))
  p.start()
  p.join()

  # 2.step check input_node_names and output_node_names
  if input_node_names == '' or output_node_names == '':
    logger.error('must set input_nodes and output_nodes')
    return

  # 3.step check infer_graph.pbtxt in experiment
  if not os.path.exists(os.path.join(dump_dir, 'infer_graph.pbtxt')):
    logger.error('dont exist graph file')
    return

  # 4.step build frozen graph
  # We retrieve our checkpoint fullpath
  checkpoint = tf.train.get_checkpoint_state(dump_dir)
  input_checkpoint = checkpoint.model_checkpoint_path

  # We precise the file fullname of our freezed graph
  input_checkpoint_path = os.path.join(dump_dir, input_checkpoint.split('/')[-1])

  input_graph_path = os.path.join(dump_dir, 'infer_graph.pbtxt')
  output_graph_path = os.path.join(dump_dir, "frozen_model.pb")

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
  with tf.Graph().as_default() as graph:
    input_graph_def = graph.as_graph_def()
    with tf.gfile.FastGFile(output_graph_path,'rb') as f:
      input_graph_def.ParseFromString(f.read())

      _ = tf.import_graph_def(input_graph_def, name="")
      summary_write = tf.summary.FileWriter(dump_dir, graph)


  output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def,
                                                                       input_node_names.split(','),
                                                                       output_node_names.split(','),
                                                                       tf.float32.as_datatype_enum)

  optimized_path = os.path.join(dump_dir, "optimized_model.pb")
  f = tf.gfile.FastGFile(optimized_path, "wb")
  f.write(output_graph_def.SerializeToString())


def tftool_visualize_pb(pb_path):
  if not os.path.exists(pb_path):
    logger.error('pb model file dont exist')
    return

  logger.info('load model pb')
  graph = tf.get_default_graph()
  graphdef = graph.as_graph_def()
  graphdef.ParseFromString(gfile.FastGFile(pb_path, "rb").read())

  _ = tf.import_graph_def(graphdef, name="")

  logger.info('start model FLOPs statistic')
  flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
  logger.info('model FLOPs: {}'.format(flops.total_float_ops))

  logger.info('generate visualization data')
  summary_write = tf.summary.FileWriter("./", graph)
  logger.info('open tensorboard --logdir=.')


def _int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _get_dataset_record_filename(dataset_record_dir, dataset_name, split_name, shard_id, num_shards):
  output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (dataset_name, split_name, shard_id, num_shards)
  return os.path.join(dataset_record_dir, output_filename)


def tftool_generate_image_records(data_dir,
                                  label_dir,
                                  record_dir,
                                  label_flag='_label',
                                  train_or_test='train',
                                  num_shards=20,
                                  dataset_flag='antgo'):
  



  # 1.step search all files in data_dir
  data_files = os.listdir(data_dir)
  is_jpg = False
  is_png = False
  kp_maps = {}
  for f in data_files:
    if f.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
      if not is_jpg and not is_png:
        if f.split('.')[-1].lower() in ['jpg', 'jpeg']:
          is_jpg = True
        else:
          is_png = True
      f_path = os.path.join(data_dir, f)
      kp_maps[f.split('.')[0]] = f_path
  
  label_files = os.listdir(label_dir)
  
  prepare_data_files = []
  prepare_label_files = []
  for f in label_files:
    prefix = f.split('.')[0].replace(label_flag, '')
    if prefix in kp_maps:
      prepare_data_files.append(kp_maps[prefix])
      prepare_label_files.append(os.path.join(label_dir, f))
    else:
      sys.stdout.write('%f couldnt find data - label pair'% f)
      sys.stdout.flush()

  sys.stdout.write('total files %d are splited to %d shards'%(len(prepare_data_files),int(num_shards)))
  sys.stdout.flush()

  # 2.step transform to tfrecord
  num_per_shard = int(np.math.ceil(len(prepare_data_files) / float(num_shards)))
  with tf.Session('') as sess:
    for shard_id in range(num_shards):
      record_filename = _get_dataset_record_filename(record_dir, dataset_flag, train_or_test, shard_id, num_shards)

      with tf.python_io.TFRecordWriter(record_filename) as tfrecord_writer:
        start_ndx = shard_id * num_per_shard
        end_ndx = min((shard_id + 1) * num_per_shard, len(prepare_data_files))

        for i in range(start_ndx, end_ndx):
          a_data = tf.gfile.FastGFile(prepare_data_files[i], 'rb').read()
          b_data = tf.gfile.FastGFile(prepare_label_files[i], 'rb').read()
          
          sys.stdout.write('\r>> Converting image %d/%d shard %d\n' % (i + 1, len(prepare_data_files), shard_id))
          sys.stdout.flush()

          format_str = 'jpg' if is_jpg else 'png'
          example = tf.train.Example(features=
                                      tf.train.Features(feature={
                                                                'image/encoded': _bytes_feature(a_data),
                                                                'image/format': _bytes_feature(format_str.encode('utf-8')),
                                                                'label/encoded': _bytes_feature(b_data),
                                                                'label/format': _bytes_feature(format_str.encode('utf-8'))}))
          tfrecord_writer.write(example.SerializeToString())