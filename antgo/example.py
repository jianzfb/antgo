# encoding=utf-8
# @Time    : 17-8-2
# @File    : example.py
# @Author  :
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.dataflow.basic import *
import cv2


# def generate_potrait_dataset(portrait_dataset, dump_dir):
#   record_writer = RecordWriter(dump_dir)
#   file_names = os.listdir(portrait_dataset)
#   img_files = []
#   annotation_files = []
#   for f in file_names:
#     if '_matte' in f:
#       annotation_files.append(os.path.join(portrait_dataset, f))
#       img_f = f.replace('_matte', '')
#       img_files.append(os.path.join(portrait_dataset, img_f))
#
#   for img_file, anno_file in zip(img_files, annotation_files):
#     image = cv2.imread(img_file)
#     annotation = cv2.imread(anno_file)[:, :, 0]
#     annotation[np.where(annotation < 128)] = 0
#     annotation[np.where(annotation >= 128)] = 1
#
#     record_writer.write(Sample(data=image, label=annotation))
#
#   record_writer.close()
#
#
# def check_potrait_dataset(dump_dir):
#   record_reader = RecordReader(dump_dir)
#
#   for sample in record_reader.iterate_read('data', 'label'):
#     a, b = sample
#     cv2.imshow('dd', a)
#     cv2.imshow('bb', b)
#     cv2.waitKey(0)

# generate_potrait_dataset('/home/mi/dataset/jiajiaya/testing/', '/home/mi/dataset/test')
# check_potrait_dataset('/home/mi/dataset/test')

record_writer = RecordWriter('/home/mi/antgo/antgo-dataset/portrait/test/')
record_writer.bind_attrs(count=300, class_num=2)
record_writer.close()

