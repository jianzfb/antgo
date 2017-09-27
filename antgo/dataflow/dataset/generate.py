# encoding=utf-8
# Time: 8/15/17
# File: generate.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import plyvel
import scipy.misc
from antgo.dataflow.basic import *
from antgo.dataflow.dataset.pascal_voc import *


def generate_standard_dataset(data_label_generator, train_or_test, data_folder, dataset_name, **kwargs):
  # build db
  if not os.path.exists(os.path.join(data_folder, dataset_name, train_or_test)):
    os.makedirs(os.path.join(data_folder, dataset_name, train_or_test))
  dataset_record = RecordWriter(os.path.join(data_folder, dataset_name, train_or_test))

  # write data and label
  for data, label in data_label_generator:
    dataset_record.write(Sample(data=data, label=label))

  # bind attributes
  if len(kwargs) > 0:
    dataset_record.bind_attrs(**kwargs)

  # close dataset
  dataset_record.close()


def generate_voc2007_standard_dataset(data_folder, target_folder):
  # train dataset
  pascal_train_2007 = Pascal2007('train', data_folder)
  generate_standard_dataset(pascal_train_2007.iterator_value(), 'train', target_folder, 'voc2007')

  # val dataset
  pascal_val_2007 = Pascal2007('val', data_folder)
  generate_standard_dataset(pascal_val_2007.iterator_value(), 'val', target_folder, 'voc2007')
  

def generate_jiajiaya_standard_dataset(data_folder, target_folder):
  # train dataset
  train_list = os.listdir(os.path.join(data_folder, 'training'))
  train_img_list = []
  train_annotation_list = []
  for f in train_list:
    if '_matte' in f:
      train_annotation_list.append(os.path.join(data_folder, 'training', f))
      train_img_list.append(os.path.join(data_folder, 'training', f.replace('_matte','')))
  
  def build_training_data_generator(img_list, annotation_list):
    for img_f, annotation_f in zip(img_list, annotation_list):
      data = scipy.misc.imread(img_f)
      matting = scipy.misc.imread(annotation_f)
      label = matting.copy()
      pos = np.where(label > 128)
      label[:, :] = 0
      label[pos] = 1
      
      matting = matting / 255.0
      matting = matting.astype(np.float32)

      yield data, (label, matting)
  
  generate_standard_dataset(build_training_data_generator(train_img_list, train_annotation_list),
                            'train',
                            target_folder,
                            'portrait')
    
  # test dataset
  test_list = os.listdir(os.path.join(data_folder, 'testing'))
  test_img_list = []
  test_annotation_list = []
  for f in test_list:
    if '_matte' in f:
      test_annotation_list.append(os.path.join(data_folder, 'testing', f))
      test_img_list.append(os.path.join(data_folder, 'testing', f.replace('_matte', '')))

  def build_testing_data_generator(img_list, annotation_list):
    for img_f, annotation_f in zip(img_list, annotation_list):
      data = scipy.misc.imread(img_f)
      matting = scipy.misc.imread(annotation_f)
      label = matting.copy()
      pos = np.where(label > 128)
      label[:, :] = 0
      label[pos] = 1

      yield data, label

  generate_standard_dataset(build_testing_data_generator(test_img_list, test_annotation_list),
                            'test',
                            target_folder,
                            'portrait')


def generate_coco_person_train_standard_dataset(data_folder, target_folder):
  # train dataset
  train_list = os.listdir(os.path.join(data_folder, ''))
  train_img_list = []
  train_annotation_list = []
  for f in train_list:
    if '_matte' in f:
      train_annotation_list.append(os.path.join(data_folder, '', f))
      train_img_list.append(os.path.join(data_folder, '', f.replace('_matte', '')))
  
  def build_training_data_generator(img_list, annotation_list):
    for img_f, annotation_f in zip(img_list, annotation_list):
      data = scipy.misc.imread(img_f)
      label = scipy.misc.imread(annotation_f)
      if len(label.shape) == 3:
        label = label[:,:,0]
      pos = np.where(label > 128)
      label[:, :] = 0
      label[pos] = 1
      
      yield data, label
  
  generate_standard_dataset(build_training_data_generator(train_img_list, train_annotation_list),
                            'train',
                            target_folder,
                            'coco_person')


# if __name__ == '__main__':
#   # transfer voc2007
#   #generate_voc2007_standard_dataset('/home/mi/下载/dataset/voc','/home/mi/antgo/antgo-dataset')
#
#   # transfer jiajiaya
#   # generate_jiajiaya_standard_dataset('/home/mi/dataset/jiajiaya', '/home/mi/antgo/antgo-dataset')
#
#   generate_coco_person_train_standard_dataset('/home/mi/dataset/coco_person/','/home/mi/antgo/antgo-dataset/coco_person')
#   pass
