# encoding=utf-8
# @Time    : 17-6-13
# @File    : recorder.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import tarfile
import numpy as np
from six.moves import range
import xml.etree.ElementTree as ET

from antgo.utils import logger, get_rng, memoized
from antgo.utils.fs import mkdir_p, download, maybe_here
from antgo.utils.timer import timed_operation
from .dataset import *
__all__ = ['ILSVRC12']


CAFFE_ILSVRC12_URL = "http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz"
class ILSVRCMeta(object):
  """
  Some metadata for ILSVRC dataset.
  """
  def __init__(self, dir=None):
    # todo: not completed
    self.dir = dir

    f = os.path.join(self.dir, 'synsets.txt')
    if not os.path.isfile(f):
        self._download_caffe_meta()

  def get_synset_words_1000(self):
    """
    :returns a dict of {cls_number: cls_name}
    """
    fname = os.path.join(self.dir, 'synset_words.txt')
    assert os.path.isfile(fname)
    lines = [x.strip() for x in open(fname).readlines()]
    return dict(enumerate(lines))

  def get_synset_1000(self):
    """
    :returns a dict of {cls_number: synset_id}
    """
    fname = os.path.join(self.dir, 'synsets.txt')
    assert os.path.isfile(fname)
    lines = [x.strip() for x in open(fname).readlines()]
    inv_synset = dict([ (b, a) for a, b in enumerate(lines)])
    return dict(enumerate(lines)), inv_synset

  def _download_caffe_meta(self):
    fpath = download(CAFFE_ILSVRC12_URL, self.dir)
    tarfile.open(fpath, 'r:gz').extractall(self.dir)


class ILSVRC12(Dataset):
  def __init__(self, train_or_test, dataset_path, params=None):
    """
    :param dataset_path: A directory containing a subdir named `name`, where the
        original ILSVRC12_`name`.tar gets decompressed.
    :param train_or_test: 'train' or 'val' or 'test'

    Dir should have the following structure:

    .. code-block:: none

        dir/
          train/
            n02134418/
              n02134418_198.JPEG
              ...
            ...
          val/
            ILSVRC2012_val_00000001.JPEG
            ...
          test/
            ILSVRC2012_test_00000001.JPEG
            ...
          bbox/
            n02134418/
              n02134418_198.xml
              ...
            ...

    After decompress ILSVRC12_img_train.tar, you can use the following
    command to build the above structure for `train/`:

    .. code-block:: none
        tar xvf ILSVRC12_img_train.tar -C train && cd train
        find -type f -name '*.tar' | parallel -P 10 'echo {} && mkdir -p {/.} && tar xf {} -C {/.}'
        Or:
        for i in *.tar; do dir=${i%.tar}; echo $dir; mkdir -p $dir; tar xf $i -C $dir; done

    """
    # init parent class
    super(ILSVRC12, self).__init__(train_or_test, dataset_path)
    self.train_or_test = train_or_test

    # read sample data
    if self.train_or_test == 'sample':
      self.data_samples, self.ids = self.load_samples()
      return

    # some meta info
    if not os.path.exists(os.path.join(self.dir, 'meta')):
      self.download(os.path.join(self.dir, 'meta'),
                    default_url="http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz",
                    auto_untar=True,
                    is_gz=True)
    
    meta = ILSVRCMeta(os.path.join(self.dir, 'meta'))
    self.synset, self.inv_synset = meta.get_synset_1000()
    
    # image list
    bbdir = os.path.join(self.dir, 'bbox')
    if self.train_or_test == 'train':
      # maybe download
      if not os.path.exists(os.path.join(self.dir, self.train_or_test)):
        self.download(self.dir, file_names=['ILSVRC2012_img_train.tar'],
          default_url="shell:mkdir train && tar xvf {file_placeholder} -C train\n cd train && find -type f -name '*.tar' | parallel -P 10 'echo {} && mkdir -p {/.} && tar xf {} -C {/.}'")
  
      self.imglist = self.get_train_image_list(self.train_or_test)
      self.bblist, _ = ILSVRC12.get_bbox(bbdir, self.imglist, self.train_or_test)
    else:
      # maybe download
      if train_or_test == 'val':
        if not os.path.exists(os.path.join(self.dir, self.train_or_test)):
          self.download(self.dir, file_names=['ILSVRC2012_img_val.tar'],
            default_url="shell:mkdir train && tar xvf {file_placeholder} -C train\n cd train && find -type f -name '*.tar' | parallel -P 10 'echo {} && mkdir -p {/.} && tar xf {} -C {/.}'")
    
      self.imglist = self.get_val_or_test_image_list(self.train_or_test)
      if self.train_or_test == 'val':
        self.bblist, label_list = ILSVRC12.get_bbox(bbdir, self.imglist, self.train_or_test)
        self.imglist = [(self.imglist[k][0], self.inv_synset[label_list[k]]) for k in range(len(self.imglist))]

    self.ids = list(range(len(self.imglist)))

  def get_train_image_list(self, name):
    """
    :param name: 'train'
    :returns: list of (image filename, cls)
    """
    image_list = [('%s/%s'%(class_name, file_name), self.inv_synset[class_name]) \
                   for class_index, class_name in enumerate(os.listdir(os.path.join(self.dir, name)) ) \
                   if os.path.isdir(os.path.join(self.dir, name, class_name)) \
                   for file_name in os.listdir(os.path.join(self.dir, name, class_name))]

    return image_list
  def get_val_or_test_image_list(self, name):
    """
    :param name: 'val'
    :returns: list of (image filename, cls)
    """
    image_list = [(filename,-1) for fileindex,filename in enumerate(os.listdir(os.path.join(self.dir, self.train_or_test, name))) ]
    return image_list

  def split(self, split_params={}, split_method='holdout'):
    assert (self.train_or_test == 'train')
    assert (split_method == 'holdout')
    validation_dataset = ILSVRC12('val', self.dir)
    return self, validation_dataset

  @property
  def size(self):
    return len(self.ids)

  def data_pool(self):
    """
    Produce original images of shape [h, w, 3], and label,
    and optionally a bbox of [xmin, ymin, xmax, ymax]
    """
    if self.train_or_test == 'sample':
      sample_idxs = copy.deepcopy(self.ids)
      if self.rng:
        self.rng.shuffle(sample_idxs)

      for index in sample_idxs:
        yield self.data_samples[index]
      return

    epoch = 0
    while True:
      max_epoches = self.epochs if self.epochs is not None else 1
      if epoch >= max_epoches:
        break
      epoch += 1

      idxs = np.arange(len(self.imglist))
      if self.rng:
          self.rng.shuffle(idxs)

      for k in idxs:
        fname, label = self.imglist[k]
        fname = os.path.join(self.dir, self.train_or_test, fname)
        im = imread(fname.strip())

        if im.ndim == 2:
          im = np.expand_dims(im, 2).repeat(3, 2)

        if self.train_or_test == "train" or \
                self.train_or_test == 'val':
          bbox = self.bblist[k]
          if bbox is None:
              bbox = np.array([0, 0, im.shape[1] - 1, im.shape[0] - 1])

          yield [im, {'category_id': label,
                      'bbox': bbox,
                      'id': k,
                      'info': (im.shape[0], im.shape[1], im.shape[2])}]
        else:
          yield [im]
  
  def at(self, k):
    if self.train_or_test == 'sample':
      return self.data_samples[k]

    fname, label = self.imglist[k]
    fname = os.path.join(self.dir, self.train_or_test, fname)
    im = imread(fname.strip())
  
    if im.ndim == 2:
      im = np.expand_dims(im, 2).repeat(3, 2)
  
    if self.train_or_test == "train" or \
            self.train_or_test == 'val':
      bbox = self.bblist[k]
      if bbox is None:
        bbox = np.array([0, 0, im.shape[1] - 1, im.shape[0] - 1])
    
      return [im, {'category_id': label,
                  'bbox': bbox,
                  'id': k,
                  'info': (im.shape[0], im.shape[1], im.shape[2])}]
    else:
      return [im]

  @staticmethod
  def get_bbox(bbox_dir, imglist, train_or_test):
    ret = []
    ret_label = []
    def parse_bbox(fname):
      root = ET.parse(fname).getroot()
      size = root.find('size').getchildren()
      size = map(int, [size[0].text, size[1].text])

      box = root.find('object').find('bndbox').getchildren()
      box = map(lambda x: float(x.text), box)

      label = root.find('object').find('name').text
      return np.asarray(list(box), dtype='float32'),label

    with timed_operation('Loading Bounding Boxes ...'):
      cnt = 0
      import tqdm
      for k in tqdm.trange(len(imglist)):
        fname = imglist[k][0]
        fname = fname[:-4] + 'xml'
        fname = os.path.join(bbox_dir,train_or_test ,fname)
        try:
          box,label = parse_bbox(fname)
          ret.append(box)
          ret_label.append(label)
          cnt += 1
        except KeyboardInterrupt:
          raise
        except:
          ret.append(None)
          ret_label.append(-1)
      logger.info("{}/{} images have bounding box.".format(cnt, len(imglist)))
    return ret, ret_label
