# -*- coding: UTF-8 -*-
# @Time : 2018/8/24
# @File : coco2017.py
# @Author: Jian <jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import copy
import itertools
import json
import time
import numpy as np
from collections import defaultdict
import sys

PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
  from urllib import urlretrieve
elif PYTHON_VERSION == 3:
  from urllib.request import urlretrieve
from antgo.dataflow.dataset.dataset import *
from antgo.utils import mask as maskUtils
from antgo.utils.fs import download
from antgo.utils.fs import maybe_here_match_format
from antgo.utils import logger

__all__ = ['COCO2017']


class CocoAPI():
  def __init__(self, annotation_file=None):
    """
    Constructor of Microsoft COCO helper class for reading and visualizing annotations.
    :param annotation_file (str): location of annotation file
    :param image_folder (str): location to the folder that hosts images.
    :return:
    """
    # load dataset
    self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
    self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
    if not annotation_file == None:
      print('loading annotations into memory...')
      tic = time.time()
      dataset = json.load(open(annotation_file, 'r'))
      assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
      print('Done (t={:0.2f}s)'.format(time.time() - tic))
      self.dataset = dataset
      self.createIndex()

  def createIndex(self):
    # create index
    print('creating index...')
    anns, cats, imgs = {}, {}, {}
    imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
    if 'annotations' in self.dataset:
      for ann in self.dataset['annotations']:
        if 'segments_info' not in ann:
          imgToAnns[ann['image_id']].append(ann)
          anns[ann['id']] = ann
        else:
          imgToAnns[ann['image_id']].append(ann['segments_info'])
          for mm in ann['segments_info']:
            anns[mm['id']] = mm

    if 'images' in self.dataset:
      for img in self.dataset['images']:
        imgs[img['id']] = img

    if 'categories' in self.dataset:
      for cat in self.dataset['categories']:
        cats[cat['id']] = cat

    if 'annotations' in self.dataset and 'categories' in self.dataset:
      for ann in self.dataset['annotations']:
        if 'segments_info' not in ann:
          catToImgs[ann['category_id']].append(ann['image_id'])
        else:
          for mm in ann['segments_info']:
            catToImgs[mm['category_id']] = ann['image_id']

    print('index created!')

    # create class members
    self.anns = anns
    self.imgToAnns = imgToAnns
    self.catToImgs = catToImgs
    self.imgs = imgs
    self.cats = cats

  def info(self):
    """
    Print information about the annotation file.
    :return:
    """
    for key, value in self.dataset['info'].items():
      print('{}: {}'.format(key, value))

  def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
    """
    Get ann ids that satisfy given filter conditions. default skips that filter
    :param imgIds  (int array)     : get anns for given imgs
           catIds  (int array)     : get anns for given cats
           areaRng (float array)   : get anns for given area range (e.g. [0 inf])
           iscrowd (boolean)       : get anns for given crowd label (False or True)
    :return: ids (int array)       : integer array of ann ids
    """
    imgIds = imgIds if type(imgIds) == list else [imgIds]
    catIds = catIds if type(catIds) == list else [catIds]

    if len(imgIds) == len(catIds) == len(areaRng) == 0:
      anns = self.dataset['annotations']
    else:
      if not len(imgIds) == 0:
        lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
        anns = list(itertools.chain.from_iterable(lists))
      else:
        anns = self.dataset['annotations']
      anns = anns if len(catIds) == 0 else [ann for ann in anns if ann['category_id'] in catIds]
      anns = anns if len(areaRng) == 0 else [ann for ann in anns if
                                             ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
    if not iscrowd == None:
      ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
    else:
      ids = [ann['id'] for ann in anns]
    return ids

  def getCatIds(self, catNms=[], supNms=[], catIds=[]):
    """
    filtering parameters. default skips that filter.
    :param catNms (str array)  : get cats for given cat names
    :param supNms (str array)  : get cats for given supercategory names
    :param catIds (int array)  : get cats for given cat ids
    :return: ids (int array)   : integer array of cat ids
    """
    catNms = catNms if type(catNms) == list else [catNms]
    supNms = supNms if type(supNms) == list else [supNms]
    catIds = catIds if type(catIds) == list else [catIds]

    if len(catNms) == len(supNms) == len(catIds) == 0:
      cats = self.dataset['categories']
    else:
      cats = self.dataset['categories']
      cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name'] in catNms]
      cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
      cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id'] in catIds]
    ids = [cat['id'] for cat in cats]
    return ids

  def getImgIds(self, imgIds=[], catIds=[]):
    '''
    Get img ids that satisfy given filter conditions.
    :param imgIds (int array) : get imgs for given ids
    :param catIds (int array) : get imgs with all given cats
    :return: ids (int array)  : integer array of img ids
    '''
    imgIds = imgIds if type(imgIds) == list else [imgIds]
    catIds = catIds if type(catIds) == list else [catIds]

    if len(imgIds) == len(catIds) == 0:
      ids = self.imgs.keys()
    else:
      ids = set(imgIds)
      for i, catId in enumerate(catIds):
        if i == 0 and len(ids) == 0:
          ids = set(self.catToImgs[catId])
        else:
          ids &= set(self.catToImgs[catId])
    return list(ids)

  def loadAnns(self, ids=[]):
    """
    Load anns with the specified ids.
    :param ids (int array)       : integer ids specifying anns
    :return: anns (object array) : loaded ann objects
    """
    if type(ids) == list:
      return [self.anns[id] for id in ids]
    elif type(ids) == int:
      return [self.anns[ids]]

  def loadCats(self, ids=[]):
    """
    Load cats with the specified ids.
    :param ids (int array)       : integer ids specifying cats
    :return: cats (object array) : loaded cat objects
    """
    if type(ids) == list:
      return [self.cats[id] for id in ids]
    elif type(ids) == int:
      return [self.cats[ids]]

  def loadImgs(self, ids=[]):
    """
    Load anns with the specified ids.
    :param ids (int array)       : integer ids specifying img
    :return: imgs (object array) : loaded img objects
    """
    if type(ids) == list:
      return [self.imgs[id] for id in ids]
    elif type(ids) == int:
      return [self.imgs[ids]]

  def loadRes(self, resFile):
    """
    Load result file and return a result api object.
    :param   resFile (str)     : file name of result file
    :return: res (obj)         : result api object
    """
    res = CocoAPI()
    res.dataset['images'] = [img for img in self.dataset['images']]

    print('Loading and preparing results...')
    tic = time.time()
    if type(resFile) == str or type(resFile) == unicode:
      anns = json.load(open(resFile))
    elif type(resFile) == np.ndarray:
      anns = self.loadNumpyAnnotations(resFile)
    else:
      anns = resFile
    assert type(anns) == list, 'results in not an array of objects'
    annsImgIds = [ann['image_id'] for ann in anns]
    assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
      'Results do not correspond to current coco set'
    if 'caption' in anns[0]:
      imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
      res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
      for id, ann in enumerate(anns):
        ann['id'] = id + 1
    elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
      res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
      for id, ann in enumerate(anns):
        bb = ann['bbox']
        x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
        if not 'segmentation' in ann:
          ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
        ann['area'] = bb[2] * bb[3]
        ann['id'] = id + 1
        ann['iscrowd'] = 0
    elif 'segmentation' in anns[0]:
      res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
      for id, ann in enumerate(anns):
        # now only support compressed RLE format as segmentation results
        ann['area'] = maskUtils.area(ann['segmentation'])
        if not 'bbox' in ann:
          ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
        ann['id'] = id + 1
        ann['iscrowd'] = 0
    elif 'keypoints' in anns[0]:
      res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
      for id, ann in enumerate(anns):
        s = ann['keypoints']
        x = s[0::3]
        y = s[1::3]
        x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
        ann['area'] = (x1 - x0) * (y1 - y0)
        ann['id'] = id + 1
        ann['bbox'] = [x0, y0, x1 - x0, y1 - y0]
    print('DONE (t={:0.2f}s)'.format(time.time() - tic))

    res.dataset['annotations'] = anns
    res.createIndex()
    return res

  def download(self, tarDir=None, imgIds=[]):
    '''
    Download COCO images from mscoco.org server.
    :param tarDir (str): COCO results directory name
           imgIds (list): images to be downloaded
    :return:
    '''
    if tarDir is None:
      print('Please specify target directory')
      return -1
    if len(imgIds) == 0:
      imgs = self.imgs.values()
    else:
      imgs = self.loadImgs(imgIds)
    N = len(imgs)
    if not os.path.exists(tarDir):
      os.makedirs(tarDir)
    for i, img in enumerate(imgs):
      tic = time.time()
      fname = os.path.join(tarDir, img['file_name'])
      if not os.path.exists(fname):
        urlretrieve(img['coco_url'], fname)
      print('downloaded {}/{} images (t={:0.1f}s)'.format(i, N, time.time() - tic))

  def loadNumpyAnnotations(self, data):
    """
    Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
    :param  data (numpy.ndarray)
    :return: annotations (python nested list)
    """
    print('Converting ndarray to lists...')
    assert (type(data) == np.ndarray)
    print(data.shape)
    assert (data.shape[1] == 7)
    N = data.shape[0]
    ann = []
    for i in range(N):
      if i % 1000000 == 0:
        print('{}/{}'.format(i, N))
      ann += [{
        'image_id': int(data[i, 0]),
        'bbox': [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
        'score': data[i, 5],
        'category_id': int(data[i, 6]),
      }]
    return ann

  def annToRLE(self, ann):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    t = self.imgs[ann['image_id']]
    h, w = t['height'], t['width']
    segm = ann['segmentation']
    if type(segm) == list:
      # polygon -- a single object might consist of multiple parts
      # we merge all parts into one mask rle code
      rles = maskUtils.frPyObjects(segm, h, w)
      rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
      # uncompressed RLE
      rle = maskUtils.frPyObjects(segm, h, w)
    else:
      # rle
      rle = ann['segmentation']
    return rle

  def annToMask(self, ann):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = self.annToRLE(ann)
    m = maskUtils.decode(rle)
    return m


class COCO2017(Dataset):
  def __init__(self, train_or_test, dir=None, ext_params=None):
    '''
    'id': 0, 'supercategory': 'person', 'name': 'person'
    'id': 1, 'supercategory': 'vehicle', 'name': 'bicycle'
    'id': 2, 'supercategory': 'vehicle', 'name': 'car'
    'id': 3, 'supercategory': 'vehicle', 'name': 'motorcycle'
    'id': 4, 'supercategory': 'vehicle', 'name': 'airplane'
    'id': 5, 'supercategory': 'vehicle', 'name': 'bus'
    'id': 6, 'supercategory': 'vehicle', 'name': 'train'
    'id': 7, 'supercategory': 'vehicle', 'name': 'truck'
    'id': 8, 'supercategory': 'vehicle', 'name': 'boat'
    'id': 9, 'supercategory': 'outdoor', 'name': 'traffic light'
    'id': 10, 'supercategory': 'outdoor', 'name': 'fire hydrant'
    'id': 12, 'supercategory': 'outdoor', 'name': 'stop sign'
    'id': 13, 'supercategory': 'outdoor', 'name': 'parking meter'
    'id': 14, 'supercategory': 'outdoor', 'name': 'bench'
    'id': 15, 'supercategory': 'animal', 'name': 'bird'
    'id': 16, 'supercategory': 'animal', 'name': 'cat'
    'id': 17, 'supercategory': 'animal', 'name': 'dog'
    'id': 18, 'supercategory': 'animal', 'name': 'horse'
    'id': 19, 'supercategory': 'animal', 'name': 'sheep'
    'id': 20, 'supercategory': 'animal', 'name': 'cow'
    'id': 21, 'supercategory': 'animal', 'name': 'elephant'
    'id': 22, 'supercategory': 'animal', 'name': 'bear'
    'id': 23, 'supercategory': 'animal', 'name': 'zebra'
    'id': 24, 'supercategory': 'animal', 'name': 'giraffe'
    'id': 26, 'supercategory': 'accessory', 'name': 'backpack'
    'id': 27, 'supercategory': 'accessory', 'name': 'umbrella'
    'id': 30, 'supercategory': 'accessory', 'name': 'handbag'
    'id': 31, 'supercategory': 'accessory', 'name': 'tie'
    'id': 32, 'supercategory': 'accessory', 'name': 'suitcase'
    'id': 33, 'supercategory': 'sports', 'name': 'frisbee'
    'id': 34, 'supercategory': 'sports', 'name': 'skis'
    'id': 35, 'supercategory': 'sports', 'name': 'snowboard'
    'id': 36, 'supercategory': 'sports', 'name': 'sports ball'
    'id': 37, 'supercategory': 'sports', 'name': 'kite'
    'id': 38, 'supercategory': 'sports', 'name': 'baseball bat'
    'id': 39, 'supercategory': 'sports', 'name': 'baseball glove'
    'id': 40, 'supercategory': 'sports', 'name': 'skateboard'
    'id': 41, 'supercategory': 'sports', 'name': 'surfboard'
    'id': 42, 'supercategory': 'sports', 'name': 'tennis racket'
    'id': 43, 'supercategory': 'kitchen', 'name': 'bottle'
    'id': 45, 'supercategory': 'kitchen', 'name': 'wine glass'
    'id': 46, 'supercategory': 'kitchen', 'name': 'cup'
    'id': 47, 'supercategory': 'kitchen', 'name': 'fork'
    'id': 48, 'supercategory': 'kitchen', 'name': 'knife'
    'id': 49, 'supercategory': 'kitchen', 'name': 'spoon'
    'id': 50, 'supercategory': 'kitchen', 'name': 'bowl'
    'id': 51, 'supercategory': 'food', 'name': 'banana'
    'id': 52, 'supercategory': 'food', 'name': 'apple'
    'id': 53, 'supercategory': 'food', 'name': 'sandwich'
    'id': 54, 'supercategory': 'food', 'name': 'orange'
    'id': 55, 'supercategory': 'food', 'name': 'broccoli'
    'id': 56, 'supercategory': 'food', 'name': 'carrot'
    'id': 57, 'supercategory': 'food', 'name': 'hot dog'
    'id': 58, 'supercategory': 'food', 'name': 'pizza'
    'id': 59, 'supercategory': 'food', 'name': 'donut'
    'id': 60, 'supercategory': 'food', 'name': 'cake'
    'id': 61, 'supercategory': 'furniture', 'name': 'chair'
    'id': 62, 'supercategory': 'furniture', 'name': 'couch'
    'id': 63, 'supercategory': 'furniture', 'name': 'potted plant'
    'id': 64, 'supercategory': 'furniture', 'name': 'bed'
    'id': 66, 'supercategory': 'furniture', 'name': 'dining table'
    'id': 69, 'supercategory': 'furniture', 'name': 'toilet'
    'id': 71, 'supercategory': 'electronic', 'name': 'tv'
    'id': 72, 'supercategory': 'electronic', 'name': 'laptop'
    'id': 73, 'supercategory': 'electronic', 'name': 'mouse'
    'id': 74, 'supercategory': 'electronic', 'name': 'remote'
    'id': 75, 'supercategory': 'electronic', 'name': 'keyboard'
    'id': 76, 'supercategory': 'electronic', 'name': 'cell phone'
    'id': 77, 'supercategory': 'appliance', 'name': 'microwave'
    'id': 78, 'supercategory': 'appliance', 'name': 'oven'
    'id': 79, 'supercategory': 'appliance', 'name': 'toaster'
    'id': 80, 'supercategory': 'appliance', 'name': 'sink'
    'id': 81, 'supercategory': 'appliance', 'name': 'refrigerator'
    'id': 83, 'supercategory': 'indoor', 'name': 'book'
    'id': 84, 'supercategory': 'indoor', 'name': 'clock'
    'id': 85, 'supercategory': 'indoor', 'name': 'vase'
    'id': 86, 'supercategory': 'indoor', 'name': 'scissors'
    'id': 87, 'supercategory': 'indoor', 'name': 'teddy bear'
    'id': 88, 'supercategory': 'indoor', 'name': 'hair drier'
    'id': 89, 'supercategory': 'indoor', 'name': 'toothbrush'

    :param train_or_test: train, test or val
    :param dir:
    :param ext_params: year, annotation_type, included, excluded, transform

    '''
    super(COCO2017, self).__init__(train_or_test, dir, ext_params)
    self.year = "2017"
    self.dir = dir
    self.train_or_test = train_or_test
    self.data_type = None
    self.is_support_random_split = False
    self.task_type = getattr(self, 'task_type', None)
    self.task_type_subset = getattr(self, 'task_type_subset', 'stuff')
    self.task_test = getattr(self, 'task_test', None)

    assert(self.train_or_test in ['sample', 'train', 'val', 'test'])
    assert (self.task_type in ['SEGMENTATION', 'OBJECT-DETECTION', 'INSTANCE-SEGMENTATION', 'LANDMARK'])

    # read sample data
    if self.train_or_test == 'sample':
      self.data_samples, self._image_index = self.load_samples()
      return

    data_type = None
    if self.train_or_test == "train":
      data_type = "train" + self.year
    elif self.train_or_test == "val":
      data_type = "val" + self.year
    elif self.train_or_test == "test":
      data_type = "test" + self.year
    self.data_type = data_type

    if self.train_or_test in ['train', 'val']:
      if self.task_type in ["OBJECT-DETECTION", "INSTANCE-SEGMENTATION"]:
        # annotation file
        ann_file = self.config_ann_file(data_type, self.dir, "Instance")
        self.coco_api = CocoAPI(ann_file)

        # parse (for object detector)
        self.cats = self.coco_api.loadCats(self.coco_api.getCatIds())
        self.cat_ids = self.coco_api.getCatIds(self.cats)
        self.img_ids = self.coco_api.getImgIds(self.cat_ids)
      elif self.task_type == "LANDMARK":
        # parse (for object detector)
        ann_file = self.config_ann_file(data_type, self.dir, "Instance")
        self.coco_api = CocoAPI(ann_file)

        # parse (for person task)
        self.cat_ids = self.coco_api.getCatIds(catNms=['person'])
        self.img_ids = self.coco_api.getImgIds(catIds=self.cat_ids)

        # annotation file
        kps_ann_file = self.config_ann_file(data_type, self.dir, 'PersonKeypoint')
        self.coco_kps_api = CocoAPI(kps_ann_file)
      elif self.task_type == "IMAGE_CAPTION":
        # parse (for object detector)
        ann_file = self.config_ann_file(data_type, self.dir, "Instance")
        self.coco_api = CocoAPI(ann_file)

        self.cats = self.coco_api.loadCats(self.coco_api.getCatIds())
        self.cat_ids = self.coco_api.getCatIds(self.cats)
        self.img_ids = self.coco_api.getImgIds(self.cat_ids)

        # parse (for caption)
        caps_ann_file = self.config_ann_file(data_type, self.dir, 'Caption')
        self.coco_caps_api = CocoAPI(caps_ann_file)
      elif self.task_type == "SEGMENTATION":
        # Panoptic Segmentation
        assert(self.task_type_subset in ['stuff', 'panoptic'])
        if self.task_type_subset == 'stuff':
          ann_file = os.path.join(self.dir, 'annotations', 'stuff_%s2017.json'%self.train_or_test)
          self.coco_api = CocoAPI(ann_file)

          self.cats = self.coco_api.loadCats(self.coco_api.getCatIds())
          self.cat_ids = self.coco_api.getCatIds(self.cats)
          self.img_ids = self.coco_api.getImgIds(self.cat_ids)
        else:
          # ann_file = os.path.join(self.dir, 'annotations', 'panoptic_%s2017.json'%self.train_or_test)
          # self.coco_api = CocoAPI(ann_file)
          # self.img_ids = self.coco_api.getImgIds()
          logger.error('not support')
          exit(-1)
      else:
        # annotation file
        ann_file = self.config_ann_file(data_type, self.dir, "Instance")
        self.coco_api = CocoAPI(ann_file)

        # parse (for object detector)
        self.cats = self.coco_api.loadCats(self.coco_api.getCatIds())
        self.cat_ids = self.coco_api.getCatIds(self.cats)
        self.img_ids = self.coco_api.getImgIds(self.cat_ids)
    else:
      ann_file = ''

      if self.task_test == 'dev':
        ann_file = os.path.join(self.dir, 'annotations', 'image_info_test-dev2017.json')
      else:
        ann_file = os.path.join(self.dir, 'annotations', 'image_info_test2017.json')

      self.coco_api = CocoAPI(ann_file)

      # parse (for object detector)
      self.cats = self.coco_api.loadCats(self.coco_api.getCatIds())
      self.cat_ids = self.coco_api.getCatIds(self.cats)
      self.img_ids = self.coco_api.getImgIds(self.cat_ids)

  def config_ann_file(self, data_type, data_dir, annotation_type):
    maybe_data_dir = maybe_here_match_format(data_dir, 'annotations')
    assert maybe_data_dir is not None

    ann_file = None
    if annotation_type == "Instance":
      ann_file = '%s/annotations/instances_%s.json' % (maybe_data_dir, data_type)
    elif annotation_type == "PersonKeypoint":
      ann_file = '%s/annotations/person_keypoints_%s.json' % (maybe_data_dir, data_type)
    elif annotation_type == "Caption":
      ann_file = '%s/annotations/captions_%s.json' % (maybe_data_dir, data_type)

    return ann_file

  def data_pool(self):
    if self.train_or_test == 'sample':
      sample_idxs = copy.copy(self.ids)
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

      idxs = copy.deepcopy(self.img_ids)
      if self.rng:
        self.rng.shuffle(idxs)

      for img_id in idxs:
        img_obj = self.coco_api.loadImgs(img_id)[0]
        if self.train_or_test == 'test':
          img = imread(os.path.join(self.dir, '%s2017' % self.train_or_test, img_obj['file_name']))
          yield [img, {}]
          continue

        if self.task_type == 'OBJECT-DETECTION' or \
                self.task_type=='INSTANCE-SEGMENTATION':
          annotation_ids = self.coco_api.getAnnIds(imgIds=img_obj['id'])
          annotation = self.coco_api.loadAnns(annotation_ids)
          img_annotation = {}

          num_objs = len(annotation)
          boxes = np.zeros((num_objs, 4), dtype=np.uint16)
          category_id = np.zeros((num_objs), dtype=np.int32)
          category = []
          supercategory = []
          area = np.zeros((num_objs), dtype=np.float32)
          segmentation = []

          category_id_name = {cc['id']: cc['name'] for cc in self.coco_api.dataset['categories']}
          category_id_supername = {cc['id']: cc['supercategory'] for cc in self.coco_api.dataset['categories']}

          for ix, obj in enumerate(annotation):
            x, y, w, h = obj['bbox']
            boxes[ix, :] = [x, y, x + w, y + h]

            category_id[ix] = obj['category_id']
            category.append(category_id_name[obj['category_id']])
            supercategory.append(category_id_supername[obj['category_id']])
            area[ix] = obj['area']
            segmentation.append(self.coco_api.annToMask(obj))

          img_annotation['bbox'] = boxes
          img_annotation['category_id'] = category_id
          img_annotation['category'] = category
          img_annotation['supercategory'] = supercategory
          img_annotation['flipped'] = False
          img_annotation['area'] = area
          img_annotation['segmentation'] = segmentation

          img = imread(os.path.join(self.dir, '%s2017'%self.train_or_test, img_obj['file_name']))
          segmentation_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
          for ix, obj_seg in enumerate(segmentation):
            obj_id = category_id[ix]
            segmentation_map[np.where(obj_seg == 1)] = obj_id

          img_annotation.update({'segmentation_map': segmentation_map, 'file_id': img_obj['file_name']})
          img_annotation['info'] = (img.shape[0], img.shape[1], img.shape[2])
          yield [img, img_annotation]
        elif self.task_type == 'IMAGE_CAPTION':
          img = imread(os.path.join(self.dir, '%s2017'%self.train_or_test, img_obj['file_name']))
          ann_ids = self.coco_caps_api.getAnnIds(imgIds=img_obj['id'])
          anns = self.coco_caps_api.loadAnns(ann_ids)
          img_annotation = {'caption': [cap['caption'] for cap in anns], 'file_id': img_obj['file_name']}
          yield [img, img_annotation]
        elif self.task_type == 'SEGMENTATION':
          # stuff, Panoptic
          assert(self.task_type_subset in ['stuff', 'panoptic'])
          if self.task_type_subset == 'stuff':
            img = imread(os.path.join(self.dir, '%s2017'%self.train_or_test, img_obj['file_name']))
            annotation_ids = self.coco_api.getAnnIds(imgIds=img_obj['id'])
            annotation = self.coco_api.loadAnns(annotation_ids)
            img_annotation = {}
            supercategory = []
            num_objs = len(annotation)
            category_id = np.zeros((num_objs), dtype=np.int32)
            category = []
            category_id_name = {cc['id']: cc['name'] for cc in self.coco_api.dataset['categories']}
            category_id_supername = {cc['id']: cc['supercategory'] for cc in self.coco_api.dataset['categories']}

            segmentation = []
            for ix, obj in enumerate(annotation):
              category_id[ix] = obj['category_id']
              category.append(category_id_name[obj['category_id']])
              supercategory.append(category_id_supername[obj['category_id']])
              segmentation.append(self.coco_api.annToMask(obj))

            segmentation_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
            for ix, obj_seg in enumerate(segmentation):
              obj_id = category_id[ix]
              segmentation_map[np.where(obj_seg == 1)] = obj_id

            img_annotation['category_id'] = category_id
            img_annotation['category'] = category
            img_annotation['supercategory'] = supercategory
            img_annotation['flipped'] = False
            img_annotation['segmentation'] = segmentation
            img_annotation['segmentation_map'] = segmentation_map
            img_annotation['file_id'] = img_obj['file_name']

            yield [img, img_annotation]
          else:
            yield [None, None]
        elif self.task_type == 'LANDMARK':
          img = imread(os.path.join(self.dir, '%s2017'%self.train_or_test, img_obj['file_name']))
          ann_ids = self.coco_kps_api.getAnnIds(imgIds=img_obj['id'])
          anns = self.coco_kps_api.loadAnns(ann_ids)

          img_annotation = {'keypoints': [], 'segmentation': []}
          person_bboxs = np.zeros((len(anns), 4))
          area = np.zeros((len(anns)))
          for person_index, person_ann in enumerate(anns):
            keypionts = person_ann['keypoints']
            segmentation = person_ann['segmentation']
            area[person_index] = person_ann['area']

            xx = keypionts[0::3]
            yy = keypionts[1::3]
            visible = keypionts[2::3]
            obj_kk = np.zeros((len(xx),3))
            obj_kk[:, 0] = xx
            obj_kk[:, 1] = yy
            obj_kk[:, 2] = visible
            img_annotation['keypoints'].append(obj_kk)
            img_annotation['segmentation'].append(segmentation)
            person_bboxs[person_index, :] = person_ann['bbox']

          img_annotation['bbox'] = person_bboxs
          img_annotation['area'] = area
          img_annotation['flipped'] = False
          img_annotation['file_id'] = img_obj['file_name']

          yield [img, img_annotation]
        else:
          img = imread(os.path.join(self.dir, '%s2017' % self.train_or_test, img_obj['file_name']))
          yield [img, None]

  @property
  def size(self):
    return len(self.img_ids)

  def split(self, train_validation_ratio=0.0, is_stratified_sampling=True):
    assert(self.train_or_test == 'train')
    validation_coco = COCO2017('val', self.dir, self.ext_params)
    return self, validation_coco