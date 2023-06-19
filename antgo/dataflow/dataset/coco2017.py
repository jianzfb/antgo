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
from urllib.request import urlretrieve
from antgo.dataflow.dataset.dataset import *
from antgo.utils import mask as maskUtils
from antgo.utils.fs import maybe_here_match_format
from antgo.framework.helper.fileio.file_client import *
from filelock import FileLock


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
          ids |= set(self.catToImgs[catId])
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
  METAINFO = {
        'classes':
        ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
         (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
         (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
         (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
         (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
         (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
         (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
         (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
         (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
         (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
         (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
         (246, 0, 122), (191, 162, 208)]
  }

  def __init__(self, train_or_test, dir=None, ext_params=None):
    super(COCO2017, self).__init__(train_or_test, dir, ext_params)
    self.year = "2017"
    self.dir = dir
    self.train_or_test = train_or_test
    self.data_type = None
    self.is_support_random_split = False
    self.task_type = getattr(self, 'task_type', None)
    self.task_type_subset = getattr(self, 'task_type_subset', 'stuff')
    self.task_test = getattr(self, 'task_test', None)

    assert(self.train_or_test in ['train', 'val', 'test'])
    assert (self.task_type in ['SEGMENTATION', 'OBJECT-DETECTION', 'INSTANCE-SEGMENTATION', 'LANDMARK'])

    if not os.path.exists(os.path.join(self.dir , 'annotations')):
      lock = FileLock('DATASET.lock')
      with lock:
        if not os.path.exists(os.path.join(self.dir, 'COCO')):
          # 数据集不存在，需要重新下载，并创建标记
          ali = AliBackend()
          ali.download('ali:///dataset/coco/COCO.tar', self.dir)
          assert(os.path.exists(os.path.join(self.dir, 'COCO.tar')))
          # 解压
          os.system(f'cd {self.dir} && tar -xf COCO.tar')

      # 修改数据目录
      self.dir = os.path.join(self.dir, 'COCO')

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
        # self.cats = self.coco_api.loadCats(self.coco_api.getCatIds())
        self.cat_ids = self.coco_api.getCatIds(catNms=list(self.METAINFO['classes']))
        self.img_ids = self.coco_api.getImgIds(catIds=self.cat_ids)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
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
          ann_file = os.path.join(self.dir, 'annotations', 'panoptic_%s2017.json'%self.train_or_test)
          self.coco_api = CocoAPI(ann_file)
          self.img_ids = self.coco_api.getImgIds()
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

  def at(self, id):
    img_obj = self.coco_api.loadImgs(self.img_ids[id])[0]
    if self.train_or_test == 'test':
      # 对于测试集没有标签
      img = imread(os.path.join(self.dir, '%s2017' % self.train_or_test, img_obj['file_name']))
      return (img, {})

    if self.task_type == 'OBJECT-DETECTION' or \
            self.task_type=='INSTANCE-SEGMENTATION':
      annotation_ids = self.coco_api.getAnnIds(imgIds=img_obj['id'])
      annotation = self.coco_api.loadAnns(annotation_ids)
      img_annotation = {}

      boxes = []
      category_id = []
      for ix, obj in enumerate(annotation):
        x, y, w, h = obj['bbox']        
        inter_w = max(0, min(x + w, img_obj['width']) - max(x, 0))
        inter_h = max(0, min(y + h, img_obj['height']) - max(y, 0))
        if inter_w * inter_h == 0:
                continue   
        if obj['area'] <= 0 or w < 1 or h < 1:
            continue
        if obj['category_id'] not in self.cat_ids:
            continue
        if obj['iscrowd'] == 1:
          # 去除群体标注
          continue

        # 目标框
        boxes.append([x, y, x + w, y + h])
        # 目标类别
        category_id.append(self.cat2label[obj['category_id']])
        # 忽略目标分割

      img_annotation['bboxes'] = np.array(boxes)
      img_annotation['labels'] = np.array(category_id)
      img = imread(os.path.join(self.dir, '%s2017'%self.train_or_test, img_obj['file_name']))
      img_annotation['image_meta'] = {
        'image_shape': (img.shape[0], img.shape[1]),
        'image_file': os.path.join(self.dir, '%s2017'%self.train_or_test, img_obj['file_name'])
      }

      return (img, img_annotation)
    elif self.task_type == 'SEGMENTATION':
      # stuff, Panoptic
      img = imread(os.path.join(self.dir, '%s2017'%self.train_or_test, img_obj['file_name']))
      annotation_ids = self.coco_api.getAnnIds(imgIds=img_obj['id'])
      annotation = self.coco_api.loadAnns(annotation_ids)
      category_id = np.zeros((len(annotation)), dtype=np.int32)
      segmentation = []
      for ix, obj in enumerate(annotation):
        category_id[ix] = self.cat2label(obj['category_id'])
        segmentation.append(self.coco_api.annToMask(obj))

      segmentation_map = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
      for ix, obj_seg in enumerate(segmentation):
        obj_id = category_id[ix]
        segmentation_map[np.where(obj_seg == 1)] = obj_id

      img_annotation = {
        'segments': segmentation_map,
        'image_meta': {
          'image_shape': (img.shape[0], img.shape[1]),
          'image_file': os.path.join(self.dir, '%s2017'%self.train_or_test, img_obj['file_name'])
        }
      }
      return (img, img_annotation)
    elif self.task_type == 'LANDMARK':
      img = imread(os.path.join(self.dir, '%s2017'%self.train_or_test, img_obj['file_name']))
      ann_ids = self.coco_kps_api.getAnnIds(imgIds=img_obj['id'])
      anns = self.coco_kps_api.loadAnns(ann_ids)

      boxes = []
      labels = []
      joints2d = []
      joints_vis = []

      for person_ann in anns:
        keypionts = person_ann['keypoints']        
        person_keypoints_xy = np.zeros((len(keypionts)//3, 2))
        person_keypoints_xy[:, 0] = keypionts[0::3]
        person_keypoints_xy[:, 1] = keypionts[1::3]

        keypoints_visible = np.array(keypionts[2::3])
        position_visible = np.where(keypoints_visible>0)
        if position_visible[0].size == 0:
          continue        
        keypoints_visible[position_visible] = 1
        person_bbox = [
          np.min(person_keypoints_xy[position_visible, 0]), 
          np.min(person_keypoints_xy[position_visible, 1]), 
          np.max(person_keypoints_xy[position_visible, 0]), 
          np.max(person_keypoints_xy[position_visible, 1])
        ]
        if person_bbox[2] - person_bbox[0] <= 5 or person_bbox[3] - person_bbox[1] <= 5:
          continue

        joints2d.append(person_keypoints_xy)
        joints_vis.append(keypoints_visible)
        boxes.append(person_bbox)    
        labels.append(0)  # person

      # 空，跳过
      if len(joints2d) == 0:
        return (None, None)

      img_annotation = {
        'joints2d': np.stack(joints2d, 0),
        'joints_vis': np.stack(joints_vis, 0),
        'bboxes': np.array(boxes),
        'labels': np.array(labels),
        'image_meta': {
          'image_shape': (img.shape[0], img.shape[1]),
          'image_file': os.path.join(self.dir, '%s2017'%self.train_or_test, img_obj['file_name'])
        }
      }
      return (img, img_annotation)

    return (None, None)

  @property
  def size(self):
    return len(self.img_ids)

  def split(self, train_validation_ratio=0.0, is_stratified_sampling=True):
    assert(self.train_or_test == 'train')
    validation_coco = COCO2017('val', self.dir, self.ext_params)
    return self, validation_coco

# coco2017 = COCO2017('train', '/root/workspace/dataset/COCO', ext_params={'task_type': 'OBJECT-DETECTION'})
# label_max = 0
# for i in range(coco2017.size):
#   data = coco2017.sample(i)
#   # if data['labels'].size > 0:
#   #   label_max = max(label_max, np.max(data['labels']))
#   #   print(f'label_max {label_max}')
#   print(i)

#   image = data['image']
#   for bi in range(len(data['bboxes'])):
#     x0,y0,x1,y1 = data['bboxes'][bi]
#     cls_label = data['labels'][bi]
#     x0=(int)(x0)
#     y0=(int)(y0)
#     x1=(int)(x1)
#     y1=(int)(y1)
    
#     color_v = coco2017.METAINFO['palette'][cls_label]
#     image = cv2.rectangle(image, (x0,y0),(x1,y1), color_v, 4)
#   cv2.imwrite("./crop_show.png", image)