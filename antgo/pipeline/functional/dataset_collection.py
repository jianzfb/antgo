# -*- coding: UTF-8 -*-
# @Time    : 2022/9/11 23:01
# @File    : dataset_collection.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from .data_collection import DataCollection, DataFrame
from .entity import Entity
from .image import *
from .common import *
from antgo.pipeline.hparam import HyperParameter as State
from antgo.pipeline.hparam import param_scope
from antgo.pipeline.hparam import dynamic_dispatch
from antgo.pipeline.functional.common.config import *
from antgo.dataflow.dataset.base_coco_style_dataset import BaseCocoStyleDataset
from tfrecord.reader import *
from tfrecord import iterator_utils
from tfrecord import example_pb2
from antgo.framework.helper.dataset.builder import DATASETS
from antgo.framework.helper.utils.registry import *
from antgo.dataflow.datasetio import *
import numpy as np
import json
import os
import cv2
import yaml
import random


@dynamic_dispatch
def coco_format_dc(dir, ann_file, data_prefix, mode='detect', normalize=False, is_random=False, is_debug=False):
    coco_style_dataset = BaseCocoStyleDataset(
        dir=dir,
        ann_file=ann_file,
        data_prefix=data_prefix,
        data_mode='bottomup'
    )

    def inner():
        sample_num = len(coco_style_dataset)
        index_list = list(range(sample_num))
        if is_random:
            random.shuffle(index_list)

        for sample_i in index_list:
            sample_info = coco_style_dataset[sample_i]
            if is_debug:
                print(sample_info['img_path'])

            bboxes = sample_info['bboxes']
            if normalize:
                for box_info in bboxes:
                    x0,y0,w,h = box_info
                    box_info[2] = x0 + w
                    box_info[3] = y0 + h

            export_info = {
                'image': sample_info['image'],
                'bboxes': bboxes,
                'labels': sample_info['category_id'],
                'joints2d': sample_info['keypoints'],
                'joints_vis': sample_info['keypoints_visible']
            }

            entity = Entity()(**export_info)
            yield entity

    return DataFrame(inner())


@dynamic_dispatch
def yolo_format_dc(ann_file, mode='detect', stage='train', normalize=False, is_random=False, is_debug=False):
    assert(stage in ['train', 'val', 'test'])
    with open(ann_file, "r", errors="ignore", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    ann_folder = os.path.dirname(ann_file)
    data_folder = data['path']
    image_folder_map = {
        'train': os.path.join(ann_folder, data_folder, data['train']),
        'val': os.path.join(ann_folder, data_folder, data['val'])
    }
    label_folder_map = {
        'train': os.path.join(ann_folder, data_folder, data['train'].replace('images', 'labels')),
        'val': os.path.join(ann_folder, data_folder, data['val'].replace('images', 'labels'))
    }
    file_name_list = os.listdir(image_folder_map[stage])
    file_name_list = [name for name in file_name_list if name[0] != '.']

    if is_random:
        random.shuffle(file_name_list)

    category_map = data["names"]
    def inner():
        sample_num = len(file_name_list)
        for sample_i in range(sample_num):
            file_name = file_name_list[sample_i]
            image_path = f'{image_folder_map[stage]}/{file_name}'
            if is_debug:
                print(image_path)
            p = file_name.rfind('.')
            pure_name = file_name[:p]
            label_path = f'{label_folder_map[stage]}/{pure_name}.txt'

            image = cv2.imread(image_path)
            image_h, image_w = image.shape[:2]
            export_info = {
                'image': image,
                'filename': image_path
            }
            if mode == 'detect':
                with open(label_path, 'r') as fp:
                    content = fp.readline().strip()
                    bboxes = []
                    labels = []
                    while content:
                        class_id, cx,cy,w,h = content.split(' ')
                        cx = float(cx)
                        cy = float(cy)
                        w = float(w)
                        h = float(h)
                        if normalize:
                            x0,y0,x1,y1 = (cx - w/2)*image_w, (cy - h/2)*image_h, (cx + w/2)*image_w, (cy + h/2)*image_h
                            bboxes.append([
                                x0,y0,x1,y1
                            ])
                        else:
                            bboxes.append([
                                cx,cy,w,h
                            ])
                        
                        labels.append(int(class_id))
                        content = fp.readline().strip()

                    export_info['bboxes'] = np.array(bboxes)
                    export_info['labels'] = np.array(labels)
            elif mode == 'pose':
                kpt_shape = data['kpt_shape']
                with open(label_path, 'r') as fp:
                    content = fp.readline().strip()
                    bboxes = []
                    labels = []
                    keypoints = []
                    while content:
                        terms = content.split(' ')
                        class_id, cx,cy,w,h = terms[:5]
                        cx = float(cx)
                        cy = float(cy)
                        w = float(w)
                        h = float(h)

                        bbox_keypoints = [float(v) for v in terms[5:]]
                        bbox_keypoints = np.array(bbox_keypoints, dtype=np.float32).reshape(kpt_shape)
                        if normalize:
                            x0,y0,x1,y1 = (cx - w/2)*image_w, (cy - h/2)*image_h, (cx + w/2)*image_w, (cy + h/2)*image_h
                            bboxes.append([
                                x0,y0,x1,y1
                            ])
                            bbox_keypoints[:,0] = bbox_keypoints[:,0] * image_w
                            bbox_keypoints[:,1] = bbox_keypoints[:,1] * image_h
                        else:
                            bboxes.append([
                                cx,cy,w,h
                            ])
                        
                        labels.append(int(class_id))
                        keypoints.append(bbox_keypoints)
                        content = fp.readline().strip()

                    export_info['bboxes'] = np.array(bboxes, dtype=np.float32)
                    export_info['labels'] = np.array(labels, dtype=np.int32)
                    export_info['keypoints'] = np.stack(keypoints, 0)
            elif mode == 'segment':
                with open(label_path, 'r') as fp:
                    content = fp.readline().strip()
                    labels = []
                    segments = []
                    while content:
                        terms = content.split(' ')
                        class_id = terms[0]

                        polygon_keypoints = [float(v) for v in terms[1:]]
                        polygon_keypoints = np.array(polygon_keypoints, dtype=np.float32).reshape((-1, 2))
                        if normalize:
                            polygon_keypoints[:,0] = polygon_keypoints[:,0] * image_w
                            polygon_keypoints[:,1] = polygon_keypoints[:,1] * image_h

                        polygon_keypoints = polygon_keypoints.astype(np.int64)
                        labels.append(int(class_id))

                        segment = np.zeros((image_h, image_w), dtype=np.uint8)
                        cv2.fillPoly(segment, [polygon_keypoints], 1)
                        segments.append(segment)
                        content = fp.readline().strip()

                    export_info['labels'] = np.array(labels, dtype=np.int32)
                    export_info['segments'] = np.stack(segments, 0)
                pass
            elif mode == 'classify':
                pass

            entity = Entity()(**export_info)
            yield entity

    return DataFrame(inner())


@dynamic_dispatch
def labelstudio_format_dc(ann_file, data_folder, category_map, is_debug=False):
    def inner():
        with open(ann_file, 'r') as fp:
            sample_anno_list = json.load(fp)
        
        for anno_info in sample_anno_list:
            if len(anno_info['annotations']) == 0:
                continue

            group_anno_ids = {}
            original_file = '-'.join(anno_info['file_upload'].split('-')[1:])

            export_info = {}
            export_info['filename'] = os.path.join(data_folder, original_file)
            export_info['image'] = cv2.imread(export_info['filename'])
            export_info['bboxes'] = []
            export_info['labels'] = []
            export_info['label_names'] = []
            export_info['segments'] = []
            export_info['has_segments'] = []
            sample_anno_info = anno_info['annotations'][0]
            for sample_anno_instance in sample_anno_info['result']:
                if sample_anno_instance['type'] == 'rectanglelabels':
                    # 矩形框标注
                    export_info['height'] = sample_anno_instance['original_height']
                    export_info['width'] = sample_anno_instance['original_width']
                                    
                    bbox_x = sample_anno_instance['value']['x'] / 100.0 * export_info['width']
                    bbox_y = sample_anno_instance['value']['y'] / 100.0 * export_info['height']
                    bbox_width = sample_anno_instance['value']['width'] / 100.0 * export_info['width']
                    bbox_height = sample_anno_instance['value']['height'] / 100.0 * export_info['height']
                    bbox_rotation = sample_anno_instance['value']['rotation']
                    export_info['bboxes'].append([bbox_x,bbox_y,bbox_x+bbox_width,bbox_y+bbox_height])

                    label_name = sample_anno_instance['value']['rectanglelabels'][0]
                    export_info['labels'].append(category_map[label_name] if label_name in category_map else -1)
                    export_info['label_names'].append(label_name)
                elif sample_anno_instance['type'] == 'keypointlabels':
                    # 2D关键点标注
                    export_info['height'] = sample_anno_instance['original_height']
                    export_info['width'] = sample_anno_instance['original_width']  
                    keypoint_x = sample_anno_instance['value']['x'] / 100.0 * export_info['width']
                    keypoint_y = sample_anno_instance['value']['y'] / 100.0 * export_info['height']
                    
                    # 先把所有关键点保存起来，然后进行组合到单个个体
                    label_name = sample_anno_instance['value']['keypointlabels'][0]
                    label_id = category_map[label_name]
                    # label_order = label_name_order[label_name]
                    
                    sample_anno_id = sample_anno_instance['id']
                    sample_parent_anno_id = sample_anno_instance['parentID'] if 'parentID' in sample_anno_instance else ''

                    group_anno_ids[sample_anno_id] = {
                        'keypoint_x': keypoint_x,
                        'keypoint_y': keypoint_y,
                        'label_name': label_name,
                        'label_id': label_id,
                        'anno_id': sample_anno_id,
                        'group_anno_id': sample_parent_anno_id
                    }
                elif sample_anno_instance['type'] == 'polygonlabels':
                    # 分割标注(polygon)
                    export_info['height'] = sample_anno_instance['original_height']
                    export_info['width'] = sample_anno_instance['original_width']  

                    points = sample_anno_instance['value']['points']
                    label_name = sample_anno_instance['value']['polygonlabels'][0]
                    label_id = category_map[label_name]
                    
                    points_array = np.array(points) 
                    points_array[:, 0] = points_array[:, 0] / 100.0 * export_info['width']
                    points_array[:, 1] = points_array[:, 1] / 100.0 * export_info['height']
                    points = points_array.tolist()
                    
                    bbox_x1 = float(np.min(points_array[:,0]))
                    bbox_y1 = float(np.min(points_array[:,1]))
                    bbox_x2 = float(np.max(points_array[:,0]))
                    bbox_y2 = float(np.max(points_array[:,1]))
                    export_info['bboxes'].append([bbox_x1, bbox_y1, bbox_x2, bbox_y2])

                    label_name = sample_anno_instance['value']['polygonlabels'][0]
                    export_info['labels'].append(category_map[label_name] if label_name in category_map else -1)
                    export_info['label_names'].append(label_name)

                    export_info['segments'].append(points)
                    export_info['has_segments'].append(1)
                    
                    assert(len(export_info['segments']) == len(export_info['bboxes']))
                elif sample_anno_instance['type'] == 'choices':
                    # 图片级类别标注
                    label_name = ','.join(sample_anno_instance['value']['choices'])
                    export_info['image_label_name'] = label_name
                    export_info['image_label'] = category_map[label_name] if label_name in category_map else -1

            if len(group_anno_ids) > 0:
                # 仅对关键点标注启用
                regroup_anno_ids = {}
                for k,v in group_anno_ids.items():
                    if v['group_anno_id'] != '': 
                        # 只抽取group id
                        regroup_anno_ids[v['group_anno_id']] = []

                for k,v in group_anno_ids.items():
                    if v['group_anno_id'] != '':
                        regroup_anno_ids[v['group_anno_id']].append(v)
                    elif v['anno_id'] in regroup_anno_ids:
                        regroup_anno_ids[v['anno_id']].append(v)
                
                # 重新排序每个group
                export_info['joints2d'] = [None for _ in range(len(regroup_anno_ids))]
                export_info['has_joints2d'] = []
                
                for group_i, group_key in enumerate(regroup_anno_ids.keys()):
                    export_info['joints2d'][group_i] = [[] for _ in range(len(category_map))]
                    export_info['has_joints2d'].append(1)
                    
                    for anno_info in regroup_anno_ids[group_key]:
                        label_id = anno_info['label_id']
                        export_info['joints2d'][group_i][label_id] = [anno_info['keypoint_x'], anno_info['keypoint_y']]

                # 
                for group_i in range(len(export_info['joints2d'])):
                    points_array = np.array(export_info['joints2d'][group_i]) 
                    
                    bbox_x1 = float(np.min(points_array[:,0]))
                    bbox_y1 = float(np.min(points_array[:,1]))
                    bbox_x2 = float(np.max(points_array[:,0]))
                    bbox_y2 = float(np.max(points_array[:,1]))
                    export_info['bboxes'].append([bbox_x1, bbox_y1, bbox_x2, bbox_y2])

                    export_info['labels'].append(0)
                    export_info['label_names'].append('unkown')
                    
            entity = Entity()(**export_info)
            yield entity

    return DataFrame(inner())


def _order_iterators(iterators):
    iterators = [iterator() for iterator in iterators]
    choice = 0
    while iterators:
        try:
            yield next(iterators[choice])
        except StopIteration:
            if iterators:
                del iterators[choice]
                choice += 1


def _transform(description, sample):
    new_sample = {}
    for k in sample.keys():
        if k == 'image':
            image = cv2.imdecode(np.frombuffer(sample[k], dtype='uint8'), 1)  # BGR mode, but need RGB mode
            new_sample[k] = image
            continue
        if not k.startswith('__'):
            if description[k] == 'numpy':
                dtype = numpy_dtype_map[sample[f'__{k}_type'][0]]
                shape = tuple(sample[f'__{k}_shape'])
                if isinstance(sample[k], bytes):
                    new_sample[k] = np.frombuffer(bytearray(sample[k]), dtype=dtype).reshape(shape).copy()
                else:
                    new_sample[k] = np.frombuffer(bytearray(sample[k].tobytes()), dtype=dtype).reshape(shape).copy()

                if new_sample[k].dtype == np.float64:
                    new_sample[k] = new_sample[k].astype(np.float32)
                if k == 'bboxes' and new_sample[k].dtype != np.float32:
                    new_sample[k] = new_sample[k].astype(np.float32)
            elif description[k] == 'str':
                new_sample[k] = sample[k].tobytes().decode('utf-8')
            elif description[k] == 'dict':
                new_sample[k] = json.loads(sample[k].tobytes().decode('utf-8'))
            else:
                new_sample[k] = sample[k]

    return new_sample


@dynamic_dispatch
def tfrecord_format_dc(dir, mode='detect', is_random=False):
    # 遍历文件夹，发现所有tfrecord数据
    dataset_folders = dir
    if isinstance(dir, str):
        dataset_folders = [dir]
    data_path_list = []
    index_path_list = []

    for dataset_folder in dataset_folders:
        part_path_list = []
        for tfrecord_file in os.listdir(dataset_folder):
            if tfrecord_file.endswith('tfrecord'):
                tfrecord_file = '-'.join(tfrecord_file.split('/')[-1].split('-')[:-1]+['tfrecord'])
                part_path_list.append(f'{dataset_folder}/{tfrecord_file}')

        part_index_path_list = []
        for i in range(len(part_path_list)):
            tfrecord_file = part_path_list[i]
            folder = os.path.dirname(tfrecord_file)
            if tfrecord_file.endswith('tfrecord') or tfrecord_file.endswith('index'):
                index_file = '-'.join(tfrecord_file.split('/')[-1].split('-')[:-1]+['index'])
                index_file = f'{folder}/{index_file}'
                tfrecord_file = '-'.join(tfrecord_file.split('/')[-1].split('-')[:-1]+['tfrecord'])
                part_path_list[i] = f'{folder}/{tfrecord_file}'
            else:
                index_file = tfrecord_file+'-index'
                part_path_list[i] = tfrecord_file+'-tfrecord'
            part_index_path_list.append(index_file)
    
        data_path_list.extend(part_path_list)
        index_path_list.extend(part_index_path_list)

    num_samples = 0
    num_samples_list = []
    for i, index_path in enumerate(index_path_list):
        index = np.loadtxt(index_path, dtype=np.int64)[:, 0]
        num_samples += len(index) 
        num_samples_list.append(len(index))

    # 分析解析信息
    description = {}
    if mode == "detect":
        description = {
            'image': 'byte', 
            'bboxes': 'numpy', 
            'labels': 'numpy'
        }
    elif mode == "segment":
        pass
    elif mode == "pose":
        pass

    inner_description = {}
    for k, v in description.items():
        if v == 'numpy':
            inner_description.update({
                k: 'byte',
                f'__{k}_type': 'int',
                f'__{k}_shape': 'int'
            })
        elif v == 'str':
            inner_description.update({
                k: 'byte'
            })
        elif v == 'dict':
            inner_description.update({
                k: 'byte'
            })
        else:
            inner_description.update({
                k: v
            })

    loaders = [functools.partial(tfrecord_loader, data_path=data_path,
                            index_path=index_path,
                            shard=None,
                            description=inner_description,
                            sequence_description=None,
                            compression_type=None,
                            )
        for data_path, index_path in zip(data_path_list, index_path_list)]

    it = _order_iterators(loaders)
    _transform_func = lambda x: _transform(description, x)
    it = map(_transform_func, it)

    def inner():
        while True:
            export_info = next(it)
            entity = Entity()(**export_info)
            yield entity

    return DataFrame(inner())


@dynamic_dispatch
def common_dataset_dc(dataset, **kwargs):
    if isinstance(dataset, str):
        kwargs.update({
            'type': dataset
        })
        dataset = build_from_cfg(dict(kwargs), DATASETS)

    def inner():
        if (not getattr(dataset, 'size', None)) or (not getattr(dataset, 'sample', None)):
            sample_num = len(dataset)
            for sample_i in range(sample_num):
                data_info = dataset[sample_i]
                entity = Entity()(**data_info)
                yield entity
        else:
            sample_num = dataset.size
            for sample_i in range(sample_num):
                data_info = dataset.sample(sample_i)
                entity = Entity()(**data_info)
                yield entity

    return DataFrame(inner())


class _dataset_dc(object):
    def __getattr__(self, name):
        if name == 'dataset':
            return common_dataset_dc
        if name not in ['coco','yolo','tfrecord','labelstudio']:
            return None

        return globals()[f'{name}_format_dc']


dataset_dc = _dataset_dc()