import sys
import os
import json
import numpy as np
from antgo.utils.sample_gt import *
from antgo.interactcontext import InteractContext
from antgo.framework.helper.fileio import *
from antgo.tools.package import *
import logging
from antgo.framework.helper.task_flag import *


def label_to_studio(src_json_file, tgt_folder, prefix='',  **kwargs):
    # 将标准GT转换到label-studio格式
    if tgt_folder is None:
        tgt_folder = './'
    

    pass


def label_start(src_json_file, tgt_folder, tags, label_type, white_users_str=None, ignore_incomplete=False, root=None, exp=None):
    # 检查参数
    assert(tags is not None)

    if tgt_folder is None:
        tgt_folder = './output'

    # 如果设置root and exp, 则按照规定规则读取等待标注数据。忽略src_json_file。
    if root is not None and exp is not None and root != '' and exp != '':
        # 开始标记
        running_flag(os.path.join(root, 'label', exp))
        if file_client_exists(os.path.join(root, 'label', exp, 'data.tar')):
            if not os.path.exists('./label'):
                os.mkdir('./label')
            
            file_client_get(os.path.join(root, 'label', exp, 'data.tar'), './label/')
            os.system('cd label && tar -xf data.tar')
            # 直接覆盖src_json_file
            src_json_file = './label/annotation.json'
            # 直接覆盖tgt_folder
            tgt_folder = './label/'

    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)

    if not os.path.isdir(tgt_folder):
        logging.error(f'--tgt {tgt_folder} must be a folder.')
        return 

    white_users = {}
    if white_users_str is not None:
        for t in white_users_str.split(','):
            user_name, password = t.split(':')
            white_users.update({
                user_name: {'password': password}
            })
    if len(white_users) == 0:
        white_users = None

    if label_type is None:
        label_type = ''

    assert(label_type in ['', 'RECT','POINT','POLYGON'])

    # label_name:0,label_name;1,...
    label_name_and_label_id_map = {}
    label_id_and_label_name_map = {}
    for label_name_and_label_id in tags.split(','):
        label_name, label_id = label_name_and_label_id.split(':')
        label_name_and_label_id_map[label_name] = int(label_id)
        label_id_and_label_name_map[int(label_id)] = label_name

    inner_category = []
    for label_name, label_id in label_name_and_label_id_map.items():
        inner_category.append({
            'class_name': label_name,
            'class_index': label_id
        })

    # 自动定义服务名字
    src_json_file_name = src_json_file.split('/')[-1].split('.')[0]

    # 启动标注服务，直接保存标准格式
    ctx = InteractContext()
    # 需要在这里设置标注配置信息，如
    # category: 设置类别信息
    # white_users: 设置允许参与的标注人员信息
    # label_type: 设置标注类型，目前仅支持'RECT','POINT','POLYGON'   
    ctx.activelearning.start(src_json_file_name, config={
            'metas':{
                'category': inner_category,
                'level': 'image-level' if label_type == '' else 'instance-level'
            },
            'white_users': white_users,
            'label_type': label_type,   # 设置标注类型，'RECT','POINT','POLYGON'
        }, json_file=src_json_file
    )

    # 切换到标注状态
    ctx.activelearning.labeling()

    # 下载当前标注结果（后台等待标注完成后，再下载）
    result = ctx.activelearning.download()

    # 转换到标准GT格式
    total_gt_list = []
    sgtt = SampleGTTemplate()
    for anno_id, anno_info in enumerate(result):
        standard_gt = sgtt.get()
        standard_gt['image_file'] = '/'.join(anno_info['file_upload'].split('/')[3:])
        standard_gt['tag'] = src_json_file_name

        if len(anno_info['annotations']) == 0:
            continue

        sample_anno_info = anno_info['annotations'][0]
        for sample_anno_instance in sample_anno_info['result']:  
            if sample_anno_instance['type'] == 'RECT':
                standard_gt['height'] = 0 if 'height' not in sample_anno_instance else sample_anno_instance['height']
                standard_gt['width'] = 0 if 'width' not in sample_anno_instance else sample_anno_instance['width']
                            
                bbox_x = sample_anno_instance['value']['x']
                bbox_y = sample_anno_instance['value']['y']
                bbox_width = sample_anno_instance['value']['width']
                bbox_height = sample_anno_instance['value']['height']
                standard_gt['bboxes'].append([bbox_x,bbox_y,bbox_x+bbox_width,bbox_y+bbox_height])

                label_id = sample_anno_instance['value']['labels']
                standard_gt['labels'].append(label_id)
                standard_gt['label_names'].append(label_id_and_label_name_map[label_id])
            elif sample_anno_instance['type'] == 'POLYGON':
                standard_gt['height'] = 0 if 'height' not in sample_anno_instance else sample_anno_instance['height']
                standard_gt['width'] = 0 if 'width' not in sample_anno_instance else sample_anno_instance['width']
                                
                points = []
                for xxyy in sample_anno_instance['value']['points']:
                    points.append([xxyy['x'], xxyy['y']])
                    
                points_array = np.array(points) 
                bbox_x1 = float(np.min(points_array[:,0]))
                bbox_y1 = float(np.min(points_array[:,1]))
                bbox_x2 = float(np.max(points_array[:,0]))
                bbox_y2 = float(np.max(points_array[:,1]))
                standard_gt['bboxes'].append([bbox_x1, bbox_y1, bbox_x2, bbox_y2])

                label_id = sample_anno_instance['value']['labels']
                standard_gt['labels'].append(label_id)
                standard_gt['label_names'].append(label_id_and_label_name_map[label_id])

                standard_gt['segments'].append(points)
                standard_gt['has_segments'].append(1)
            elif sample_anno_instance['type'] == 'POINT':
                pass
            elif sample_anno_instance['type'] == 'CHOICES':
                standard_gt['height'] = 0 if 'height' not in sample_anno_instance else sample_anno_instance['height']
                standard_gt['width'] = 0 if 'width' not in sample_anno_instance else sample_anno_instance['width']
                                
                image_labe_name = ','.join(sample_anno_instance['value']['labels'])
                standard_gt['image_label_name'] = image_labe_name
                standard_gt['image_label'] = label_name_and_label_id_map[image_labe_name] if image_labe_name in label_name_and_label_id_map else -1
 
        if ignore_incomplete:
            if len(standard_gt['bboxes']) == 0 and standard_gt['image_label'] == -1:
                # 忽略空标注样本
                continue
        
        total_gt_list.append(standard_gt)    

    # # debug
    # total_gt_list = []
    # sgtt = SampleGTTemplate()
    # for _ in range(1000):
    #     standard_gt = sgtt.get()
    #     image_files = os.listdir('./label/images')
    #     print('images files number')
    #     print(len(image_files))
    #     standard_gt['image_file'] = f'images/{image_files[0]}'
    #     total_gt_list.append(standard_gt)

    anno_json_file_name = f'{src_json_file_name}_label.json'
    if os.path.exists(os.path.join(tgt_folder,anno_json_file_name)):
        count = 0
        while True:
            anno_json_file_name = f'{src_json_file_name}_label_{count}.json'
            if not os.path.exists(os.path.join(tgt_folder,anno_json_file_name)):
                break
            count += 1

    with open(os.path.join(tgt_folder, anno_json_file_name), 'w',  encoding="utf-8") as fp:
        json.dump(total_gt_list, fp)

    meta_info = sgtt.meta()    
    if os.path.exists(os.path.join(tgt_folder, 'meta.json')):
         with open(os.path.join(tgt_folder, 'meta.json'), 'r', encoding="utf-8") as fp:
             meta_info = json.load(fp)

    meta_info['extent']['total'] = len(total_gt_list)
    with open(os.path.join(tgt_folder, 'meta.json'), 'w', encoding="utf-8") as fp:
        json.dump(meta_info, fp)

    if root is not None and exp is not None and root != '' and exp != '':
        # 仅在设置root and exp后，起作用
        # 1. 打包到tfrecord格式
        package_to_tfrecord(
            os.path.join(tgt_folder, anno_json_file_name), 
            os.path.join(tgt_folder, 'package'),
            prefix=exp,
            size_in_shard=40000,
            thread_num=1
        )

        # 2. 上传到远程
        if not file_client_exists(os.path.join(root, 'dataset', 'label')):
            file_client_mkdir(os.path.join(root, 'dataset', 'label'), True)

        # 打包后的数据
        for file_name in os.listdir(os.path.join(tgt_folder, 'package')):
            file_client_put(
                os.path.join(root, 'dataset', 'label'), 
                os.path.join(tgt_folder, 'package', file_name)
            )

        # 标注文件
        file_client_put(
            os.path.join(root, 'label', exp),
            os.path.join(tgt_folder, anno_json_file_name)
        )

        # 完成标记
        finish_flag(os.path.join(root, 'label', exp))
        
    # 全局结束
    ctx.activelearning.exit()


def label_from_native(src_json_file, tgt_folder, tags, ignore_incomplete=False, **kwargs):
    # 加载原生标注工具的结果
    with open(src_json_file, 'r') as fp:
        result = json.load(fp)
    
    # label_name:0,label_name;1,...
    label_name_and_label_id_map = {}
    label_id_and_label_name_map = {}
    for label_name_and_label_id in tags.split(','):
        label_name, label_id = label_name_and_label_id.split(':')
        label_name_and_label_id_map[label_name] = int(label_id)
        label_id_and_label_name_map[int(label_id)] = label_name
        
    # 转换到标准GT格式
    total_gt_list = []
    src_json_file_name = src_json_file.split('/')[-1].split('.')[0]
    sgtt = SampleGTTemplate()
    for anno_id, anno_info in enumerate(result):
        standard_gt = sgtt.get()
        standard_gt['image_file'] = '/'.join(anno_info['file_upload'].split('/')[3:])
        standard_gt['tag'] = src_json_file_name
        
        if len(anno_info['annotations']) == 0:
            continue
        
        sample_anno_info = anno_info['annotations'][0]
        for sample_anno_instance in sample_anno_info['result']:  
            if sample_anno_instance['type'] == 'RECT':
                standard_gt['height'] = 0 if 'height' not in sample_anno_instance else sample_anno_instance['height']
                standard_gt['width'] = 0 if 'width' not in sample_anno_instance else sample_anno_instance['width']
                            
                bbox_x = sample_anno_instance['value']['x']
                bbox_y = sample_anno_instance['value']['y']
                bbox_width = sample_anno_instance['value']['width']
                bbox_height = sample_anno_instance['value']['height']
                standard_gt['bboxes'].append([bbox_x,bbox_y,bbox_x+bbox_width,bbox_y+bbox_height])

                label_id = sample_anno_instance['value']['labels']
                standard_gt['labels'].append(label_id)
                standard_gt['label_names'].append(label_id_and_label_name_map[label_id])
            elif sample_anno_instance['type'] == 'POLYGON':
                standard_gt['height'] = 0 if 'height' not in sample_anno_instance else sample_anno_instance['height']
                standard_gt['width'] = 0 if 'width' not in sample_anno_instance else sample_anno_instance['width']
                                
                points = []
                for xxyy in sample_anno_instance['value']['points']:
                    points.append([xxyy['x'], xxyy['y']])
                    
                points_array = np.array(points) 
                bbox_x1 = float(np.min(points_array[:,0]))
                bbox_y1 = float(np.min(points_array[:,1]))
                bbox_x2 = float(np.max(points_array[:,0]))
                bbox_y2 = float(np.max(points_array[:,1]))
                standard_gt['bboxes'].append([bbox_x1, bbox_y1, bbox_x2, bbox_y2])

                label_id = sample_anno_instance['value']['labels']
                standard_gt['labels'].append(label_id)
                standard_gt['label_names'].append(label_id_and_label_name_map[label_id])

                standard_gt['segments'].append(points)
                standard_gt['has_segments'].append(1)
            elif sample_anno_instance['type'] == 'POINT':
                pass
            elif sample_anno_instance['type'] == 'CHOICES':
                standard_gt['height'] = 0 if 'height' not in sample_anno_instance else sample_anno_instance['height']
                standard_gt['width'] = 0 if 'width' not in sample_anno_instance else sample_anno_instance['width']
                                
                image_labe_name = ','.join(sample_anno_instance['value']['labels'])
                standard_gt['image_label_name'] = image_labe_name
                standard_gt['image_label'] = label_name_and_label_id_map[image_labe_name] if image_labe_name in label_name_and_label_id_map else -1

        if ignore_incomplete:
            if len(standard_gt['bboxes']) == 0 and standard_gt['image_label'] == -1:
                # 忽略空标注样本
                continue
            
        total_gt_list.append(standard_gt)    
    
    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)
    
    anno_json_file_name = f'{src_json_file_name}_label.json'
    if os.path.exists(os.path.join(tgt_folder,anno_json_file_name)):
        count = 0
        while True:
            anno_json_file_name = f'{src_json_file_name}_label_{count}.json'
            if not os.path.exists(os.path.join(tgt_folder,anno_json_file_name)):
                break
            count += 1
            
    with open(os.path.join(tgt_folder, anno_json_file_name), 'w',  encoding="utf-8") as fp:
        json.dump(total_gt_list, fp)
    
    meta_info = sgtt.meta()    
    if os.path.exists(os.path.join(tgt_folder, 'meta.json')):
         with open(os.path.join(tgt_folder, 'meta.json'), 'r', encoding="utf-8") as fp:
             meta_info = json.load(fp)

    meta_info['extent']['total'] = len(total_gt_list)
    with open(os.path.join(tgt_folder, 'meta.json'), 'w', encoding="utf-8") as fp:
        json.dump(meta_info, fp)    


def label_from_studio(src_json_file, tgt_folder, prefix='', tags=None, ignore_incomplete=False, **kwargs):
    # 将label-studio转换到标注GT
    if tgt_folder is None:
        tgt_folder = './'
    
    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)
        
    if not os.path.isdir(tgt_folder):
        logging.error(f'--tgt {tgt_folder} must be a folder.')
        return 
    
    if tags is None:
        tags = ''
    if prefix is None:
        prefix = ''
    
    # label_name:0,label_name;1,...
    label_name_and_label_id_map = {}
    for label_name_and_label_id in tags.split(','):
        label_name, label_id = label_name_and_label_id.split(':')
        label_name_and_label_id_map[label_name] = int(label_id)
    
    src_json_file_name = src_json_file.split('/')[-1].split('.')[0]
    
    with open(src_json_file, 'r') as fp:
        sample_anno_list = json.load(fp)
    
    total_gt_list = []
    sgtt = SampleGTTemplate()
    for anno_info in sample_anno_list:
        # only use human 0 result
        if len(anno_info['annotations']) == 0:
            continue
        
        group_anno_ids = {}
        original_file = '-'.join(anno_info['file_upload'].split('-')[1:])
        standard_gt = sgtt.get()
        standard_gt['image_file'] = os.path.join(prefix, original_file) if prefix != '' else original_file
        standard_gt['tag'] = src_json_file_name
        sample_anno_info = anno_info['annotations'][0]
        for sample_anno_instance in sample_anno_info['result']:
            if sample_anno_instance['type'] == 'rectanglelabels':
                # 矩形框标注
                standard_gt['height'] = sample_anno_instance['original_height']
                standard_gt['width'] = sample_anno_instance['original_width']
                                
                bbox_x = sample_anno_instance['value']['x'] / 100.0 * standard_gt['width']
                bbox_y = sample_anno_instance['value']['y'] / 100.0 * standard_gt['height']
                bbox_width = sample_anno_instance['value']['width'] / 100.0 * standard_gt['width']
                bbox_height = sample_anno_instance['value']['height'] / 100.0 * standard_gt['height']
                bbox_rotation = sample_anno_instance['value']['rotation']
                standard_gt['bboxes'].append([bbox_x,bbox_y,bbox_x+bbox_width,bbox_y+bbox_height])

                label_name = sample_anno_instance['value']['rectanglelabels'][0]
                standard_gt['labels'].append(label_name_and_label_id_map[label_name] if label_name in label_name_and_label_id_map else -1)
                standard_gt['label_names'].append(label_name)
            elif sample_anno_instance['type'] == 'keypointlabels':
                # 2D关键点标注
                standard_gt['height'] = sample_anno_instance['original_height']
                standard_gt['width'] = sample_anno_instance['original_width']  
                keypoint_x = sample_anno_instance['value']['x'] / 100.0 * standard_gt['width']
                keypoint_y = sample_anno_instance['value']['y'] / 100.0 * standard_gt['height']
                
                # 先把所有关键点保存起来，然后进行组合到单个个体
                label_name = sample_anno_instance['value']['keypointlabels'][0]
                label_id = label_name_and_label_id_map[label_name]
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
                standard_gt['height'] = sample_anno_instance['original_height']
                standard_gt['width'] = sample_anno_instance['original_width']  
                
                points = sample_anno_instance['value']['points']
                label_name = sample_anno_instance['value']['polygonlabels'][0]
                label_id = label_name_and_label_id_map[label_name]
                
                points_array = np.array(points) 
                points_array[:, 0] = points_array[:, 0] / 100.0 * standard_gt['width']
                points_array[:, 1] = points_array[:, 1] / 100.0 * standard_gt['height']
                points = points_array.tolist()
                
                bbox_x1 = float(np.min(points_array[:,0]))
                bbox_y1 = float(np.min(points_array[:,1]))
                bbox_x2 = float(np.max(points_array[:,0]))
                bbox_y2 = float(np.max(points_array[:,1]))
                standard_gt['bboxes'].append([bbox_x1, bbox_y1, bbox_x2, bbox_y2])

                label_name = sample_anno_instance['value']['polygonlabels'][0]
                standard_gt['labels'].append(label_name_and_label_id_map[label_name] if label_name in label_name_and_label_id_map else -1)
                standard_gt['label_names'].append(label_name)

                standard_gt['segments'].append(points)
                standard_gt['has_segments'].append(1)
                
                assert(len(standard_gt['segments']) == len(standard_gt['bboxes']))
            elif sample_anno_instance['type'] == 'choices':
                # 图片级类别标注
                label_name = ','.join(sample_anno_instance['value']['choices'])
                standard_gt['image_label_name'] = label_name
                standard_gt['image_label'] = label_name_and_label_id_map[label_name] if label_name in label_name_and_label_id_map else -1
       
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
            standard_gt['joints2d'] = [None for _ in range(len(regroup_anno_ids))]
            standard_gt['has_joints2d'] = []
            
            for group_i, group_key in enumerate(regroup_anno_ids.keys()):
                standard_gt['joints2d'][group_i] = [[] for _ in range(len(label_name_and_label_id_map))]
                standard_gt['has_joints2d'].append(1)
                
                for anno_info in regroup_anno_ids[group_key]:
                    label_id = anno_info['label_id']
                    standard_gt['joints2d'][group_i][label_id] = [anno_info['keypoint_x'], anno_info['keypoint_y']]

            # 
            for group_i in range(len(standard_gt['joints2d'])):
                points_array = np.array(standard_gt['joints2d'][group_i]) 
                
                bbox_x1 = float(np.min(points_array[:,0]))
                bbox_y1 = float(np.min(points_array[:,1]))
                bbox_x2 = float(np.max(points_array[:,0]))
                bbox_y2 = float(np.max(points_array[:,1]))
                standard_gt['bboxes'].append([bbox_x1, bbox_y1, bbox_x2, bbox_y2])

                standard_gt['labels'].append(0)
                standard_gt['label_names'].append('unkown')
                
        total_gt_list.append(standard_gt)

    anno_json_file_name = f'{src_json_file_name}_label.json'
    if os.path.exists(os.path.join(tgt_folder,anno_json_file_name)):
        count = 0
        while True:
            anno_json_file_name = f'{src_json_file_name}_label_{count}.json'
            if not os.path.exists(os.path.join(tgt_folder,anno_json_file_name)):
                break
            count += 1
            
    with open(os.path.join(tgt_folder, anno_json_file_name), 'w',  encoding="utf-8") as fp:
        json.dump(total_gt_list, fp)
    
    meta_info = sgtt.meta()    
    if os.path.exists(os.path.join(tgt_folder, 'meta.json')):
         with open(os.path.join(tgt_folder, 'meta.json'), 'r', encoding="utf-8") as fp:
             meta_info = json.load(fp)

    meta_info['extent']['total'] = len(total_gt_list)
    with open(os.path.join(tgt_folder, 'meta.json'), 'w', encoding="utf-8") as fp:
        json.dump(meta_info, fp)  


def label_to_merge(src_json_file, tgt_folder, **kwargs):
    src_json_files = src_json_file.split(',')
    total_list = []
    for json_file in src_json_files:
        if not json_file.endswith('json'):
            logging.warn(f'Ignore {json_file}')
            continue
        
        with open(json_file, 'r', encoding='utf-8') as fp:
            total_list.extend(json.load(fp))

    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)
    
    if not os.path.isdir(tgt_folder):
        logging.error(f'--tgt={tgt_folder} must be a folder')
        return
    
    anno_json_file_name = f'annotation_merge.json'
    if os.path.exists(os.path.join(tgt_folder,anno_json_file_name)):
        count = 0
        while True:
            anno_json_file_name = f'annotation_merge_{count}.json'
            if not os.path.exists(os.path.join(tgt_folder,anno_json_file_name)):
                break
            count += 1
            
    with open(os.path.join(tgt_folder, anno_json_file_name), 'w',  encoding="utf-8") as fp:
        json.dump(total_list, fp)
    
    sgtt = SampleGTTemplate()
    meta_info = sgtt.meta()    
    if os.path.exists(os.path.join(tgt_folder, 'meta.json')):
         with open(os.path.join(tgt_folder, 'meta.json'), 'r', encoding="utf-8") as fp:
             meta_info = json.load(fp)

    meta_info['extent']['total'] = len(total_list)
    with open(os.path.join(tgt_folder, 'meta.json'), 'w', encoding="utf-8") as fp:
        json.dump(meta_info, fp)  

# 测试检测标注
# label_from_studio('/Users/bytedance/Downloads/project-1-at-2023-03-11-02-17-0c19afd8.json', None, '', "Airplane:0,Car:1")
# 测试分割
# label_from_studio('/Users/bytedance/Downloads/project-4-at-2023-03-11-03-41-0557a589.json', None, '', "Airplane:0,Car:1")
# 测试图片标签
# label_from_studio('/Users/bytedance/Downloads/project-2-at-2023-03-11-02-25-96db91e7.json', None, '', "Weapons:0,Violence:1")
# 测试关键点
# label_from_studio('/Users/bytedance/Downloads/project-3-at-2023-03-11-03-01-3a7cf0e3.json', None, '', "Face:0,Nose:1")
# print('a')

# 测试标注
# if __name__ == '__main__':
#     label_start('/Users/bytedance/Downloads/mm/annotation.json', None, 'Car:0', "POLYGON")
#     print('sdf')