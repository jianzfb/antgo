import sys
from fileinput import filename
from random import shuffle
import shutil
import cv2
import os
import json
import copy
import glob
import logging
import numpy as np
from pprint import pprint
import imagesize
import requests
from antgo.utils.sample_gt import *
from pycocotools.coco import COCO


def extract_from_videos(video_folder, target_folder, frame_rate=10, max_size=0, **kwargs):
    # 输出
    # -data
    # -annotation.json
    if target_folder is None:
        target_folder = './'
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 不支持多级目录
    # 抽帧过程
    support_video_ext = ['mp4', 'avi']
    frame_height = 0
    frame_width = 0
    for video_file_name in os.listdir(video_folder):
        if video_file_name[0] == '.':
            continue
        if video_file_name.split('.')[-1].lower() not in support_video_ext:
            continue
        
        video_file_pure_name = video_file_name.split('.')[0]
        if not os.path.exists(os.path.join(target_folder, video_file_pure_name)):
            os.makedirs(os.path.join(target_folder, video_file_pure_name))
        video_file_path = os.path.join(video_folder, video_file_name)
        cap = cv2.VideoCapture(video_file_path) 
        print(f'process video {video_file_name}')
        count = 0
        while(cap.isOpened()): 
            ret, frame = cap.read() 
            if not ret:
                break
            
            frame_height, frame_width = frame.shape[:2]
            
            if max_size > 0:
                scale = 1.0
                if frame_height > frame_width:
                    scale = max_size / frame_height
                else:
                    scale = max_size / frame_width
                frame = cv2.resize(frame, None, None, fx=scale, fy=scale)
                
            if count % frame_rate == 0:
                cv2.imwrite(os.path.join(target_folder, video_file_pure_name, f'{video_file_pure_name}_rate-{frame_rate}_frame-{count}.png'), frame)

            count += 1

        cap.release() 

    # 生成默认格式解析json
    annotation_file_name = 'annotation.json'
    annotation_list = []
    # 加载默认标准格式
    sample_gt_file = os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-1]), 'resource', 'templates', 'sample_gt.json')
    with open(sample_gt_file, 'r', encoding="utf-8") as fp:
        sample_gt = json.load(fp)

    for video_name in os.listdir(target_folder):
        if video_name[0] == '.':
            continue
        if not os.path.isdir(os.path.join(target_folder, video_name)):
            continue

        for file_name in os.listdir(os.path.join(target_folder, video_name)):
            if file_name[0] == '.':
                continue
            
            sample_gt_cp = copy.deepcopy(sample_gt)
            sample_gt_cp.update({
                'image_file': f'{video_name}/{file_name}'
            })
            sample_gt_cp['height'] = frame_height
            sample_gt_cp['width'] = frame_width
            annotation_list.append(sample_gt_cp)

    print(f'Extract frame number {len(annotation_list)}')

    with open(os.path.join(target_folder, annotation_file_name), 'w', encoding="utf-8") as fp:
        json.dump(annotation_list, fp)
    

def extract_from_images(
    source_folder, 
    target_folder, 
    filter_prefix=None, 
    filter_suffix=None, 
    filter_ext=None, 
    shuffle=False, num=0, max_size=0, **kwargs):
    support_image_ext = ['png', 'jpeg', 'jpg']
    if filter_ext is not None:
        temp = []
        for ext in support_image_ext:
            if ext in filter_ext:
                temp.append(ext)
        
        support_image_ext = temp

    if target_folder is None:
        target_folder = './'

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        
    # 遍历source_folder 所有文件并保留文件目录结构复制
    sample_file_list = []
    for root, dirnames, filenames in os.walk(source_folder):
        for filename in filenames:
            if filename[0] == '.':
                continue
            if filename.split('/')[-1].split('.')[-1].lower() not in support_image_ext:
                continue
            if filter_prefix is not None:
                if not filename.startswith(filter_prefix):
                    continue
            
            if filter_suffix is not None:
                if not filename.split('.')[0].endswith(filter_suffix):
                    continue
            
            if filter_ext is not None:
                if not filename.endswith(filter_ext):
                    continue
                    
            # copy to 对应目录
            source_path = os.path.join(root, filename)
            rel_path = os.path.relpath(source_path, source_folder)
            sample_file_list.append(rel_path)

    if shuffle:
        np.random.shuffle(sample_file_list)
    
    if num <= 0:
        num = len(sample_file_list)
    
    # 生成默认格式解析json
    annotation_file_name = 'annotation.json'
    annotation_list = []
    # 加载默认标准格式
    sample_gt_file = os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-1]), 'resource', 'templates', 'sample_gt.json')
    with open(sample_gt_file, 'r', encoding="utf-8") as fp:
        sample_gt = json.load(fp)

    for sample_file_name in sample_file_list[:num]:
        source_path = os.path.join(source_folder, sample_file_name)
        target_path = os.path.join(target_folder, sample_file_name)
        dir_path = os.path.dirname(target_path)
        if not os.path.exists(os.path.join(dir_path)):
            os.makedirs(dir_path)
        
        if max_size > 0:
            image = cv2.imread(source_path)
            height, width = image.shape[:2]
            scale = 1.0
            if height > width:
                scale = max_size / height
            else:
                scale = max_size / width
            image = cv2.resize(image, None, None, fx=scale, fy=scale)
            cv2.imwrite(target_path, image)
        else:  
            shutil.copy(source_path, target_path)
            
        sample_file_path = os.path.join(target_folder, sample_file_name)
        width, height = imagesize.get(sample_file_path)
    
        sample_gt_cp = copy.deepcopy(sample_gt)
        sample_gt_cp.update({
            'image_file': sample_file_name
        })
        sample_gt_cp['height'] = height
        sample_gt_cp['width'] = width
        annotation_list.append(sample_gt_cp)

    print(f'extract image number {len(annotation_list)}')

    with open(os.path.join(target_folder, annotation_file_name), 'w', encoding="utf-8") as fp:
        json.dump(annotation_list, fp)


def extract_from_coco(source_file, target_folder, filter_label=None, **kwargs):
    if not source_file.endswith('json'):
        logging.error('Source file must be json file')
        return

    if filter_label is None:
        filter_label = ''    
    filter_label_map = {}
    for ll in filter_label.split(','):
        label_name, label_index = ll.split(":")
        filter_label_map[label_name] = int(label_index)
    
    if target_folder is None:
        target_folder = './'
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    total_samples = []
    sgt = SampleGTTemplate() 
    db = COCO(source_file)
    for aid in db.anns.keys():
        ann = db.anns[aid]
        image_id = ann['image_id']
        img_info = db.loadImgs(image_id)[0]
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        gt_template = sgt.get()
        gt_template['image_file'] = file_name
        gt_template['width'] = width
        gt_template['height'] = height
        anns = db.imgToAnns[image_id]
        
        is_ignore = True
        for ins_info in anns:
            ins_segmentation = ins_info['segmentation']
            ins_bbox = ins_info['bbox']
            box_x, box_y, box_w, box_h = ins_bbox
            ins_category = ins_info['category_id']
            ins_category_name = db.cats[ins_category]['name']
            
            if len(filter_label_map) > 0:
                if ins_category_name not in filter_label_map:
                    continue
                ins_category = filter_label_map[ins_category_name]
                
            is_ignore = False
            if len(ins_bbox) > 0:
                gt_template['labels'].append(ins_category)
                gt_template['label_names'].append(ins_category_name)
                gt_template['bboxes'].append([box_x, box_y, box_x+box_w, box_y+box_h])
            
                gt_template['segments'].append(ins_segmentation)
                if len(ins_segmentation) > 0:
                    gt_template['has_segments'].append(1)
                else:
                    gt_template['has_segments'].append(0)

        if not is_ignore:
            total_samples.append(gt_template)
        
    with open(os.path.join(target_folder, 'annotation.json'), 'w', encoding="utf-8") as fp:
        json.dump(total_samples, fp)

    with open(os.path.join(target_folder, 'meta.json'), 'w', encoding="utf-8") as fp:
        json.dump(sgt.meta(), fp)


def extract_from_crop(source_file, target_folder, **kwargs):
    # 仅支持标准GT
    source_file_name = source_file.split('/')[-1]
    pure_file_name = source_file_name.split('.')[0]
    ext_name =source_file_name.split('.')[-1]
    if ext_name != 'json':
        logging.error('Only support json or txt file.')
        return
    
    if target_folder is None:
        target_folder = './'

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    if not os.path.exists(os.path.join(target_folder, 'images')):
        os.makedirs(os.path.join(target_folder, 'images'))

    with open(source_file, 'r', encoding="utf-8") as fp:
        content = json.load(fp)

    src_folder = os.path.dirname(source_file)
    sgtt = SampleGTTemplate()
    total_gt_list = []
    for sample_i, sample in enumerate(content):
        bboxes = sample['bboxes']
        labels = sample['labels']
        label_names = sample['label_names']
        if len(label_names) == 0:
            label_names = [str(l) for l in labels]
        
        image = None
        file_name = f'{str(sample_i)}.png'
        if sample['image_file'] != '':
            image_path = os.path.join(src_folder, sample['image_file'])
            try:
                image = cv2.imread(image_path)
                file_name = image_path.split('/')[-1]
            except:
                logging.error("Couldnt imread %s."%sample['image_file'])
                image = None
        else:
            try:
                pic = requests.get(sample['image_url'], timeout=20)
                image = cv2.imdecode(np.frombuffer(pic.content, np.uint8), cv2.IMREAD_COLOR)
            except:
                logging.error("Couldnt download %s."%sample['image_url'])
                image = None

        if image is None:
            continue
        
        ext_name = file_name.split('.')[-1]
        pure_file_name = file_name.split(f'.{ext_name}')[0]
        height, width = image.shape[:2]
        
        for box_i, (box, label, label_name) in enumerate(zip(bboxes, labels, label_names)):
            x0,y0,x1,y1 = box
            
            box_w = x1 - x0
            box_h = y1 - y0
            
            if kwargs.get('ext_ratio') > 0:
                box_w = box_w * (1+kwargs['ext_ratio'])
                box_h = box_h * (1+kwargs['ext_ratio'])
            box_size = max(box_w, box_h)
            
            cx = (x0+x1)/2.0
            cy = (y0+y1)/2.0
               
            x0 = int(max(cx-box_size/2, 0))
            y0 = int(max(cy-box_size/2, 0))
            x1 = int(min(cx+box_size/2, width))
            y1 = int(min(cy+box_size/2, height))
            
            if x0 == x1 or y0 == y1:
                continue
                        
            patch_image = image[y0:y1,x0:x1]
            patch_file_name = f'{pure_file_name}_box_{box_i}_label_{label_name}.png'
            subfolder = label_name
            if not os.path.exists(os.path.join(target_folder, 'images', subfolder)):
                os.makedirs(os.path.join(target_folder, 'images', subfolder))
            cv2.imwrite(os.path.join(target_folder, 'images', subfolder, patch_file_name), patch_image)
            
            standard_gt = sgtt.get()
            standard_gt['image_file'] = f'images/{subfolder}/{patch_file_name}'
            standard_gt['height'] = patch_image.shape[0]
            standard_gt['width'] = patch_image.shape[1]
            standard_gt['image_label'] = label
            standard_gt['image_label_name'] = label_name
            
            if len(sample['has_joints2d']) > 0:
                if sample['has_joints2d'][box_i]:
                    standard_gt['has_joints2d'].append(1)
                    standard_gt['joints2d'].append(sample['joints2d'][box_i])
                else:
                    standard_gt['has_joints2d'].append(0)
                    standard_gt['joints2d'].append([])
            
            if len(sample['has_segments']) > 0:
                if sample['has_segments'][box_i]:
                    standard_gt['has_segments'].append(1)
                    standard_gt['segments'].append(sample['segments'][box_i])
                else:
                    standard_gt['has_segments'].append(0)
                    standard_gt['segments'].append([])
            
            total_gt_list.append(standard_gt)

    with open(os.path.join(target_folder, 'annotation.json'), 'w', encoding="utf-8") as fp:
        json.dump(total_gt_list, fp)

    with open(os.path.join(target_folder, 'meta.json'), 'w', encoding="utf-8") as fp:
        json.dump(sgtt.meta(), fp)


def extract_from_samples(source_file, target_folder, num=1, feedback=False, **kwargs):
    source_file_name = source_file.split('/')[-1]
    pure_file_name = source_file_name.split('.')[0]
    ext_name =source_file_name.split('.')[-1]
    if ext_name not in ['json', 'txt']:
        logging.error('Only support json or txt file.')
        return
    
    if target_folder is None:
        target_folder = './'
    
    # 随机抽取10条数据
    samples = []
    sample_indexs = []
    total_sample_num = 0
    if ext_name == 'json':
        with open(source_file, 'r', encoding="utf-8") as fp:
            content = json.load(fp)
            total_sample_num = len(content)
            sample_num = min(total_sample_num, num)

            sample_indexs = np.random.choice(total_sample_num, sample_num, replace=False).tolist()
            for i in sample_indexs:
                samples.append(content[i])

        target_file_path = os.path.join(target_folder, f'{pure_file_name}_samples_{len(samples)}.{ext_name}')
        with open(target_file_path, 'w', encoding="utf-8") as fp:
            json.dump(samples, fp)

        if feedback:
            # step1: 显示样本总条数
            print(f'Total Sample Num: {total_sample_num}')
            
            # step2: 显示keys
            if isinstance(samples[0], dict):
                print('Total Keys: ')
                print(samples[0].keys())

            # step3: 显示采样的索引
            print('Sample Index: ')
            print(sample_indexs)

            # step4: 显示采样的样本
            print('Samples: ')
            pprint(samples)
    else:
        with open(source_file, 'r', encoding="utf-8") as fp:
            content = fp.readlines()

            total_sample_num = len(content)
            sample_num = min(total_sample_num, num)
            sample_indexs = np.random.choice(total_sample_num, sample_num, replace=False).tolist()
            for i in sample_indexs:
                samples.append(content[i].strip())
        
        target_file_path = os.path.join(target_folder, f'{pure_file_name}_samples_{len(samples)}.{ext_name}')
        with open(target_file_path, 'w', encoding="utf-8") as fp:
            for s in samples:
                fp.write(f'{s}\n')

        if feedback:
            # step1: 显示样本总条数
            print(f'Total Sample Num: {total_sample_num}')

            # step2: 显示采样的样本行索引
            print('Sample Index: ')
            print(sample_indexs)

            # step3: 显示采样的样本
            print('Samples: ')
            for s in samples:
                print(f'{s}\n')


# extract_from_coco('/Volumes/Elements/手势/gesture/annotations/dumix_test_face_merge_200622_bin.json', './', 'hand:1')
# extract_from_coco('/Volumes/Elements/手势/gesture/annotations/dumix_train_face_201117_tri.json', './', 'hand:1')
# print('sdf')