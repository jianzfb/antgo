from fileinput import filename
from random import shuffle
import shutil
import cv2
import os
import json
import copy
import glob


def extract_from_videos(video_folder, target_folder, frame_rate=10, **kwargs):
    # 输出
    # -data
    # -annotation.json
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 不支持多级目录
    # 抽帧过程
    support_video_ext = ['mp4', 'avi']
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

            if count % frame_rate == 0:
                cv2.imwrite(os.path.join(target_folder, video_file_pure_name, f'{video_file_pure_name}_rate-{frame_rate}_frame-{count}.png'), frame)

            count += 1

        cap.release() 

    # 生成默认格式解析json
    annotation_file_name = 'annotation.json'
    annotation_list = []
    # 加载默认标准格式
    sample_gt_file = os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-1]), 'resource', 'templates', 'sample_gt.json')
    with open(sample_gt_file, 'r') as fp:
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
            annotation_list.append(sample_gt_cp)

    print(f'extract frame number {len(annotation_list)}')

    with open(os.path.join(target_folder, annotation_file_name), 'w') as fp:
        json.dump(annotation_list, fp)


def extract_from_images(source_folder, target_folder, filter_prefix=None, filter_suffix=None, filter_ext=None, **kwargs):
    support_image_ext = ['png', 'jpeg', 'jpg']
    if filter_ext is not None:
        temp = []
        for ext in support_image_ext:
            if ext in filter_ext:
                temp.append(ext)
        
        support_image_ext = temp

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
            target_path = os.path.join(target_folder, rel_path)
            dir_path = os.path.dirname(target_path)
            if not os.path.exists(os.path.join(dir_path)):
                os.makedirs(dir_path)
            
            shutil.copy(source_path, target_path)
            sample_file_list.append(rel_path)

    # 生成默认格式解析json
    annotation_file_name = 'annotation.json'
    annotation_list = []
    # 加载默认标准格式
    sample_gt_file = os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-1]), 'resource', 'templates', 'sample_gt.json')
    with open(sample_gt_file, 'r') as fp:
        sample_gt = json.load(fp)

    for sample_file_name in sample_file_list:
        sample_gt_cp = copy.deepcopy(sample_gt)
        sample_gt_cp.update({
            'image_file': sample_file_name
        })
        annotation_list.append(sample_gt_cp)

    print(f'extract image number {len(annotation_list)}')

    with open(os.path.join(target_folder, annotation_file_name), 'w') as fp:
        json.dump(annotation_list, fp)
