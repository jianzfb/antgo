# -*- coding: UTF-8 -*-
# @Time    : 2020-06-25 23:30
# @File    : browser_server.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import copy
import numpy as np
import tornado.httpserver
import tornado.ioloop
import tornado.options
from tornado.options import define, options
from tornado import web, gen
from tornado import httpclient
import tornado.web
import os
import shutil
import signal
from antgo.crowdsource.base_server import *
from antgo.utils import logger
from antgo.ant import environment
import traceback
import sys
import uuid
import json
import time
import base64
import requests
import cv2
import torch
mano_layer_map = None


# # 新用户，从队列中获取新数据并加入用户队列中
# self.db['data'].append({
#   'value': self.response_queue.get(),
#   'status': False,
#   'time': time.time()
# })

class FreshApiHandler(BaseHandler):
  @gen.coroutine
  def post(self):
    # 添加样本记录
    samples = self.get_argument('samples', None)
    if samples is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    # 解析json
    samples = json.loads(samples)

    is_replace = self.get_argument('is_replace', None)
    if is_replace:
      self.db['data'] = []

    for sample_i, sample in enumerate(samples):
      for data in sample:
        if data['type'] == 'IMAGE':
          large_size = max(data['width'], data['height'])
          scale = large_size / 400
          if scale > 1.0:
            data['width'] = (int)(data['width'] / scale)
            data['height'] = (int)(data['height'] / scale)

      self.db['data'].append({
        'value': sample,
        'status': False,
        'time': time.time()
      })

    self.response(RESPONSE_STATUS_CODE.SUCCESS)


class EntryApiHandler(BaseHandler):
  @gen.coroutine
  def get(self, *args, **kwargs):
    user = self.get_current_user()
    if user is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    # 检查是否有输入，如果有先下载，在更新数据库
    input_info = self.get_argument('input', '')
    if input_info != '':
      # step 1: 下载 
      if input_info.startswith('http') and input_info.endswith('json'):
        # step 1.1: 下载 (from url)
        try:
          static_path = self.settings.get('static_path')
          if not input_info.endswith('.json'):
            self.response(RESPONSE_STATUS_CODE.RESOURCE_NOT_FOUND)
            return
          input_ext_name = input_info.split('.')[-1]          
          local_path = os.path.join(static_path, f'download.{input_ext_name}')
          
          data = requests.get(input_info, timeout=7)
          with open(local_path, 'wb') as fp:
            fp.write(data.content)     
          
          input_info = local_path
        except:
          logger.error('Fail to download from html.')
          self.response(RESPONSE_STATUS_CODE.EXECUTE_FORBIDDEN)
          return
      elif input_info.lower().startswith("hdfs") and input_info.endswith('json'):
        # step 1.2: 下载 (from htfs, jsonfile)
        try:
          static_path = self.settings.get('static_path')
          environment.hdfs_client.get(input_info, static_path)
          file_name = input_info.split('/')[-1]
          local_path = os.path.join(static_path, file_name)
          if not os.path.exists(local_path):
            logger.error('Fail to download from hdfs.')
            self.response(RESPONSE_STATUS_CODE.EXECUTE_FORBIDDEN)
            return
          
          input_info = local_path
        except:
          logger.error('Fail to download from hdfs.')
          self.response(RESPONSE_STATUS_CODE.EXECUTE_FORBIDDEN)
          return
      elif input_info.lower().startswith("hdfs"):
        # step 1.3: 下载（from htfs, folder)
        try:
          static_path = self.settings.get('static_path')          
          sub_folder = 'json-folder'
          if os.path.exists(os.path.join(static_path, sub_folder)):
            shutil.rmtree(os.path.join(static_path, sub_folder))
          os.makedirs(os.path.join(static_path, sub_folder))
          
          temp_path = os.path.join(static_path, sub_folder)
          if input_info.endswith('/'):
            input_info = input_info[:-1]            
          environment.hdfs_client.get(f'{input_info}/*', temp_path)

          json_file_path = ''
          if os.path.isdir(temp_path):
            for file_name in os.listdir(temp_path):
              if file_name.endswith('json'):
                json_file_path = os.path.join(temp_path, file_name)
                shutil.copyfile(json_file_path, os.path.join(static_path, file_name))
                json_file_path = os.path.join(static_path, file_name)
                break
          
          assert(json_file_path != '')
          input_info = json_file_path
        except:
          logger.error('Fail to download folder from hdfs.')
          self.response(RESPONSE_STATUS_CODE.EXECUTE_FORBIDDEN)
          return
      else:
        # step 1.4: 本地路径
        if not os.path.exists(input_info):
          self.response(RESPONSE_STATUS_CODE.RESOURCE_NOT_FOUND)
          return
        
      # step 2: 更新数据库
      self.update_db(self.settings.get('static_path'), input_info)
    
    if len(self.db['data']) == 0:
      # 当前无数据，直接返回
      self.response(RESPONSE_STATUS_CODE.RESOURCE_NOT_FOUND)
      return

    user_name = user['full_name']
    if user_name not in self.db['user_record']:
      self.db['user_record'][user_name] = []
    if len(self.db['user_record'][user_name]) == 0:
      self.db['user_record'][user_name].append(0)

    # 获得当前用户浏览位置
    entry_id = self.db['user_record'][user_name][-1]

    # 构建返回数据
    try:
      response_content = {
        'value': update_vis_elem(self.db['data'][entry_id]['value']),
        'step': len(self.db['user_record'][user_name]) - 1,
        'tags': self.settings.get('tags', []),
        'operators': [],
        'dataset_flag': self.db['state'],
        'samples_num': self.db['dataset'][self.db['state']]['samples_num'],
        'samples_num_checked': self.db['dataset'][self.db['state']]['samples_num_checked'],
      }
    except Exception:
      print(traceback.format_exc())
      self.response(RESPONSE_STATUS_CODE.INTERNAL_SERVER_ERROR)
      return

    self.response(RESPONSE_STATUS_CODE.SUCCESS, content=response_content)
    return

  def update_db(self, dump_dir, json_file_path):
    # 1.step 清空数据
    self.db['data'] = []
    
    # 2.step 解析下载数据（json, or tar）
    if not os.path.exists(json_file_path):
      return
    if os.path.isdir(json_file_path):
      for file_name in os.listdir(json_file_path):
        if file_name.endswith('json'):
          json_file_path = os.path.join(json_file_path, file_name)
          break
              
    sample_anno_json_file = json_file_path
    sample_list = []
    # step2.1: 加载样本信息
    # 兼容两种常用的存储样本信息方式（1.纯json格式，所有样本以list形式存储；2.样本按行存储，每个样本的信息是json格式）
    try:
      # try 1 (纯json格式，所有样本以list形式存储)
      with open(sample_anno_json_file, 'r', encoding="utf-8") as fp:
        sample_list = json.load(fp)
    except:
      # try 2 (样本信息按行存储，每个样本信息是json格式)
      with open(sample_anno_json_file, 'r', encoding="utf-8") as fp:
        sample_info_str = fp.readline()
        sample_info_str = sample_info_str.strip()
        
        while sample_info_str:
          sample_info = json.loads(sample_info_str)
          sample_list.append(sample_info)
          sample_info_str = fp.readline()
          sample_info_str = sample_info_str.strip()
          if sample_info_str == '':
            break        
          
    # step2.2: 尝试加载样本集meta信息
    sample_meta = {'meta': {}}
    sample_folder = os.path.dirname(sample_anno_json_file)
    data_meta_file = os.path.join(sample_folder, 'meta.json')
    if os.path.exists(data_meta_file):
      with open(data_meta_file, 'r', encoding="utf-8") as fp:
        sample_meta = json.load(fp)
            
    # 3.step 加载数据
    sample_folder = os.path.abspath(sample_folder)  # 将传入的路径转换为绝对路径
    static_dir = self.settings.get('static_path')
    if sample_list is not None and sample_folder is not None:
      # 为样本所在目录建立软连接到static下面
      if not sample_folder.startswith(static_dir):
        # 建立软连接
        os.system(f'cd {static_dir}; ln -s {sample_folder} dataset;')
      
      # 将数据信息写如本地数据库
      # 1.step 尝试加载meta信息
      meta_info = sample_meta
      # 2.step 样本信息写入数据库
      for sample_id, sample in enumerate(sample_list):
        # 仅考虑image_url
        data_source = sample['image_url']
        file_name = data_source
        sample_label = ''
        if 'image_label_name' in sample:
          sample_label = sample['image_label_name']
        elif 'image_label' in sample:
          sample_label = sample['image_label']
 
        convert_sample = [{
          'type': 'IMAGE',
          'data': data_source,
          'tag': [sample_label],
          'title': sample['image_url'],
          'id': sample_id
        }]
        
        #############################  标准信息  ###############################
        # 添加框元素
        if 'bboxes' in sample:
          convert_sample[0].update({
            'bboxes': sample['bboxes']
          })
        # 添加多边形元素
        if 'segments' in sample:
            convert_sample[0].update({
            'segments': sample['segments']
          })          
        # 添加框标签信息
        if 'labels' in sample:
          convert_sample[0].update({
            'labels': sample['labels']
          })
        # 添加样本标签
        if 'image_label' in sample:
            convert_sample[0].update({
            'image_label': sample['image_label']
          })
            
        #############################  扩展信息 1  #############################
        # 添加2d关键点元素
        if 'joints2d' in sample:
          convert_sample[0].update({
            'joints2d': sample['joints2d'],
            'skeleton': meta_info['meta']['skeleton'] if 'skeleton' in meta_info['meta'] else []
          })
        
        #############################  扩展信息 2  #############################
        # 添加3d关键点元素（需要相机参数）
        if 'joints3d' in sample and \
          'cam_param' in sample and len(sample['cam_param']) > 0:
          convert_sample.append({
            'type': 'IMAGE',
            'data': data_source,
            'tag': [sample_label],
            'title': f'3D-POINTS-({file_name})',
            'id': sample_id
          })
          convert_sample[-1].update({
            'joints3d': sample['joints3d'],
            'skeleton': meta_info['meta']['skeleton'] if 'skeleton' in meta_info['meta'] else [],
            'cam_param': sample['cam_param']
          })
        
        #############################  扩展信息 3  #############################    
        # 添加3d关键点元素（mano）(需要相机参数)
        if 'pose' in sample and 'trans' in sample and 'shape' in sample and \
          'cam_param' in sample and len(sample['cam_param']) > 0 and \
          'model' in meta_info['meta']:
          
          pose_shape_model = meta_info['meta']['model']
          convert_sample.append({
            'type': 'IMAGE',
            'data': data_source,
            'tag': [sample_label],
            'title': f'{pose_shape_model}-({file_name})',
            'id': sample_id
          })
          convert_sample[-1].update({
            'skeleton': meta_info['meta']['skeleton'] if 'skeleton' in meta_info['meta'] else [],
            'labels': sample['labels'] if 'labels' in sample else [],
            'category': meta_info['meta']['category'] if 'category' in meta_info['meta'] else {},            
            'pose': sample['pose'],
            'shape': sample['shape'],
            'trans': sample['trans'],
            'cam_param': sample['cam_param'],
            'model': pose_shape_model
          })          
                
        self.db['data'].append({
          'value': convert_sample,
          'status': False,
          'time': time.time()
        })


def update_vis_elem(data_group):
  data_group = copy.deepcopy(data_group)
  for data in data_group:
    if 'joints3d' in data and 'cam_param' in data:
      # 基于相机模型，进行投影
      if 'D' in data['cam_param'] and 'Xi' in data['cam_param'] and 'K' in data['cam_param']:
        # 鱼眼模型(omni)
        # points_3d: Nx21x3
        K = np.array(data['cam_param']['K']).astype(np.float32)
        Xi = float(data['cam_param']['Xi'])
        D = np.array(data['cam_param']['D']).astype(np.float32)
        points_2d_list = []
        for i in range(len(data['joints3d'])):
          points_3d = np.array(data['joints3d'][i]).astype(np.float32)
          points_3d = points_3d.reshape(points_3d.shape[0],1,3).astype(np.float32)
          uv,_ = cv2.omnidir.projectPoints(points_3d, np.zeros(3), np.zeros(3), K, Xi, D)
          points_2d = uv[:,0,:].tolist()
          points_2d_list.append(points_2d)
        
        # update elem
        data.update({
          'joints2d': points_2d_list
        })
      elif 'K' in data['cam_param']:
        # 小孔成像模型
        # points_3d: Nx21x3
        K = np.array(data['cam_param']['K']).astype(np.float32)
        points_2d_list = []
        for i in range(len(data['joints3d'])):
          points_3d = np.array(data['joints3d'][i])
          points_2d = np.matmul(K, np.transpose(points_3d))
          points_2d = np.transpose(points_2d).tolist()
          points_2d_list.append(points_2d)
        
        # update elem
        data.update({
          'joints2d': points_2d_list
        })
    elif 'model' in data and 'cam_param' in data:
      # 基于物体模型（mano,smpl）->绘制3D点->绘制2D点
      global mano_layer_map
      if data['model'] == 'mano' and mano_layer_map is not None:
        try:
          category_names = []
          if len(data['labels']) > 0 and len(data['category']) > 0:
            # left or right
            category_names = [data['category'][str(label)] for label in data['labels']]
                    
          points_2d_list = []
          for i in range(len(data['pose'])):
            category_name = category_names[i]
            mano_layer = mano_layer_map[category_name]
            pose = np.array(data['pose'][i])
            shape = np.array(data['shape'][i])
            trans = np.array(data['trans'][i])
            _, joints_fk = mano_layer.forward(
                torch.tensor(pose.reshape(1, -1), dtype=torch.float32),
                torch.tensor(shape.reshape(1,-1), dtype=torch.float32),
                torch.tensor(trans.reshape(1,-1), dtype=torch.float32)
            )
            joints_fk = 0.001 * joints_fk[0].numpy()    
            if 'D' in data['cam_param'] and 'Xi' in data['cam_param'] and 'K' in data['cam_param']:
              K = np.array(data['cam_param']['K']).astype(np.float32)
              Xi = float(data['cam_param']['Xi'])
              D = np.array(data['cam_param']['D']).astype(np.float32)
              
              joints_fk = joints_fk.reshape(joints_fk.shape[0],1,3).astype(np.float32)
              uv,_ = cv2.omnidir.projectPoints(joints_fk, np.zeros(3), np.zeros(3), K, Xi, D)
              points_2d = uv[:,0,:].tolist()
              points_2d_list.append(points_2d)
            elif 'K' in data['cam_param']:
              points_2d = np.matmul(K, np.transpose(joints_fk))
              points_2d = np.transpose(points_2d).tolist()
              points_2d_list.append(points_2d)
            
          #update elem
          data.update({
            'joints2d': points_2d_list
          })      
        except:
          logger.error('Couldnt process by mano')
    
  return data_group


class PrevApiHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    user = self.get_current_user()
    if user is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    step = self.get_argument('step', None)
    data = self.get_argument("data", None)
    if step is None or data is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'request parameter wrong')
      return

    user_name = user['full_name']
    if user_name not in self.db['user_record']:
      self.db['user_record'] = []
    step = int(step)
    if step < 0 or step >= len(self.db['user_record'][user_name]):
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'request parameter wrong')
      return

    # 保存当前修改到内存
    data = json.loads(data)
    entry_id = self.db['user_record'][user_name][step]
    self.db['data'][entry_id] = {'value': data, 'status': True, 'time': time.time()}

    # 保存当前修改到文件
    # 使用 offset
    state = self.db['state']
    offset = self.db['dataset'][state]['offset']
    if not os.path.exists(os.path.join(self.dump_folder, state)):
      os.makedirs(os.path.join(self.dump_folder, state))

    try:
      # 发现数据id
      data_id = None
      for item in data:
        if item['title'] == 'ID':
          data_id = str(item['data'])
      if data_id is None:
        data_id = str(entry_id+offset)

      with open(os.path.join(self.dump_folder, state, '%s.json'%data_id), "w") as file_obj:
        json.dump(data, file_obj)
    except Exception as e:
      print('str(Exception):\t', str(Exception))
      print('str(e):\t\t', str(e))
      print('repr(e):\t', repr(e))
      print('e.message:\t', e.message)

    # 获得当前用户上一步数据
    if step == 0:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'dont have pre step')
      return

    pre_step = step - 1
    pre_entry_id = self.db['user_record'][user_name][pre_step]
    
    try:
      response_content = {
        'value': update_vis_elem(self.db['data'][pre_entry_id]['value']),
        'step': pre_step,
        'tags': self.settings.get('tags', []),
        'operators': [],
        'state': self.db['state'],
        'samples_num': self.db['dataset'][self.db['state']]['samples_num'],
        'samples_num_checked': self.db['dataset'][self.db['state']]['samples_num_checked'],
      }
    except Exception:
      print(traceback.format_exc())
      self.response(RESPONSE_STATUS_CODE.INTERNAL_SERVER_ERROR)      
      return
    
    self.response(RESPONSE_STATUS_CODE.SUCCESS, content=response_content)


class NextApiHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    user = self.get_current_user()
    if user is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    step = self.get_argument("step", None)
    data = self.get_argument("data", None)
    if step is None or data is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'request parameter wrong')
      return

    user_name = user['full_name']
    if user_name not in self.db['user_record']:
      self.db['user_record'] = []
    step = int(step)
    if step < 0 or step >= len(self.db['user_record'][user_name]):
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'request parameter wrong')
      return

    # 保存当前修改到内存
    data = json.loads(data)
    entry_id = self.db['user_record'][user_name][step]
    if not self.db['data'][entry_id]['status']:
      self.db['dataset'][self.db['state']]['samples_num_checked'] += 1

    self.db['data'][entry_id] = {'value': data, 'status': True, 'time': time.time()}

    # 保存当前修改到文件
    # 使用 offset
    state = self.db['state']
    offset = self.db['dataset'][state]['offset']
    if not os.path.exists(os.path.join(self.dump_folder, state)):
      os.makedirs(os.path.join(self.dump_folder, state))

    try:
      # 发现数据id
      data_id = None
      for item in data:
        assert('id' in item or 'ID' in item)
        if 'id' in item:
          data_id = str(item['id'])
        if 'ID' in item:
          data_id = str(item['ID'])

      if data_id is None:
        data_id = str(entry_id + offset)

      with open(os.path.join(self.dump_folder, state, '%s.json' % data_id), "w") as fp:
        json.dump(data, fp)
    except Exception as e:
      print('str(Exception):\t', str(Exception))
      print('str(e):\t\t', str(e))
      print('repr(e):\t', repr(e))
      print('e.message:\t', e.message)

    # 获得用户下一步数据
    if step < len(self.db['user_record'][user_name]) - 1:
      next_step = step + 1
      next_entry_id = self.db['user_record'][user_name][next_step]
      try:
        response_content = {
          'value': update_vis_elem(self.db['data'][next_entry_id]['value']),
          'step': next_step,
          'tags': self.settings.get('tags', []),
          'operators': [],
          'state': self.db['state'],
          'samples_num': self.db['dataset'][self.db['state']]['samples_num'],
          'samples_num_checked': self.db['dataset'][self.db['state']]['samples_num_checked'],
        }
      except Exception:
        print(traceback.format_exc())
        self.response(RESPONSE_STATUS_CODE.INTERNAL_SERVER_ERROR)
        return
      
      self.response(RESPONSE_STATUS_CODE.SUCCESS, content=response_content)
      return

    # 发下下一个还没有进行审核的样本
    next_entry_id = -1
    for id in range(len(self.db['data'])):
      if 'status' not in self.db['data'][id] or not self.db['data'][id]['status']:
        next_entry_id = id
        break

    if next_entry_id == -1:
      try:
        self.response(RESPONSE_STATUS_CODE.SUCCESS, content={
          'value': update_vis_elem(self.db['data'][self.db['user_record'][user_name][-1]]['value']),
          'step': len(self.db['user_record'][user_name]) - 1,
          'tags': self.settings.get('tags', []),
          'operators': [],
          'state': self.db['state'],
          'samples_num': self.db['dataset'][self.db['state']]['samples_num'],
          'samples_num_checked': self.db['dataset'][self.db['state']]['samples_num_checked'],
        })
      except Exception:
        print(traceback.format_exc())
        self.response(RESPONSE_STATUS_CODE.INTERNAL_SERVER_ERROR)
      return

    # 为当前用户分配下一个审查数据
    self.db['user_record'][user_name].append(next_entry_id)

    #
    try:
      response_content = {
        'value': update_vis_elem(self.db['data'][next_entry_id]['value']),
        'step': len(self.db['user_record'][user_name]) - 1,
        'tags': self.settings.get('tags', []),
        'operators': [],
        'state': self.db['state'],
        'samples_num': self.db['dataset'][self.db['state']]['samples_num'],
        'samples_num_checked': self.db['dataset'][self.db['state']]['samples_num_checked'],
      }
    except Exception:
      print(traceback.format_exc())
      self.response(RESPONSE_STATUS_CODE.INTERNAL_SERVER_ERROR)
      return

    self.response(RESPONSE_STATUS_CODE.SUCCESS, content=response_content)
    return
  
  
class RandomApiHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    user = self.get_current_user()
    if user is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    step = self.get_argument("step", None)
    data = self.get_argument("data", None)
    if step is None or data is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'request parameter wrong')
      return

    user_name = user['full_name']
    if user_name not in self.db['user_record']:
      self.db['user_record'] = []
    step = int(step)
    if step < 0 or step >= len(self.db['user_record'][user_name]):
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'request parameter wrong')
      return

    # 保存当前修改到内存
    data = json.loads(data)
    entry_id = self.db['user_record'][user_name][step]
    if not self.db['data'][entry_id]['status']:
      self.db['dataset'][self.db['state']]['samples_num_checked'] += 1

    self.db['data'][entry_id] = {'value': data, 'status': True, 'time': time.time()}

    # 保存当前修改到文件
    # 使用 offset
    state = self.db['state']
    offset = self.db['dataset'][state]['offset']
    if not os.path.exists(os.path.join(self.dump_folder, state)):
      os.makedirs(os.path.join(self.dump_folder, state))

    try:
      # 发现数据id
      data_id = None
      for item in data:
        assert('id' in item or 'ID' in item)
        if 'id' in item:
          data_id = str(item['id'])
        if 'ID' in item:
          data_id = str(item['ID'])

      if data_id is None:
        data_id = str(entry_id + offset)

      with open(os.path.join(self.dump_folder, state, '%s.json' % data_id), "w") as fp:
        json.dump(data, fp)
    except Exception as e:
      print('str(Exception):\t', str(Exception))
      print('str(e):\t\t', str(e))
      print('repr(e):\t', repr(e))
      print('e.message:\t', e.message)

    ################  随机找寻一个样本  ##################
    accessed_ids =  self.db['user_record'][user_name]
    candidate_ids = [i for i in range(len(self.db['data'])) if i not in accessed_ids]
    if len(candidate_ids) == 0:
      # 已经没有未访问的数据
      candidate_ids = [i for i in range(len(self.db['data']))]
    next_entry_id = int(np.random.choice(candidate_ids))
    self.db['user_record'][user_name].append(next_entry_id)

    #
    try:
      response_content = {
        'value': update_vis_elem(self.db['data'][next_entry_id]['value']),
        'step': len(self.db['user_record'][user_name]) - 1,
        'tags': self.settings.get('tags', []),
        'operators': [],
        'state': self.db['state'],
        'samples_num': self.db['dataset'][self.db['state']]['samples_num'],
        'samples_num_checked': self.db['dataset'][self.db['state']]['samples_num_checked'],
      }
    except Exception:
      print(traceback.format_exc())
      self.response(RESPONSE_STATUS_CODE.INTERNAL_SERVER_ERROR)      
      return 

    self.response(RESPONSE_STATUS_CODE.SUCCESS, content=response_content)
    return


class FileApiHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    # 获得代码目录结构
    root_path = self.data_folder
    files_tree = [{
      'name': ".",
      'type': "folder",
      'size': "",
      'folder': [],
      'path': ''
    }]
    queue = [files_tree[-1]]
    while len(queue) != 0:
      folder = queue.pop(-1)
      folder_path = os.path.join(root_path, folder['path'])
      for f in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, f)):
          folder['folder'].append({
            'name': f,
            'type': "folder",
            'size': "",
            'folder': [],
            'path': '%s/%s' % (folder['path'], f) if folder['path'] != '' else f,
          })

          queue.append(folder['folder'][-1])
        else:
          fsize = os.path.getsize(os.path.join(folder_path, f))
          fsize = fsize / 1024.0  # KB
          folder['folder'].append({
            'name': f,
            'type': "file",
            'size': "%0.2f KB" % round(fsize, 2),
            'path': '%s/%s' % (folder['path'], f) if folder['path'] != '' else f,
          })

    response = {
      'files_tree': files_tree
    }
    self.response(RESPONSE_STATUS_CODE.SUCCESS, content=response)


class DownloadApiHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    if not os.path.exists(os.path.join(self.dump_folder, self.db['state'])):
      self.finish()
      return
    
    package_data = {}
    for file_name in os.listdir(os.path.join(self.dump_folder, self.db['state'])):
      if not file_name.endswith('json'):
        continue
      
      data_id = file_name.split('.')[0]
      with open(os.path.join(self.dump_folder, self.db['state'], file_name), 'r') as fp:
        package_data[data_id] = json.load(fp)
              
    now_time = time.strftime("%Y-%m-%dx%H:%M:%S", time.localtime(time.time()))
    download_file = f'{now_time}.json'
    download_path = os.path.join(self.dump_folder, download_file)
    with open(download_path, 'w') as fp:
      json.dump(package_data, fp)

    # Content-Type
    self.set_header('Content-Type', 'application/octet-stream')
    self.set_header('Content-Disposition', 'attachment; filename=' + download_file)

    buffer_size = 64 * 1024
    with open(download_path, 'rb') as fp:
      content = fp.read()
      content_size = len(content)
      buffer_segments = content_size // buffer_size
      for buffer_seg_i in range(buffer_segments):
        buffer_data = content[buffer_seg_i * buffer_size: (buffer_seg_i + 1) * buffer_size]
        yield self.write(buffer_data)

      yield self.write(content[buffer_segments * buffer_size:])

    self.finish()


class OperatorApiHandler(BaseHandler):
  def post(self):
    self.response(RESPONSE_STATUS_CODE.SUCCESS)


class ConfigApiHandler(BaseHandler):
  def post(self):
    offset_config = self.get_argument('offset_config', None)
    if offset_config is not None:
      offset_config = json.loads(offset_config)
      dataset_flag = offset_config['dataset_flag']
      dataset_offset = offset_config['dataset_offset']
      if 'dataset' not in self.db:
        self.db['dataset'] = {}
      if dataset_flag not in self.db['dataset']:
        self.db['dataset'][dataset_flag] = {}

      self.db['dataset'][dataset_flag]['offset'] = dataset_offset

    profile_config = self.get_argument('profile_config', None)
    if profile_config is not None:
      profile_config = json.loads(profile_config)
      dataset_flag = profile_config['dataset_flag']
      samples_num = profile_config['samples_num']
      samples_num_checked = profile_config['samples_num_checked']

      if 'dataset' not in self.db:
        self.db['dataset'] = {}
      if dataset_flag not in self.db['dataset']:
        self.db['dataset'][dataset_flag] = {}

      self.db['dataset'][dataset_flag]['samples_num'] = samples_num
      self.db['dataset'][dataset_flag]['samples_num_checked'] = samples_num_checked

    state = self.get_argument('state', None)
    if state is not None:
      # train,val or test
      self.db['state'] = state

class LoginHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    user_name = self.get_argument('user_name', None)
    user_password = self.get_argument('user_password', None)
    if user_name is None or user_password is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    white_users = self.db['white_users']
    if white_users is None:
      # 无需登录
      self.response(RESPONSE_STATUS_CODE.SUCCESS)
      return

    if user_name not in white_users:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    if user_password != self.db['white_users'][user_name]['password']:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    cookie_id = str(uuid.uuid4())
    self.db['users'].update({
      cookie_id: {
        "full_name": user_name,
        'short_name': user_name[0].upper(),
      }
    })
    self.set_login_cookie({'cookie_id': cookie_id})
    self.response(RESPONSE_STATUS_CODE.SUCCESS)


class LogoutHandler(BaseHandler):
  @gen.coroutine
  def post(self):
    self.clear_cookie('antgo')


class UserInfoHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    # 获取当前用户名
    user = self.get_current_user()
    if user is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    # 遍历所有样本获得本轮，当前用户的标注记录
    statistic_info = {
    }

    self.response(RESPONSE_STATUS_CODE.SUCCESS, content={
      'user_name': user['full_name'],
      'short_name': user['short_name'],
      'task_name': 'DEFAULT',
      'task_type': 'DEFAULT',
      'project_type': 'BROWSER',
      'statistic_info': statistic_info
    })


class ProjectInfoHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    
    self.response(RESPONSE_STATUS_CODE.SUCCESS, content={
      'project_type': 'BROWSER',
      'project_state': {
        'stage': \
          'finish' if len(self.db['data']) > 0 and \
            self.db['dataset'][self.db['state']]['samples_num_checked'] == len(self.db['data'])  else 'checking',
        'need_input': self.settings.get('need_input', False),
      },      
    })
    return

class PingApiHandler(BaseHandler):
  @gen.coroutine
  def get(self, *args, **kwargs):
    self.response(RESPONSE_STATUS_CODE.SUCCESS)


class GracefulExitException(Exception):
  @staticmethod
  def sigterm_handler(signum, frame):
    raise GracefulExitException()


def browser_server_start(browser_dump_dir,
                         tags,
                         server_port,
                         offset_configs,
                         profile_config,
                         sample_folder=None, sample_list=None, sample_meta={}, need_input=False,
                         white_users=None):
  # register sig
  signal.signal(signal.SIGTERM, GracefulExitException.sigterm_handler)

  # local ext module
  global mano_layer_map
  try:
    from manopth.manolayer import ManoLayer
    left_mano_layer = ManoLayer(
      mano_root="extra/models/mano", use_pca=False, flat_hand_mean=False, side='left'
    )   
    right_mano_layer = ManoLayer(
      mano_root="extra/models/mano", use_pca=False, flat_hand_mean=False, side='right'
    )   
    mano_layer_map = {
      'left': left_mano_layer,
      'right': right_mano_layer
    }
  except Exception:
    print(traceback.format_exc())
    logger.warn('Couldnt load mano model.')


  # 0.step define http server port
  define('port', default=server_port, help='run on port')

  try:
    # 静态资源目录
    browser_dir = os.path.join(browser_dump_dir, 'browser')
    static_dir = os.path.join(browser_dir, 'static')
    if not os.path.exists(static_dir):
      os.makedirs(static_dir)

    # 数据数据目录
    browser_dump_dir = os.path.join(browser_dump_dir, 'record')
    if not os.path.exists(browser_dump_dir):
      os.makedirs(browser_dump_dir)

    # 2.step launch show server
    db = {'data': [], 'users': {}, 'dataset': {}, 'user_record': {}}

    if sample_list is not None and sample_folder is not None:
      # 为样本所在目录建立软连接到static下面
      sample_folder = os.path.abspath(sample_folder)  # 将传入的路径转换为绝对路径    
      os.system(f'cd {static_dir}; ln -s {sample_folder} dataset;')
      
      # 将数据信息写如本地数据库
      # 1.step 尝试加载meta信息
      meta_info = sample_meta
      if len(meta_info) == 0:
        meta_info.update({
          'meta': {}
        })
      # 2.step 样本信息写入数据库
      for sample_id, sample in enumerate(sample_list):
        file_name = sample['image_file'] if sample['image_file'] != '' else sample['image_url']
        data_source = ''
        if sample['image_url'] != '':
          data_source = sample['image_url']
        else:
          data_source = f'/static/dataset/{sample["image_file"]}'

        sample_label = ''
        if 'image_label_name' in sample:
          sample_label = sample['image_label_name']
        elif 'image_label' in sample:
          sample_label = sample['image_label']        
        convert_sample = [{
          'type': 'IMAGE',
          'data': data_source,
          'tag': [sample_label],
          'title': file_name,
          'id': sample_id
        }]
        
        #############################  标准信息  ###############################
        # 添加框元素
        if 'bboxes' in sample:
          convert_sample[0].update({
            'bboxes': sample['bboxes']
          })
        # 添加多边形元素
        if 'segments' in sample:
            convert_sample[0].update({
            'segments': sample['segments']
          })          
        # 添加框标签信息
        if 'labels' in sample:
          convert_sample[0].update({
            'labels': sample['labels']
          })
        # 添加样本标签
        if 'image_label' in sample:
            convert_sample[0].update({
            'image_label': sample['image_label']
          })
            
        #############################  扩展信息 1  #############################
        # 添加2d关键点元素
        if 'joints2d' in sample:
          convert_sample[0].update({
            'joints2d': sample['joints2d'],
            'skeleton': meta_info['meta']['skeleton'] if 'skeleton' in meta_info['meta'] else []
          })
        
        #############################  扩展信息 2  #############################
        # 添加3d关键点元素（需要相机参数）
        if 'joints3d' in sample and \
          'cam_param' in sample and len(sample['cam_param']) > 0:
          convert_sample.append({
            'type': 'IMAGE',
            'data': data_source,
            'tag': [sample_label],
            'title': f'3D-POINTS-({file_name})',
            'id': sample_id
          })
          convert_sample[-1].update({
            'joints3d': sample['joints3d'],
            'skeleton': meta_info['meta']['skeleton'] if 'skeleton' in meta_info['meta'] else [],
            'cam_param': sample['cam_param']
          })
        
        #############################  扩展信息 3  #############################    
        # 添加3d关键点元素（mano）(需要相机参数)
        if 'pose' in sample and 'trans' in sample and 'shape' in sample and \
          'cam_param' in sample and len(sample['cam_param']) > 0 and \
          'model' in meta_info['meta']:
          
          pose_shape_model = meta_info['meta']['model']
          convert_sample.append({
            'type': 'IMAGE',
            'data': data_source,
            'tag': [sample_label],
            'title': f'{pose_shape_model}-({file_name})',
            'id': sample_id
          })
          convert_sample[-1].update({
            'joints3d': sample['joints3d'],
            'skeleton': meta_info['meta']['skeleton'] if 'skeleton' in meta_info['meta'] else [],
            'labels': sample['labels'] if 'labels' in sample else [],
            'category': meta_info['meta']['category'] if 'category' in meta_info['meta'] else {},
            'pose': sample['pose'],
            'shape': sample['shape'],
            'trans': sample['trans'],
            'cam_param': sample['cam_param'],
            'model': pose_shape_model
          })          
                
        db['data'].append({
          'value': convert_sample,
          'status': False,
          'time': time.time()
        })

    # 设置白盒用户
    db['white_users'] = white_users

    for offset_config in offset_configs:
      db['dataset'][offset_config['dataset_flag']] = {
        'offset': offset_config['dataset_offset']
      }

    db['dataset'][profile_config['dataset_flag']]['samples_num'] = \
      profile_config['samples_num']
    db['dataset'][profile_config['dataset_flag']]['samples_num_checked'] = \
      profile_config['samples_num_checked']
    db['state'] = profile_config['dataset_flag']

    # cookie
    cookie_secret = base64.b64encode(uuid.uuid4().bytes + uuid.uuid4().bytes)

    settings = {
      'static_path': os.path.join(browser_dir, 'static'),
      'dump_path': browser_dump_dir,
      'port': server_port,
      'tags': tags,
      'need_input': need_input,
      'cookie_secret': cookie_secret,
      'cookie_max_age_days': 30,
      'Content-Security-Policy': "frame-ancestors 'self' {}".format('http://localhost:8080/'),
      'db': db,
    }

    app = tornado.web.Application(handlers=[(r"/antgo/api/browser/sample/prev/", PrevApiHandler),
                                            (r"/antgo/api/browser/sample/next/", NextApiHandler),
                                            (r"/antgo/api/browser/sample/random/", RandomApiHandler),
                                            (r'/antgo/api/browser/sample/fresh/', FreshApiHandler),
                                            (r"/antgo/api/browser/sample/entry/", EntryApiHandler),
                                            (r"/antgo/api/browser/operators/", OperatorApiHandler),
                                            (r"/antgo/api/browser/file/", FileApiHandler),
                                            (r"/antgo/api/browser/download/", DownloadApiHandler),
                                            (r"/antgo/api/browser/config/", ConfigApiHandler),
                                            (r"/antgo/api/ping/", PingApiHandler),
                                            (r"/antgo/api/user/login/", LoginHandler),    # 登录，仅支持预先指定用户
                                            (r"/antgo/api/user/logout/", LogoutHandler),  # 退出
                                            (r"/antgo/api/user/info/", UserInfoHandler),  # 获得用户信息
                                            (r"/antgo/api/info/", ProjectInfoHandler),
                                            (r'/(.*)', tornado.web.StaticFileHandler,
                                             {"path": static_dir, "default_filename": "index.html"}),],
                                  **settings)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)

    logger.info('browser is providing server on port %d' % server_port)
    tornado.ioloop.IOLoop.instance().start()
    logger.info('browser stop server')
  except GracefulExitException:
    logger.info('browser server exit')
    sys.exit(0)
  except KeyboardInterrupt:
    pass


if __name__ == '__main__':
  data_path = ''
  browser_dump_dir = '/Users/jian/Downloads/BB'
  tags = ['A','B','D']
  server_port = 9000
  offset_configs = [{
    'dataset_flag': 'TRAIN',
    'dataset_offset': 0
  }, {
    'dataset_flag': 'VAL',
    'dataset_offset': 0
  }, {
    'dataset_flag': 'TEST',
    'dataset_offset': 0
  }]
  profile_config = {
    'dataset_flag': 'TRAIN',
    'samples_num': 10,
    'samples_num_checked': 0
  }
  white_users = {
    'jian@baidu.com':{
      'password': '112233'
    }
  }
  samples=[]
  for _ in range(10):
    temp = [
      {
        'type': 'IMAGE',
        'data': '/static/data/1.jpeg',
        'width': 1200//4,
        'height': 800//4,
        'tag': ['A'],
        'title': 'MIAO'
      },
      {
        'type': 'STRING',
        'data': 'AABBCC',
        'tag': ['B'],
        'title': 'H'
      }
    ]

    samples.append(copy.deepcopy(temp))

  browser_server_start(
    browser_dump_dir,
    tags,
    server_port,
    offset_configs,
    profile_config,
    white_users=white_users,
    sample_list=samples, sample_folder='')
