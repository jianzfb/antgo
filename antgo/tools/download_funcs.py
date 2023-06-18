# -*- coding: UTF-8 -*-
# @Time    : 2020/10/26 10:26 上午
# @File    : spider_api.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.utils import logger
from antgo.ant.download import *
from antgo.ant import flags
from antgo.framework.helper.fileio.file_client import *
import os
import requests
import time
import  urllib.parse
import uuid


def download_from_baidu(target_folder, keys, src_path=None):
    datasource_name = 'baidu'
    datasource_type = ''
    datasource_keyword = ''
    
    # key:image,keyword:k/k/,
    for p in keys.split(','):
        k,v = p.split(":")     
        if k == 'type':
            datasource_type = v
        elif k == 'keyword':
            datasource_keyword = v
            # 替换/为,
            datasource_keyword = datasource_keyword.replace('/', ',')

    if datasource_name not in ['baidu', 'google', 'bing', 'vcg']:
        logger.error('Only support datasource baidu/google/bing/vcg')
        return
    
    if datasource_type not in ['image', 'video']:
        logger.error('Only support datasource type image/video')
        return
    
    if datasource_keyword == '':
        logger.error('Must set keyword')
        return

    if target_folder is None:
        target_folder = './'

    time_stamp = (int)(time.time())
    if not os.path.exists(os.path.join(target_folder, 'spider_%d'%(time_stamp))):
        os.makedirs(os.path.join(target_folder, 'spider_%d'%(time_stamp)))
    baidu_download(
        datasource_keyword,
        {'download_data_type': datasource_type}, 
        os.path.join(target_folder, 'spider_%d'%(time_stamp)))


def download_from_bing(target_folder, keys, src_path=None):
    datasource_name = 'bing'
    datasource_type = ''
    datasource_keyword = ''
    for p in keys.split(','):
        k,v = p.split(":")    
        if k == 'type':
            datasource_type = v
        elif k == 'keyword':
            datasource_keyword = v
            # 替换/为,
            datasource_keyword = datasource_keyword.replace('/', ',')

    if datasource_name not in ['baidu', 'google', 'bing', 'vcg']:
        logger.error('Only support datasource baidu/google/bing/vcg')
        return
    
    if datasource_type not in ['image', 'video']:
        logger.error('Only support datasource type image/video')
        return
    
    if datasource_keyword == '':
        logger.error('Must set keyword')
        return

    time_stamp = (int)(time.time())
    if not os.path.exists(os.path.join(target_folder, 'spider_%d'%(time_stamp))):
        os.makedirs(os.path.join(target_folder, 'spider_%d'%(time_stamp)))        
    bing_download(
        datasource_keyword,
        {'download_data_type': datasource_type}, 
        os.path.join(target_folder, 'spider_%d'%(time_stamp)))


def download_from_google(target_folder, keys, src_path=None):
    FLAGS = flags.AntFLAGS

    logger.error("In coming")


def download_from_vcg(target_folder, keys, src_path=None):
    datasource_name = 'vcg'
    datasource_type = ''
    datasource_keyword = ''
    for p in keys.split(','):
        k,v = p.split(":")    
        if k == 'type':
            datasource_type = v
        elif k == 'keyword':
            datasource_keyword = v
            # 替换/为,
            datasource_keyword = datasource_keyword.replace('/', ',')            

    if datasource_name not in ['baidu', 'google', 'bing', 'vcg']:
        logger.error('Only support datasource baidu/google/bing/vcg')
        return
    
    if datasource_type not in ['image', 'video']:
        logger.error('Only support datasource type image/video')
        return
    
    if datasource_keyword == '':
        logger.error('Must set keyword')
        return

    time_stamp = (int)(time.time())
    if not os.path.exists(os.path.join(target_folder, 'spider_%d'%(time_stamp))):
        os.makedirs(os.path.join(target_folder, 'spider_%d'%(time_stamp))) 
    vcg_download(
        datasource_keyword, 
        {'download_data_type': datasource_type}, 
        os.path.join(target_folder, 'spider_%d'%(time_stamp)))


def download_from_aliyun(target_folder, keys=None, src_path=None):
    if not src_path.startswith('ali://'):
        src_path = f'ali://{src_path}'

    ali = AliBackend()
    ali.download(src_path, target_folder)
