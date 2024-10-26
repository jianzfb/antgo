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
import antvis.client.mlogger as mlogger
import os
import requests
import time
import urllib.parse
import uuid
import json
import yaml


def download_from_baidu(target_folder, keys, src_path=None, target_num=10000, **kwargs):
    datasource_name = 'baidu'
    datasource_type = ''
    datasource_keyword = ''
    
    # type:image,keyword:k/k/,
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

    os.makedirs(target_folder, exist_ok=True)
    baidu_download(
        datasource_keyword,
        {'download_data_type': datasource_type}, 
        target_folder, target_num=target_num)


def download_from_bing(target_folder, keys, src_path=None, target_num=10000, **kwargs):
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

    os.makedirs(target_folder, exist_ok=True)  
    bing_download(
        datasource_keyword,
        {'download_data_type': datasource_type}, 
        target_folder, target_num=target_num)


def download_from_google(target_folder, keys, src_path=None, target_num=10000, **kwargs):
    FLAGS = flags.AntFLAGS

    logger.error("In coming")


def download_from_vcg(target_folder, keys, src_path=None, target_num=10000, **kwargs):
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

    os.makedirs(target_folder, exist_ok=True)  
    vcg_download(
        datasource_keyword, 
        {'download_data_type': datasource_type}, 
        target_folder, target_num=target_num)


def download_from_aliyun(target_folder, keys=None, src_path=None, **kwargs):
    if not src_path.startswith('ali://'):
        src_path = f'ali://{src_path}'

    ali = AliBackend()
    ali.download(src_path, target_folder)


def download_from_logger(target_folder, keys=None, src_path=None, **kwargs):
    # step 1: 检测当前路径下收否有token缓存
    token = None
    if os.path.exists('./.token'):
        with open('./.token', 'r') as fp:
            token = fp.readline()

    # step 2: 检查antgo配置目录下的配置文件中是否有token
    if token is None or token == '':
        config_xml = os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')
        config.AntConfig.parse_xml(config_xml)
        token = getattr(config.AntConfig, 'server_user_token', '')
    if token == '' or token is None:
        print('No valid vibstring token, directly return')
        return

    # 激活实验
    experiment_id, checkpoint_name = src_path[9:].split('/')

    mlogger.config(token=token)
    project_name = kwargs.get('project', os.path.abspath(os.path.curdir).split('/')[-1])
    status = mlogger.activate(project_name, experiment_id)
    if status is None:
        print(f'Couldnt find {project_name}/{experiment_id}, from logger platform')
        return

    local_checkpoint_path = None
    remote_checkpoint_path = None
    # 下载配置文件
    os.makedirs(target_folder, exist_ok=True)
    mlogger.FileLogger.cache_folder = target_folder
    # 下载checkpoint文件
    file_logger = mlogger.Container()
    file_logger.checkpoint_file = mlogger.FileLogger('file', None)
    file_list, remote_list = file_logger.checkpoint_file.get(checkpoint_name)

    for file_name, remote_info in zip(file_list, remote_list):
        if file_name.endswith(checkpoint_name):
            local_checkpoint_path = file_name
            remote_checkpoint_path = remote_info
            break

    return local_checkpoint_path