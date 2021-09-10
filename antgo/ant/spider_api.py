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
import os
import requests
import time
import  urllib.parse
import uuid

def spider_api_baidu():
    FLAGS = flags.AntFLAGS

    datasource_name = 'baidu'
    datasource_type = ''
    datasource_keyword = ''
    for p in FLAGS.param().split(','):
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
    if not os.path.exists('./spider_%d'%(time_stamp)):
        os.makedirs('./spider_%d'%(time_stamp))
    baidu_download(datasource_keyword, {'download_data_type': datasource_type}, './spider_%d'%(time_stamp))

def spider_api_bing():
    FLAGS = flags.AntFLAGS

    datasource_name = 'bing'
    datasource_type = ''
    datasource_keyword = ''
    for p in FLAGS.param().split(','):
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
    if not os.path.exists('./spider_%d'%(time_stamp)):
        os.makedirs('./spider_%d'%(time_stamp))
    bing_download(datasource_keyword, {'download_data_type': datasource_type}, './spider_%d'%(time_stamp))

def spider_api_google():
    FLAGS = flags.AntFLAGS

    logger.error("In coming")

def spider_api_vcg():
    FLAGS = flags.AntFLAGS

    datasource_name = 'vcg'
    datasource_type = ''
    datasource_keyword = ''
    for p in FLAGS.param().split(','):
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
    if not os.path.exists('./spider_%d'%(time_stamp)):
        os.makedirs('./spider_%d'%(time_stamp))
    vcg_download(datasource_keyword, {'download_data_type': datasource_type}, './spider_%d'%(time_stamp))
