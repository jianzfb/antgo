# -*- coding: UTF-8 -*-
# @Time    : 2020/10/26 10:26 上午
# @File    : download.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.utils import logger
from bs4 import BeautifulSoup
import os
import sys
import cv2
import numpy as nps
import time
import threading
import requests
import re
import uuid
import os
try:
    import queue
except:
    import Queue as queue


def baidu_download(keyward, download_params, save_dir, process_queue=None):
    headers = {
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0',
        'Upgrade-Insecure-Requests': '1'
    }
    A = requests.Session()
    A.headers = headers
    url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + keyward + '&pn='

    def __baidu_find_and_download(waiting_process_queue, search_url, session, dir, max_page_num=50):
        t = 0
        num = 0
        while t < max_page_num:
            Url = search_url + str(t)
            t = t+1
            try:
                Result = session.get(Url, timeout=7, allow_redirects=False)
            except BaseException:
                t = t + 60
                continue
            else:
                pic_url = re.findall('"objURL":"(.*?)",', Result.text, re.S)  # 先利用正则表达式找到图片url
                for each in pic_url:
                    logger.info("Downloading(%d) %s."%(num+1, str(each)))
                    try:
                        if each is not None:
                            pic = requests.get(each, timeout=7)
                        else:
                            continue
                    except BaseException:
                        logger.error("Couldnt download %s."%each)
                        continue
                    else:
                        # 分配唯一文件标识
                        file_folder = os.path.join(dir, 'test')
                        if not os.path.exists(file_folder):
                            os.makedirs(file_folder)
                        
                        file_path = os.path.join(file_folder, 'baidu_%s.jpg'%str(uuid.uuid4()))
                        with open(file_path, 'wb') as fp:
                            fp.write(pic.content)
                        num += 1

                        logger.info("Finish download %s ."% str(each))
                        # 加入等待处理队列
                        if waiting_process_queue is not None:
                            waiting_process_queue.put(file_path)

        # 添加结束标记
        if waiting_process_queue is not None:
            waiting_process_queue.put(None)

    # 搜索和下载
    if process_queue is not None:
        t = threading.Thread(target=__baidu_find_and_download, args=(process_queue, url, A, save_dir))
        t.start()
    else:
        __baidu_find_and_download(process_queue, url, A, save_dir)


def bing_download(keyward, download_params, save_dir, process_queue=None):
    headers = {
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0',
        'Upgrade-Insecure-Requests': '1'
    }
    A = requests.Session()
    A.headers = headers
    #url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + keyward + '&pn='
    url = 'http://cn.bing.com/images/async?q='+keyward+'&first={0}&count=35&relp=35&lostate=r&mmasync=1&dgState=x*175_y*848_h*199_c*1_i*106_r*0'

    def __bing_find_and_download(waiting_process_queue, search_url, session, dir, max_page_num=50):
        t = 0
        num = 0
        while t < max_page_num:
            Url = search_url.format(t*35+1)

            t = t+1
            try:
                Result = session.get(Url, timeout=7, allow_redirects=False)
            except BaseException:
                t = t + 60
                continue
            else:
                pic_url = re.findall('src="(.*?)"', Result.text, re.S)  # 先利用正则表达式找到图片url
                for each in pic_url:
                    logger.info("Downloading(%d) %s."%(num+1, str(each)))
                    try:
                        if each is not None:
                            pic = requests.get(each, timeout=7)
                        else:
                            continue
                    except BaseException:
                        logger.error("Couldnt download %s."%each)
                        continue
                    else:
                        # 分配唯一文件标识
                        file_folder = os.path.join(dir, 'test')
                        if not os.path.exists(file_folder):
                            os.makedirs(file_folder)
                        
                        file_path = os.path.join(file_folder, 'bing_%s.jpg'%str(uuid.uuid4()))
                        with open(file_path, 'wb') as fp:
                            fp.write(pic.content)
                        num += 1
                        logger.info("Finish download %s ."% str(each))

                        # 加入等待处理队列
                        if waiting_process_queue is not None:
                            waiting_process_queue.put(file_path)
        
        # 结束标记
        if waiting_process_queue is not None:
            waiting_process_queue.put(None)

    # 搜索和下载
    if process_queue is not None:
        t = threading.Thread(target=__bing_find_and_download, args=(process_queue, url, A, save_dir))
        t.start()
    else:
        __bing_find_and_download(process_queue, url, A, save_dir)


def google_download(keyward, download_params, save_dir, process_queue=None):
    pass


def vcg_download(keyword, download_params, download_save_dir, process_queue=None):
    # 视觉中国
    headers = {
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0',
        'Upgrade-Insecure-Requests': '1'
    }
    download_data_type = 'image'
    if download_params is not None:
        download_data_type = download_params.get('download_data_type', 'image')
        if download_data_type not in ['image','video']:
            logger.warn('Only support donwload data_type: image or video')
            download_data_type = 'image'

    def __vcg_img_download(waiting_process_queue, save_dir, img_url, keyword, count):
        try:
            logger.info("Downloading(%d) %s."%(count+1, img_url))
            pic = requests.get(img_url, timeout=7)
        except BaseException:
            logger.error("Couldnt download %s."%img_url)
            return
        else:
            file_prefix = 'VCG_'+keyword + '_' + str(count)
            file_name = file_prefix + '.jpg' if download_data_type == 'image' else file_prefix + '.mp4'
            file_path = os.path.join(save_dir,  file_name)
            fp = open(file_path, 'wb')
            fp.write(pic.content)
            fp.close()
            logger.info("Finish download %s ."%img_url)
            
            if waiting_process_queue is not None:
                waiting_process_queue.put(file_path)

    def __vcg_try_get(A, page_url, try_times=5):
        while try_times != 0:
            Result = A.get(page_url, timeout=7, allow_redirects=True)
            if Result.status_code == 200:
                return Result

            logger.warn("sleep 10s, continue try.")
            time.sleep(10)
            try_times -= 1
        
        return None

    def __vcg_find_and_download(waiting_process_queue, save_dir):
        A = requests.Session()
        A.headers = headers
        query_url = 'https://www.vcg.com/creative-image/{}'.format(keyword)
        if download_data_type == 'video':
            query_url = 'https://www.vcg.com/creative-video-search/{}'.format(keyword)

        download_count = 0
        content = None
        nav_page_list = []
        Result = A.get(query_url, timeout=7, allow_redirects=True)
        if Result.status_code != 200:
            logger.error("%s couldnt connect."%query_url)
            return

        content = Result.text
        soup = BeautifulSoup(content)
        # 1.step 分析待检索页面
        if download_data_type == 'image':
            pages = soup.findAll("a", class_="paginationClickBox")
            last_page = pages[-1]

            total_page = soup.findAll("span", class_="paginationTotal")
            try:
                total_page_num = int(last_page.text)
                if len(total_page) > 0:
                    total_page_num = (int)(total_page[0].text[1:-1])
            except:
                logger.error("Fail to parse nav page, use default 2.")
                total_page_num = 2

            page_nav_url = last_page.get('href').split('?')[0]
            for i in range(2, total_page_num+1):
                nav_page_list.append('%s?page=%d'%(page_nav_url, i))
        else:
            pages = soup.findAll("a", class_="_2IlL4")
            last_page = pages[-2]
            try:
                total_page_num = int(last_page.text)
            except:
                logger.error("Fail to parse nav page, use default 2.")
                total_page_num = 2

            page_nav_url = last_page.get('href').split('?')[0]
            for i in range(2, total_page_num+1):
                nav_page_list.append('%s?page=%d'%(page_nav_url, i))

        # 2.step 分析当前页面的图像
        logger.info("Analyze nav page(%d/%d) %s"%(1, len(nav_page_list), query_url))
        img_url_list = []
        if download_data_type == 'image':
            # 分析图像资源
            img_list = soup.findAll("img", class_="lazyload_hk")
            img_url_list = ["http://{}".format(p.get("data-src")[2:]) for p in img_list]
        else:
            # 分析视频资源
            img_list = soup.findAll('source', type="image/webp")
            img_url_list = []
            video_name_list = []
            for p in img_list:
                kk = p.get('data-srcset')[2:].split('/')[3].split('?')[0].split('_')[0]
                if kk not in video_name_list:
                    video_name_list.append(kk)
                    img_url_list.append('http://gossv.cfp.cn/videos/mts_videos/medium/temp/{}.mp4'.format(kk))

        for img_url in img_url_list:
            __vcg_img_download(waiting_process_queue, save_dir, img_url, keyword, download_count)
            download_count += 1

        # 3.step 继续分析所有等待导航页面
        for page_index, page_url in enumerate(nav_page_list):
            logger.info("Analyze nav page(%d/%d) %s"%(page_index+2, len(nav_page_list), page_url))
            Result = A.get(page_url, timeout=7, allow_redirects=True)
            if Result.status_code != 200:
                logger.warn("Couldnt connect and analyze %s. (page %s)"%((Result.text, page_url)))
                Result = __vcg_try_get(A, page_url, 5)
                if Result is None:
                    logger.warn("Couldnt connect %s, return."%page_url)
                    return

            content = Result.text
            soup = BeautifulSoup(content)

            img_url_list = []
            if download_data_type == 'image':
                # 分析图像资源
                img_list = soup.findAll("img", class_="lazyload_hk")
                img_url_list = ["http://{}".format(p.get("data-src")[2:]) for p in img_list]
            else:
                # 分析视频资源
                img_list = soup.findAll('source', type="image/webp")
                img_url_list = []
                video_name_list = []
                for p in img_list:
                    kk = p.get('data-srcset')[2:].split('/')[3].split('?')[0].split('_')[0]
                    if kk not in video_name_list:
                        video_name_list.append(kk)
                        img_url_list.append('http://gossv.cfp.cn/videos/mts_videos/medium/temp/{}.mp4'.format(kk))

            logger.info("Finding %d data"%len(img_url_list))
            for img_url in img_url_list:
                __vcg_img_download(waiting_process_queue, save_dir, img_url, keyword, download_count)
                download_count += 1

        # 添加结束标记
        if waiting_process_queue is not None:
            waiting_process_queue.put(None)

    # 搜索和下载
    if not os.path.exists(os.path.join(download_save_dir, 'test')):
        os.makedirs(os.path.join(download_save_dir, 'test'))

    if process_queue is not None:
        t = threading.Thread(target=__vcg_find_and_download, args=(process_queue, os.path.join(download_save_dir, 'test')))
        t.start()
    else:
        __vcg_find_and_download(process_queue, os.path.join(download_save_dir, 'test'))


