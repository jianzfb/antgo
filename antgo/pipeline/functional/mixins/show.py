# -*- coding: UTF-8 -*-
# @Time    : 2022/9/15 23:19
# @File    : show.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.interactcontext import *
from antgo.pipeline.hparam import param_scope
import subprocess


class ShowMixin:
  def browser(self, show_type={}, title='antgo'):
    # 图片显示
    ctx = InteractContext()
    with ctx.Browser(title, {}, browser={'size': 0}) as hp:
      for index, info in enumerate(self):
        data = {}

        for k, v in show_type.items():
          data[k] = {
            'data': getattr(info, k),
            'type': v
          }

        data['id'] = index
        hp.context.recorder.record(data)

      hp.wait_until_stop()


  def rtsp(self, frame_key, fps=25, rtsp_ip='127.0.0.1', rtsp_port=8554, title='antgo'):
    # 视频流显示
    # step 1: 检查ffmpeg是否安装
    # apt update && sudo apt upgrade
    # apt install ffmpeg
    # dpkg -l ffmpeg
    ret = subprocess.Popen('dpkg -l ffmpeg', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    response = ret.stdout.read().decode('utf-8')
    if 'no packages found' in response:
      # 安装ffmpeg
      os.system('apt update && sudo apt upgrade')
      os.system('apt install ffmpeg')

    # step 2: 检查是否有推流服务器是否启动
    ret = subprocess.Popen(f'docker inspect rtsp-server', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    response = ret.stdout.read().decode('utf-8')
    if '[]\n' in response:
      # os.system(f'docker run -d --rm -it -e MTX_PROTOCOLS=tcp -e MTX_WEBRTCADDITIONALHOSTS={rtsp_ip} -p {rtsp_port}:8554 -p 1935:1935 -p 8888:8888 -p 8889:8889 -p 8890:8890/udp -p 8189:8189/udp --name rtsp-server bluenviron/mediamtx')
      os.system('docker run -d --rm -it --name rtsp-server --network=host bluenviron/mediamtx:latest')

    # step 3: 运行推流
    pipe = None
    for index, info in enumerate(self):
      frame = getattr(info, frame_key)
      if pipe is None:
        frame_height, frame_width = frame.shape[:2]
        command = ['ffmpeg',
            '-y', '-an',
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-pix_fmt', 'bgr24',        #像素格式
            '-s', "{}x{}".format(frame_width, frame_height),
            '-r', str(int(fps)),        # 自己的摄像头的fps是0，若用自己的notebook摄像头，设置为15、20、25都可。 
            '-i', '-',
            '-c:v', 'libx264',          # 视频编码方式
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-f', 'rtsp',               #  flv rtsp
            '-rtsp_transport', 'tcp',   # 使用TCP推流，linux中一定要有这行
            f'rtsp://{rtsp_ip}:8554/stream'] # rtsp rtmp  
        pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

      pipe.stdin.write(frame.tobytes())
      time.sleep(1.0/fps)


  def label(self, label_type='RECT', label_metas={}, title='antgo'):
    ctx = InteractContext()
    with ctx.Activelearning(title, {}, activelearning={
      'label_type': label_type,
      'label_metas': label_metas,
      'stage': 'labeling'
    }, clear=False) as activelearning:
      for index, info in enumerate(self):
        data = {
          'image': None,  # 第二优先级
          'label_info': [],
          'id': index
        }
        for k in info.__dict__.keys():
          data['image'] = getattr(info, k)
          break
        activelearning.context.recorder.record(data)

      activelearning.wait_until_stop()