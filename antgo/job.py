# encoding=utf-8
# Time: 8/28/17
# File: job.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import uuid
import threading
import numpy as np
import re
import copy
import sys
from antgo.utils.encode import *
from antgo.utils import logger
import scipy.misc
import base64

PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    import Queue as queue
elif PYTHON_VERSION == 3:
    import queue as queue


class Chart():
  def __init__(self,title="chart", x_axis="x", y_axis="y"):
    self.chart_title = title
    self.chart_x_axis = x_axis
    self.chart_y_axis = y_axis
    self.chart_id = unicode(uuid.uuid1()) if PYTHON_VERSION == 2 else str(uuid.uuid1())
    self.chart_channels = []

  @property
  def id(self):
    return self.chart_id

  @property
  def title(self):
    return self.chart_title

  @property
  def x_axis(self):
    return self.chart_x_axis

  @property
  def y_axis(self):
    return self.chart_y_axis

  @property
  def channels_num(self):
    return len(self.chart_channels)

  def bind_channel(self, channel):
    channel.id = len(self.chart_channels)
    channel.chart = self
    self.chart_channels.append(channel)

  def clone(self):
    self.chart_id = unicode(uuid.uuid1()) if PYTHON_VERSION == 2 else str(uuid.uuid1())


class Channel():
  def __init__(self, channel_name = None, channel_type = None, channel_job=None, channel_params={}):
    self.channel_id = -1
    self.channel_name = channel_name
    self.channel_type = channel_type
    self.channel_chart = None
    self.channel_job = channel_job
    self.channel_params = channel_params
    assert(self.channel_type in ["IMAGE", "NUMERIC", "TEXT", "HISTOGRAM"])

  @property
  def params(self):
    return self.channel_params

  @property
  def chart(self):
    return self.channel_chart
  @chart.setter
  def chart(self,val):
    self.channel_chart = val

  @property
  def id(self):
    return self.channel_id
  @id.setter
  def id(self,val):
    self.channel_id = val

  @property
  def name(self):
    return self.channel_name

  def reorganize_data(self, data_type, data):
    if data_type == "IMAGE":
      return self.reorganize_image_data(data)
    elif data_type == "NUMERIC":
      return self.reorganize_numeric_data(data)
    elif data_type == "TEXT":
      return self.reorganize_text_data(data)
    elif data_type == "HISTOGRAM":
      return self.reorganize_histogram_data(data)
    else:
      return data

  def reorganize_image_data(self, data):
    data_x, data_y = data
    try:
      data_x = float(data_x)
    except:
      logger.error("Channel X Must be Scalar Data")
      return None

    try:
      if len(data_y.shape) != 2 and len(data_y.shape) != 3:
        logger.error("Channel Y Must be 2 or 3 Dimension")
        return None
      if len(data_y.shape) == 3:
        if data_y.shape[2] != 3:
          logger.error("Channel Y Must Possess 3 or 1 Channels")
          return None

      allowed_size = 50.0
      height, width = data_y.shape[:2]
      min_scale = allowed_size / np.minimum(height, width)

      new_height = int(height * min_scale)
      new_width = int(width * min_scale)
      resized_img = scipy.misc.imresize(data_y,(new_height, new_width))
      if resized_img.dtype == np.uint8:
        return (data_x, base64.b64encode(png_encode(resized_img)))

      max_val = np.max(resized_img.flatten())
      min_val = np.min(resized_img.flatten())
      if len(data_y.shape) == 3:
          resized_img = ((resized_img - np.tile(min_val, (1,1,3))) / np.tile(max_val, (1,1,3))) * 255
          resized_img = resized_img.astype(np.uint8)
      else:
          resized_img = (resized_img - min_val) / max_val * 255
          resized_img = resized_img.astype(np.uint8)

      return (data_x, base64.b64encode(png_encode(resized_img)))
    except:
      logger.error("Channel Y Must be Numpy Array")

  def reorganize_numeric_data(self, data):
    data_x, data_y = data
    try:
      data_x = float(data_x)
    except:
      logger.error("Channel X Must be Scalar Data")

    try:
      data_y = float(data_y)
    except:
      logger.error("Channel Y Must be Scalar Data")
    return (data_x, data_y)

  def reorganize_text_data(self, data):
    data_x, data_y = data
    try:
      data_x = float(data_x)
    except:
      logger.error("Channel X Must be Scalar Data")

    try:
      data_y = str(data_y)
    except:
      logger.error("Channel Y Must Could Transfer to String")
    return (data_x, data_y)

  def reorganize_histogram_data(self, data):
    data_x, data_y = data
    try:
      data_x = float(data_x)
    except:
      logger.error("Channel X Must be Scalar Data")

    try:
      data_y = data_y.flatten()
      bins = 10 # default bins
      if "BINS" in self.params:
        bins = self.params['BINS']

      data_y = np.histogram(data_y, bins)
    except:
      logger.error("Channel Y Must be Numpy Array")
    return (data_x, data_y)

  def send(self, x=0, y=0):
    # {"CHART", (chart_id, chart_title,...)}
    x_copy = copy.deepcopy(x)
    y_copy = copy.deepcopy(y)
    data = {"CHANNEL": self,
            "DATA": {"CHART": [self.chart.id,
                               self.chart.title,
                               self.chart.x_axis,
                               self.chart.y_axis,
                               self.chart.channels_num,
                               self.id,
                               self.channel_type,
                               self.channel_name,
                               x_copy,
                               y_copy]}}

    if self.channel_job != None:
      # send data
      self.channel_job.send(data)


class Job(threading.Thread):
  def __init__(self, context=None):
    super(Job, self).__init__()
    self.data_queue = queue.Queue()
    self.job_context = context
    self.setDaemon(True)

    self.charts = []

  def create_channel(self, channel_name, channel_type):
    return Channel(channel_name, channel_type, self)

  def create_chart(self, chart_channels, chart_title, chart_x_axis="x", chart_y_axis="y"):
    chart = Chart(chart_title, chart_x_axis, chart_y_axis)
    self.charts.append(chart)
    for cc in chart_channels:
        chart.bind_channel(cc)

  @property
  def context(self):
      return self.job_context

  def send(self, data):
    if data is None:
        return

    # running stage
    data["STAGE"] = self.context.stage
    self.data_queue.put(data)

  def stop(self):
    self.data_queue.put(None)

  def clone_charts(self):
    for chart in self.charts:
      chart.clone()

  def run(self):
    while True:
      # 0.step get data
      data = self.data_queue.get()

      # check whether stop thread
      if data is None:
          break

      # 1.step reorganize data
      job_stage = data.pop('STAGE')
      if 'CHANNEL' in data:
          job_channel = data['CHANNEL']
          chart_data = data['DATA']["CHART"]

          # reorganize (channel_type, channel_y)
          reorganized_xy= job_channel.reorganize_data(chart_data[6], [chart_data[8], chart_data[9]])
          if reorganized_xy is None:
            continue
          chart_data[8] = reorganized_xy[0]
          chart_data[9] = reorganized_xy[1]

          data['DATA']["CHART"] = chart_data

      # 2.step sending to mltalker
      if self.job_context != None and data['DATA'] != None:
          self.job_context.send(data['DATA'], job_stage)