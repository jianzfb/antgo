#encoding=utf-8
from __future__ import unicode_literals
from __future__ import division

import uuid
import threading
import numpy as np
from antgo.utils import logger
from antgo.utils.encode import *
import re
import copy
import sys

PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    import Queue as queue
elif PYTHON_VERSION == 3:
    import queue as queue


class Chart():
    def __init__(self,title="chart",x_axis="x",y_axis="y"):
        self.chart_title = title
        self.chart_x_axis = x_axis
        self.chart_y_axis = y_axis
        self.chart_id = str(uuid.uuid1())
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


class Channel():
    def __init__(self,channel_name = None,channel_type = None,channel_job=None,channel_params={}):
        self.channel_id = -1
        self.channel_name = channel_name
        self.channel_type = channel_type
        self.channel_chart = None
        self.channel_job = channel_job
        self.channel_params = channel_params
        assert(self.channel_type in ["IMAGE","NUMERIC","TEXT","HISTOGRAM","TABLE","STATISTIC","MONITOR"])

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

    def preprocess(self,data_type,data):
        if data_type == "IMAGE":
            return self.preprocess_image(data)
        elif data_type == "NUMERIC":
            return self.preprocess_numeric(data)
        elif data_type == "TEXT":
            return self.preprocess_text(data)
        elif data_type == "HISTOGRAM":
            return self.preprocess_histogram(data)
        elif data_type == "TABLE":
            return self.preprocess_table(data)
        elif data_type == "STATISTIC":
            return self.preprocess_statistic(data)
        elif data_type == "MONITOR":
            return self.preprocess_monitor(data)
        else:
            return data

    def preprocess_image(self,data):
        data_x,data_y = data
        try:
            data_x = float(data_x)
        except:
            logger.error("Channel X Must be Scalar Data")

        try:
            encoder_img = None
            after_data_y = None
            if len(data_y.shape) != 2 and len(data_y.shape) != 3:
                logger.error("Channel Y Must be 2 or 3 Dimension")

            if len(data_y.shape) == 3:
                img_3c = np.zeros((data_y.shape[0],data_y.shape[1],data_y.shape[2]))
                channles = data_y.shape[2]
                for ci in range(channles):
                    max_val = np.max(data_y[:,:,ci].flatten())
                    min_val = np.min(data_y[:,:,ci].flatten())
                    img_3c[:,:,ci] = (data_y[:,:,ci] - min_val) / max_val * 255.0
                img_3c = img_3c.astype(np.uint8)
                after_data_y = img_3c
            else:
                max_val = np.max(data_y.flatten())
                min_val = np.min(data_y.flatten())
                img_1c = (data_y - min_val) / max_val * 255.0
                img_1c = img_1c.astype(np.uint8)
                after_data_y = img_1c

            # convert to png
            compression = 5 # default compression
            if "COMPRESSION" in self.params:
                compression = self.params['COMPRESSION']

            encoder_img = png_encode(after_data_y)
            return (data_x, encoder_img)
        except:
            logger.error("Channel Y Must be Numpy Array")

    def preprocess_numeric(self,data):
        data_x,data_y = data
        try:
            data_x = float(data_x)
        except:
            logger.error("Channel X Must be Scalar Data")

        try:
            data_y = float(data_y)
        except:
            logger.error("Channel Y Must be Scalar Data")
        return (data_x,data_y)

    def preprocess_text(self,data):
        data_x,data_y = data
        try:
            data_x = float(data_x)
        except:
            logger.error("Channel X Must be Scalar Data")

        try:
            data_y = str(data_y)
        except:
            logger.error("Channel Y Must Could Transfer to String")
        return (data_x,data_y)

    def preprocess_histogram(self,data):
        data_x,data_y = data
        try:
            data_x = float(data_x)
        except:
            logger.error("Channel X Must be Scalar Data")

        try:
            data_y = data_y.flatten()
            bins = 10 # default bins
            if "BINS" in self.params:
                bins = self.params['BINS']

            min_val = float(np.min(data_y))
            max_val = float(np.max(data_y)) + 0.00000001
            step_val = (max_val - min_val) / bins

            data_y = np.histogram(data_y,bins)
        except:
            logger.error("Channel Y Must be Numpy Array")
        return (data_x,data_y)

    def preprocess_table(self,data):
        data_x,data_y = data
        try:
            data_x = float(data_x)
        except:
            logger.error("Channel X Must be Scalar Data")

        try:
            if len(data_y.shape) != 2:
                logger.error("Channel Y Shape Must be 2 Dimension")

            # transform to string
            width = data_y.shape[1]
            height = data_y.shape[0]
            yy = [[] for i in range(height)]
            for y in range(height):
                for x in range(width):
                    yy[y].append(str(int(data_y[y,x] * 100) / 100))
            data_y = yy
        except:
            logger.error("Channel Y Must be Numpy Array")

        return (data_x,data_y)

    def preprocess_statistic(self,data):
        data_x,data_y = data
        try:
            data_x = float(data_x)
        except:
            logger.error("Channel X Must be Scalar Data")

        try:
            mean_val = float(np.mean(data_y.flatten()))
            std_val = float(np.std(data_y.flatten()))
            data_y = (mean_val,std_val)
        except:
            logger.error("Channel Y Must be Numpy Array")
        return (data_x,data_y)

    def preprocess_monitor(self,data):
        return data

    def send(self,x = 0,y = 0):
        # {"CHART",(chart_id,chart_title,...)}
        x_copy = copy.deepcopy(x)
        y_copy = copy.deepcopy(y)
        data = {"CHANNEL":self,
                "DATA":{"CHART":[self.chart.id,self.chart.title,
                         self.chart.x_axis,self.chart.y_axis,
                         self.chart.channels_num,
                         self.id,self.channel_type,
                         self.channel_name,
                         x_copy,y_copy]}}

        # TODO: using fixed size queue
        if self.channel_job != None:
            # send data
            self.channel_job.send(data)

class Job(threading.Thread):
    def __init__(self,context=None):
        super(Job, self).__init__()
        self.data_queue = queue.Queue()
        self.job_context = context
        self.setDaemon(True)

    def create_channel(self,channel_name,channel_type):
        return Channel(channel_name,channel_type,self)

    def create_chart(self,chart_channels,chart_title,chart_x_axis="x",chart_y_axis="y"):
        chart = Chart(chart_title,chart_x_axis,chart_y_axis)
        for cc in chart_channels:
            chart.bind_channel(cc)

    @property
    def context(self):
        return self.job_context

    def send(self,data):
        # additional key info
        data["STAGE"] = self.context.stage
        self.data_queue.put(data)

    def stop(self):
        self.data_queue.put(None)

    def run(self):
        while True:
            # 0.step get data
            data = self.data_queue.get()

            # check whether stop thread
            if data is None:
                break

            # 1.step preprocess
            job_data = None
            job_stage = data.pop('STAGE')
            if 'CHANNEL' in data:
                # preprocess all data from channles
                job_channel = data['CHANNEL']
                job_data = data['DATA']
                # extract
                chart_data = job_data["CHART"]

                # preprocess (channel_type, channel_y)
                x,y = job_channel.preprocess(chart_data[6],[chart_data[8],chart_data[9]])
                chart_data[8] = x
                chart_data[9] = y

                # reorganize
                job_data["CHART"] = chart_data

            if job_data is None:
                # all data from no channels
                job_data = data

            # 2.step sending to mltalker
            if self.job_context != None and job_data != None:
                self.job_context.send(job_data,job_stage)