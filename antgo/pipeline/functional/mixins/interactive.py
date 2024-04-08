# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 14:19
# @File    : interactive.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.pipeline.engine import *
from antgo.pipeline.functional.mixins.serve import _APIWrapper
import cv2
import os
import base64
import numpy as np
import requests
import imagesize


class InteractiveMixin:
	interactive_elements = {}
	# POLYGON, POINT, LINE, RECT
	def interactive_points(self, bind_src, bind_tgt, num=1, config={}):

		InteractiveMixin.interactive_elements[f'{_APIWrapper.tls.placeholder._name}/{bind_src}'] = {
			'mode': 'POINT',
			'num': num,
			'target': bind_tgt,
			'config': config
		}

		def inner():
			for x in self:
				if getattr(x, bind_tgt, None) is None:
					x(**{bind_tgt: None})
				yield x
		return self._factory(inner())

	def interactive_rects(self, bind_src, bind_tgt, num=1, config={}):
		InteractiveMixin.interactive_elements[f'{_APIWrapper.tls.placeholder._name}/{bind_src}'] = {
			'mode': 'RECT',
			'num': num,
			'target': bind_tgt,
			'config': config
		}

		def inner():
			for x in self:
				if getattr(x, bind_tgt, None) is None:
					x(**{bind_tgt: None})
				yield x
		return self._factory(inner())


	def interactive_lines(self, bind_src, bind_tgt, num=1, config={}):
		InteractiveMixin.interactive_elements[f'{_APIWrapper.tls.placeholder._name}/{bind_src}'] = {
			'mode': 'LINE',
			'num': num,
			'target': bind_tgt,
			'config': config
		}

		def inner():
			for x in self:
				if getattr(x, bind_tgt, None) is None:
					x(**{bind_tgt: None})
				yield x
		return self._factory(inner())
	
	def interactive_polygon(self, bind_src, bind_tgt, num=1, config={}):
		InteractiveMixin.interactive_elements[f'{_APIWrapper.tls.placeholder._name}/{bind_src}'] = {
			'mode': 'POLYGON',
			'num': num,
			'target': bind_tgt,
			'config': config
		}

		def inner():
			for x in self:
				if getattr(x, bind_tgt, None) is None:
					x(**{bind_tgt: None})
				yield x
		return self._factory(inner())