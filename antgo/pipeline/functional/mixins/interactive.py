# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 14:19
# @File    : interactive.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


from antgo.pipeline.engine import *
import cv2
import os
import base64
import numpy as np
import requests
import imagesize


class InteractiveMixin:
	interactive_elements = {}
	# POLYGON, POINT, LINE, RECT
	def interactive_points(self, bind_src, bind_tgt):
		InteractiveMixin.interactive_elements[bind_src] = {
			'mode': 'POINT',
			'target': bind_tgt,
		}

		def inner():
			for x in self:
				if getattr(x, bind_tgt, None) is None:
					x(**{bind_tgt: None})
				yield x
		return self._factory(inner())

	def interactive_rects(self, bind_src, bind_tgt):
		InteractiveMixin.interactive_elements[bind_src] = {
			'mode': 'RECT',
			'target': bind_tgt,
		}

		def inner():
			for x in self:
				if getattr(x, bind_tgt, None) is None:
					x(**{bind_tgt: None})
				yield x
		return self._factory(inner())


	def interactive_lines(self, bind_src, bind_tgt):
		InteractiveMixin.interactive_elements[bind_src] = {
			'mode': 'LINE',
			'target': bind_tgt,
		}

		def inner():
			for x in self:
				if getattr(x, bind_tgt, None) is None:
					x(**{bind_tgt: None})
				yield x
		return self._factory(inner())
	
	def interactive_polygon(self, bind_src, bind_tgt):
		InteractiveMixin.interactive_elements[bind_src] = {
			'mode': 'POLYGON',
			'target': bind_tgt,
		}

		def inner():
			for x in self:
				if getattr(x, bind_tgt, None) is None:
					x(**{bind_tgt: None})
				yield x
		return self._factory(inner())