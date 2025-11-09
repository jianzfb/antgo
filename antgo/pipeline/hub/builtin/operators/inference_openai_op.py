# -*- coding: UTF-8 -*-
# @Time    : 2025/7/5 22:42
# @File    : inference_openai_op.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.pipeline.engine import *
import logging
import os
import numpy as np
import cv2
import base64
from openai import OpenAI


@register
class inference_openai_op(object):
    def __init__(self, base_url, model_name, api_key='dummy', prompt_template=None, max_tokens=500, temperature=None, post_process_func=None):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens
        self.post_process_func = post_process_func
        self.temperature = temperature

    def __call__(self, *args):
        image = args[0]
        if image is None:
            return None

        # 图片编码成webp（需要保证图片）
        _, encoded_image = cv2.imencode('.webp', image)
        # 图片编码成base64
        base64_image = base64.b64encode(encoded_image).decode('utf-8')

        # 构建prompt
        prompt_template = None
        if len(args) > 1:
            prompt_template = self.prompt_template
        if prompt_template is None:
            prompt_template = self.prompt_template
        if prompt_template is None:
            logging.error('LLM model need prompt input')
            return None

        # 构建图片输入
        mime_type = 'image/webp'
        image_url = f"data:{mime_type};base64,{base64_image}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_template},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ]

        kwargs_config = {
            'model': self.model_name,
            'messages': messages,
            'max_tokens': self.max_tokens,
        }
        if self.temperature is not None:
            kwargs_config.update({
                'temperature': self.temperature
            })

        response = self.client.chat.completions.create(**kwargs_config)
        raw_output = response.choices[0].message.content 
        if self.post_process_func is not None:
            raw_output = self.post_process_func(raw_output)
        return raw_output
