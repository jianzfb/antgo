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
import re
from openai import OpenAI


class PromptTemplate(object):
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template

    def contains_braced_string(self, text):
        """
        判断字符串中是否包含 {任意字符} 这样的模式。
        
        参数:
            text (str): 要检查的输入字符串。
            
        返回:
            bool: 如果找到匹配模式，返回 True；否则返回 False。
        """
        # 正则表达式解释:
        # \{ : 匹配左大括号 '{' (需要转义，因为 '{' 在正则表达式中有特殊含义)
        # .+ : 匹配至少一个 (或更多) 任意字符
        # \} : 匹配右大括号 '}' (需要转义)
        pattern = r"\{.+\}"
        
        # re.search() 在字符串中查找匹配模式。如果找到，返回一个匹配对象；否则返回 None。
        if re.search(pattern, text):
            return True
        else:
            return False    

    def format(self, **kwargs):
        if self.contains_braced_string(self.prompt_template):
            return self.prompt_template.format(**kwargs)
        
        return self.prompt_template

@register
class inference_openai_op(object):
    def __init__(self, base_url, model_name, api_key='dummy', prompt_template=None, max_tokens=500, temperature=None, post_process_func=None):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model_name = model_name
        self.prompt_template = PromptTemplate(prompt_template)
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
        kwargs = args[1] if len(args) > 1 else {}
        prompt_template = self.prompt_template.format(**kwargs)

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
