# -*- coding: UTF-8 -*-
# @Time    : 2022/9/13 12:30
# @File    : utils.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import platform
import subprocess
import time
from pathlib import Path
import requests
import torch


def select_device(device=''):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    return torch.device('cuda:0' if cuda else 'cpu')


def torch_model_download(file, repo='jianzfb/antgo', file_type='.pt'):
    # Attempt file download if does not exist
    if not file.endswith(file_type):
        file = f'{file}{file_type}'

    file = file.lower()
    model_file = os.path.join(os.environ['HOME'], '.antgo', 'models', file)
    model_file = Path(model_file)
    if model_file.exists():
        return model_file

    response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()  # github api
    assets = [x['name'] for x in response['assets']]  # release assets
    tag = response['tag_name']  # i.e. 'v1.0'

    name = model_file.name
    if name in assets:
        msg = f'{model_file} missing, try downloading from https://github.com/{repo}/releases/'
        try:  # GitHub
            url = f'https://github.com/{repo}/releases/download/{tag}/{name}'
            print(f'Downloading {url} to {model_file}...')
            torch.hub.download_url_to_file(url, str(model_file))
            assert model_file.exists() and model_file.stat().st_size > 1E6  # check
        except Exception as e:  # GCP
            print(f'Download {name} error.')
        finally:
            if not model_file.exists() or model_file.stat().st_size < 1E6:  # check
                model_file.unlink(missing_ok=True)  # remove partial downloads
                print(f'ERROR: Download failure: {msg}')
            print('')
            return model_file