import os
from antgo.pipeline.extent import op
import logging
import pathlib

# 内置C++算子，需要opencv支持
# 需检测C++依赖环境（opencv,eigen)
ANTGO_DEPEND_ROOT = os.environ.get('ANTGO_DEPEND_ROOT', f'{str(pathlib.Path.home())}/.3rd')
if os.path.exists(ANTGO_DEPEND_ROOT):
    if os.path.exists(os.path.join(ANTGO_DEPEND_ROOT,'eigen')) and os.path.exists(os.path.join(ANTGO_DEPEND_ROOT, 'opencv-install')):
        for op_file_name in os.listdir(os.path.dirname(__file__)):
            if op_file_name[0] == '.':
                continue
            if op_file_name.startswith('_'):
                continue

            op.load(op_file_name, os.path.dirname(__file__))
else:
    logging.warning('Fail auto load build-in c++ ops')
