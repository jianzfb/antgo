import os
import sys
import copy
from typing import Any
import uuid
import pathlib
import numpy as np
import importlib
from antgo.pipeline.extent.op.loader import gen_code
import hashlib
from .build import build_eagleeye_env
ANTGO_DEPEND_ROOT = os.environ.get('ANTGO_DEPEND_ROOT', f'{str(pathlib.Path.home())}/.3rd')

class CompileOp(object):
    def __init__(self, func_op_name, func_op_file=None, folder=None, project_folder='', **kwargs):
        if '_' in func_op_name:
            a,b = func_op_name.split('_')
            func_op_name = f'{a.capitalize()}{b.capitalize()}'
        self.func_op_name = func_op_name        
        self.op_id = str(uuid.uuid4())
        self.func_op_file = func_op_file
        if folder is not None:
            self.func_op_file = os.path.join(folder, self.func_op_file)

        self.project_folder = project_folder
        self.include_dirs = []
        self.source_files = []
        self.finished_traverse_files = []
        md5code = ''
        with open(self.func_op_file, 'r') as fp:
            file_text = fp.read()
            md5code = hashlib.md5(file_text.encode()).hexdigest()

        is_need_compile = True
        os.makedirs('.temp', exist_ok=True)
        if os.path.exists(os.path.join('.temp/', f'{func_op_name}_md5.txt')):
            with open(os.path.join('.temp/', f'{func_op_name}_md5.txt'), 'r') as fp:
                old_md5code = fp.read()
                if md5code == old_md5code:
                    is_need_compile = False
        if is_need_compile:
            self.iteration_parse_includes(self.func_op_file)
            self.compile()
            with open(os.path.join('.temp/', f'{func_op_name}_md5.txt'), 'w') as fp:
                fp.write(md5code)

        self.param_1 = dict()   # {"key": [float,float,float,...]}
        self.param_2 = dict()   # {"key": ["","","",...]}
        self.param_3 = dict()   # {"key": [[float,float,...],[],...]}

        if 'params' in kwargs:
            params = kwargs.pop('params')
            kwargs.update(params)

        for var_key, var_value in kwargs.items():
            if isinstance(var_value, list) or isinstance(var_value, tuple):
                if len(var_value) > 0:
                    if isinstance(var_value[0], str):
                        self.param_2[var_key] = var_value
                    elif isinstance(var_value[0], list):
                        temp = np.array(var_value).astype(np.float32).tolist()
                        if len(temp.shape) == 1:
                            self.param_3[var_key] = temp
                    else:
                        self.param_1[var_key] = np.array(var_value).astype(np.float32).tolist()
            elif isinstance(var_value, np.ndarray):
                if len(var_value.shape) == 1:
                    self.param_1[var_key] = var_value.astype(np.float32).tolist()
                elif len(var_value.shape) == 2:
                    self.param_3[var_key] = var_value.astype(np.float32).tolist()
                else:
                    print(f'Dont support {var_key}')
                    print(var_value)
            elif isinstance(var_value, str):
                self.param_2[var_key] = var_value
            else:
                self.param_1[var_key] = [float(var_value)]

    def compile(self):
        source_file_str = ' '.join(self.source_files)
        include_dir_str = ' '.join([f'-I{f}' for f in self.include_dirs])
        opencv_include_dir = os.path.join(ANTGO_DEPEND_ROOT,'opencv-install/include')
        opencv_lib_dir = os.path.join(ANTGO_DEPEND_ROOT, 'opencv-install/lib')
        opencv_lib_link = '-lopencv_calib3d -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_tracking -lopencv_ximgproc -lopencv_xfeatures2d -lopencv_features2d -lopencv_stitching'

        eagleeye_include_dir = os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye', f'{sys.platform}-x86-64-install', 'include')
        eagleeye_lib_dir = os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye', f'{sys.platform}-x86-64-install', 'libs', 'x86-64')
        pybind_include_dir = os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye', f'{sys.platform}-x86-64-install', '3rd', 'pybind11', 'include')

        folder =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        antgo_include_dir = f'{folder}/extent/cpp/include/'
        warp_cpp_code_content = \
            gen_code('./templates/compile_op_code.cpp')(
                op_name=self.func_op_name,
                cls_name=self.func_op_name,
                include_file=self.func_op_file
            )
        with open(os.path.join(f'{self.func_op_name}Py.cpp'), 'w') as fp:
            fp.write(warp_cpp_code_content)

        compile_script = f'g++ -O3 -Wall -shared -std=c++14 -fPIC $(python3-config --includes) {source_file_str} {self.func_op_name}Py.cpp -o {self.func_op_name}Py$(python3-config --extension-suffix) -L{opencv_lib_dir} {opencv_lib_link} -L{eagleeye_lib_dir} -leagleeye -I{pybind_include_dir} -I{self.project_folder} -I{eagleeye_include_dir} -I{opencv_include_dir} -DTENSORRT_NN_ENGINE -I/root/.3rd/TensorRT-8.6.1.6/include -I/usr/local/cuda/include {include_dir_str} -I{antgo_include_dir}  -L/root/.3rd/TensorRT-8.6.1.6/lib/ -lnvinfer -lnvonnxparser -lnvparsers'
        print(compile_script)
        os.system(compile_script)

    def iteration_parse_includes(self, file_path, is_continue_check_cpp=True):
        file_folder = os.path.dirname(os.path.abspath(file_path))

        with open(file_path, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.strip()
                if line == '':
                    continue

                if line.startswith('#ifndef'):
                    line = line.replace(' ', '')
                    flag = line[len('#ifndef'):]
                    if flag in self.finished_traverse_files:
                        # 已经完成遍历
                        return None
                    self.finished_traverse_files.append(flag)

                if line.startswith('#include'):
                    info = line.replace(' ','').replace('#include','')
                    if info.startswith('<'):
                        # 默认系统库自动关联
                        continue
                    info = info[1:-1]
                    if info == 'defines.h' or info.startswith('pybind11'):
                        # 保留头文件，直接过滤
                        continue

                    if info.startswith('eagleeye'):
                        # 默认核心库自动关联
                        continue
                    if info.startswith('opencv'):
                        # opencv库自动关联
                        continue

                    # .h, .hpp，深入检索
                    if info.endswith('.h') or info.endswith('.hpp'):
                        # 搜索
                        if info.startswith('.') or '/' not in info:
                            if file_folder not in self.include_dirs:
                                self.include_dirs.append(file_folder)

                        if is_continue_check_cpp:
                            source_file = info
                            if info.endswith('.h'):
                                source_file = source_file[:-2]+'.cpp'
                            
                            if os.path.exists(os.path.join(self.project_folder, source_file)):
                                source_file = os.path.join(self.project_folder, source_file)
                            elif os.path.exists(os.path.join(self.project_folder, source_file.replace('/include/', '/src/'))):
                                source_file = os.path.join(self.project_folder, source_file.replace('/include/', '/src/'))
                            elif os.path.exists(os.path.join(file_folder, source_file.replace('/include/', '/src/'))):
                                source_file = os.path.join(file_folder, source_file.replace('/include/', '/src/'))
                            else:
                                source_file = os.path.join(file_folder, source_file)

                            # 允许source file not exist
                            if os.path.exists(source_file) and source_file not in self.source_files:
                                # self.iteration_parse_includes(source_file, False)
                                self.source_files.append(source_file)

                        traverse_file_path = info
                        # traverse_file_path.startswith('.') -> 与当前文件存在相对路径关系
                        # '/' not in traverse_file_path -> 表明为当前目录下的文件
                        if traverse_file_path.startswith('.') or '/' not in traverse_file_path:
                            traverse_file_path = os.path.join(file_folder, traverse_file_path)
                        else:
                            traverse_file_path = os.path.join(self.project_folder, traverse_file_path)

                        if not os.path.exists(traverse_file_path):
                            print(f'abnormal {traverse_file_path} not exists')
                            continue

                        self.iteration_parse_includes(traverse_file_path, True)
        return None

    def __call__(self, *args):
        input_tensors = []
        for tensor in args:
            if isinstance(tensor, str):
                print(f'Concert str {tensor} to numpy mode')
                tensor = np.frombuffer(tensor.encode('utf-8'), dtype=np.uint8)
            assert(isinstance(tensor, np.ndarray))
            input_tensors.append(tensor)

        modellib = importlib.import_module(f'{self.func_op_name}Py') 
        output_tensors = getattr(modellib, f'{self.func_op_name}Func')(self.op_id, self.func_op_name, self.param_1, self.param_2,self.param_3, input_tensors)
        return output_tensors if len(output_tensors) > 1 else output_tensors[0]
