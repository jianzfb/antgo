from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np
import os
import subprocess

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

# To compile and install locally run "python setup.py build_ext --inplace"
# Install: pip install . -r requirements.txt (from pip)
# Install: python setup.py build_ext sdist install -r requirements.txt (from github)
ext_modules = [
    Extension(
        'antgo.utils._mask',
        sources=['antgo/cutils/maskApi.c', 'antgo/utils/_mask.pyx'],
        include_dirs = [np.get_include(), 'antgo/cutils'],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    ),
    Extension('antgo.utils._bbox',
              ["antgo/utils/_bbox.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args=["-Wno-cpp", "-Wno-unused-function",'-std=c99']
    ),
    Extension('antgo.utils._nms',
              ["antgo/utils/_nms.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args=["-Wno-cpp", "-Wno-unused-function", '-std=c99']
              ),
]

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='antgo',
      version='0.0.1',
      description='machine learning community',
      url='https://github.com/jianzfb/antgo',
      author='jian',
      author_email='jian.fbehind@gmail.com',
      packages=['antgo',
                'antgo.ant',
                'antgo.utils',
                'antgo.dataflow',
                'antgo.dataflow.dataset',
                'antgo.dataflow.datasynth',
                'antgo.dataflow.imgaug',
                'antgo.html',
                'antgo.html.templates',
                'antgo.measures',
                'antgo.task',
                'antgo.trainer',
                'antgo.cutils',
                'antgo.example',
                'antgo.annotation'],
      ext_modules=cythonize(ext_modules),
      entry_points={'console_scripts':['antgo=antgo.main:main'],},
      long_description=readme(),
      include_package_data=True,
      zip_safe=False,)

# config key folder
# 1.step config data_factory and task_factory
config_xml = os.path.join(os.path.split(os.path.realpath(__file__))[0],'antgo','config.xml')
tree = ET.ElementTree(file=config_xml)
root = tree.getroot()

data_factory = None
task_factory = None
for child in root:
  if child.tag == 'data_factory':
    data_factory = child.text.strip()
  elif child.tag == 'task_factory':
    task_factory = child.text.strip()

assert(data_factory is not None)
assert(task_factory is not None)

# make data_factory folder
print('construct data and task factory')
if not os.path.exists(data_factory):
  os.makedirs(data_factory)
  
# make task_factory folder
if not os.path.exists(task_factory):
  os.makedirs(task_factory)



# 2.step example data
# 2.1 step heart_scale
