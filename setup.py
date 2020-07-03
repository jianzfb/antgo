from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import os
import numpy as np
import shutil

# install: python setup.py build_ext install -r requirements.txt (from github)
ext_modules = [
    Extension('antgo.utils._mask',
              sources=['antgo/cutils/maskApi.c', 'antgo/utils/_mask.pyx'],
              include_dirs = [np.get_include(), 'antgo/cutils'],
              extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    ),
    Extension('antgo.utils._bbox',
              ["antgo/utils/_bbox.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args=["-Wno-cpp", "-Wno-unused-function", '-std=c99']
    ),
    Extension('antgo.utils._nms',
              ["antgo/utils/_nms.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args=["-Wno-cpp", "-Wno-unused-function", '-std=c99']
              ),
    Extension('antgo.utils._resize',
              ["antgo/utils/_resize.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args=["-Wno-cpp", "-Wno-unused-function"]
              ),
]

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='antgo',
      version='0.0.1',
      description='machine learning experiment platform',
      __short_description__='machine learning experiment platform',
      url='https://github.com/jianzfb/antgo',
      author='jian',
      author_email='jian@mltalker.com',
      packages=['antgo',
                'antgo.ant',
                'antgo.utils',
                'antgo.dataflow',
                'antgo.dataflow.dataset',
                'antgo.dataflow.datasynth',
                'antgo.dataflow.imgaug',
                'antgo.resource',
                'antgo.resource.templates',
                'antgo.resource.static',
                'antgo.resource.browser',
                'antgo.resource.browser.static',
                'antgo.resource.browser.static.css',
                'antgo.resource.browser.static.fonts',
                'antgo.resource.browser.static.img',
                'antgo.resource.browser.static.js',
                'antgo.crowdsource',
                'antgo.measures',
                'antgo.task',
                'antgo.trainer',
                'antgo.activelearning',
                'antgo.activelearning.samplingmethods',
                'antgo.cutils',
                'antgo.example',
                'antgo.annotation',
                'antgo.sandbox',
                'antgo.codebook',
                'antgo.codebook.tf'],
      ext_modules=cythonize(ext_modules),
      entry_points={'console_scripts': ['antgo=antgo.main:main'], },
      long_description=readme(),
      include_package_data=True,
      zip_safe=False,)

# config file
if not os.path.exists(os.path.join(os.environ['HOME'], '.config', 'antgo')):
  os.makedirs(os.path.join(os.environ['HOME'], '.config', 'antgo'))

if not os.path.exists(os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml')):
  shutil.copy(os.path.join('/'.join(os.path.realpath(__file__).split('/')[0:-1]),'antgo', 'config.xml'),
              os.path.join(os.environ['HOME'], '.config', 'antgo', 'config.xml'))
