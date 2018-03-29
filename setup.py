from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np
import os
import subprocess

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
                'antgo.crowdsource',
                'antgo.measures',
                'antgo.task',
                'antgo.trainer',
                'antgo.cutils',
                'antgo.example',
                'antgo.annotation',
                'antgo.sandbox',
                'antgo.codebook',
                'antgo.codebook.tf',
                'antgo.tools'],
      ext_modules=cythonize(ext_modules),
      entry_points={'console_scripts': ['antgo=antgo.main:main'], },
      long_description=readme(),
      include_package_data=True,
      zip_safe=False,)