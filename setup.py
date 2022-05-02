from setuptools import setup
from setuptools import dist
from distutils.extension import Extension
from os import path as os_path
import os
import shutil

this_directory = os_path.abspath(os_path.dirname(__file__))
dist.Distribution().fetch_build_eggs(['numpy'])

# install: python setup.py build_ext install -r requirements.txt (from github)
# install: pip install .
def ext_modules():
    import numpy as np

    # some_extention = Extension(..., include_dirs=[np.get_include()])
    some_extention = [
      Extension('antgo.utils._mask',
                sources=['antgo/cutils/maskApi.c', 'antgo/utils/_mask.pyx'],
                include_dirs=[np.get_include(), 'antgo/cutils'],
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
                )
    ]

    return some_extention



def readme():
    with open('README.rst') as f:
        return f.read()


def read_file(filename):
    with open(os_path.join(this_directory, filename), encoding='utf-8') as f:
        long_description = f.read()
    return long_description


def read_requirements(filename):
    return [line.strip() for line in read_file(filename).splitlines()
            if not line.startswith('#')]


setup(name='antgo',
      version='0.0.19',
      description='machine learning experiment platform',
      __short_description__='machine learning experiment platform',
      url='https://github.com/jianzfb/antgo',
      author='jian',
      author_email='jian@mltalker.com',
      setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0',
        'Cython',
      ],
      packages=['antgo',
                'antgo.ant',
                'antgo.utils',
                'antgo.utils.shared_queue',
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
                'antgo.resource.batch',
                'antgo.resource.batch.static',
                'antgo.resource.batch.static.css',
                'antgo.resource.batch.static.fonts',
                'antgo.resource.batch.static.img',
                'antgo.resource.batch.static.js',
                'antgo.crowdsource',
                'antgo.measures',
                'antgo.task',
                'antgo.trainer',
                'antgo.activelearning',
                'antgo.activelearning.samplingmethods',
                'antgo.cutils',
                'antgo.annotation',
                'antgo.sandbox',
                'antgo.framework.paddle2torch',
                'antgo.framework.paddle2torch.mapper',
                'antgo.framework.paddle2torch.tools',
                'antgo.framework.paddle2torch.vision',
                'antgo.framework.torch2paddle',
                'antgo.framework.torch2paddle.mapper',
                'antgo.framework.torch2paddle.mapper.cuda',
                'antgo.framework.torch2paddle.mapper.utils',
                'antgo.framework.torch2paddle.tools',
                'antgo.framework.torch2paddle.vision',
                ],
      ext_modules=ext_modules(),
      entry_points={'console_scripts': ['antgo=antgo.main:main'], },
      install_requires=read_requirements('requirements.txt'), 
      long_description=readme(),
      include_package_data=True,
      zip_safe=False,)
