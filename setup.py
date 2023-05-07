from setuptools import setup
from setuptools import dist
from setuptools import find_packages
from distutils.extension import Extension
from antgo import __version__
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
      version=str(__version__),
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
      packages=find_packages(),
      ext_modules=ext_modules(),
      entry_points={'console_scripts': ['antgo=antgo.main:main'], },
      # install_requires=read_requirements('requirements.txt'),
      long_description=readme(),
      include_package_data=True,
      zip_safe=False,)
