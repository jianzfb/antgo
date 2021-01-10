======================
antgo
======================

.. image:: https://raw.githubusercontent.com/jianzfb/antgo/master/antgo/resource/static/card.png
    :alt: Antgo

Target
----------------------
antgo is a machine learning experiment manage platform, which has been integrated deeply with MLTalker.
antgo provides some easy cli commands to help ML researchers to manage, analyze, and challenge all kinds
of ML tasks.

Based on amounts of statistical evaluation methods, antgo could give a fruitful evaluation report, which
help researchers analyze and trade-off their model.

Antgo tutorial is at `MLTalker Blog <http://www.mltalker.com/blog/>`__.

Installation
----------------------
install antgo::

    pip install antgo


install antgo from source::

    1. git clone https://github.com/jianzfb/antgo.git
    2. cd antgo
    3. pip install -r requirements.txt
    4. python setup.py build_ext install

Register
-----------------------
Register in `MLTalker <http://www.mltalker.com/>`__.

.. image:: https://raw.githubusercontent.com/jianzfb/antgo/master/antgo/resource/static/register.png
    :alt: Antgo and MLTalker

All user experiment records would be managed by MLTalker in user's personal page.

Quick Example
-----------------------
1.step Build and Apply Task::

.. image:: https://raw.githubusercontent.com/jianzfb/antgo/master/antgo/resource/static/apply.png
    :alt: Build Task in MLTalker


2.step Create Your Project::

.. image:: https://raw.githubusercontent.com/jianzfb/antgo/master/antgo/resource/static/start-experiment.png
    :alt: Get Task ApiToken in MLTalker


antgo startproject --name=MNIST --author=xxx --token=Task API-TOKEN

after, you will get in current folder

.. image:: https://raw.githubusercontent.com/jianzfb/antgo/master/antgo/resource/static/filetree.png
    :alt: file tree


3.step finish train callback function and infer callback function::

...

you can go `MLTalker Blog <http://www.mltalker.com/blog/>`__, to see how to use antgo.


4.step Run Train Task ::

antgo train exp


5.step Run Challenge Task::

antgo challenge exp