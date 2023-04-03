# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from antgo.dataflow.dataset.clsdataset import *
import os


def list_dir_or_file(dir_path,
                        list_dir = True,
                        list_file = True,
                        suffix = None,
                        recursive = False):
    """Scan a directory to find the interested directories or files in
    arbitrary order.

    Note:
        :meth:`list_dir_or_file` returns the path relative to ``dir_path``.

    Args:
        dir_path (str | Path): Path of the directory.
        list_dir (bool): List the directories. Default: True.
        list_file (bool): List the path of files. Default: True.
        suffix (str or tuple[str], optional):  File suffix
            that we are interested in. Default: None.
        recursive (bool): If set to True, recursively scan the
            directory. Default: False.

    Yields:
        Iterable[str]: A relative path to ``dir_path``.
    """
    if list_dir and suffix is not None:
        raise TypeError('`suffix` should be None when `list_dir` is True')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('`suffix` must be a string or tuple of strings')

    root = dir_path

    def _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                            recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = os.path.relpath(entry.path, root)
                if (suffix is None
                        or rel_path.endswith(suffix)) and list_file:
                    yield rel_path
            elif os.path.isdir(entry.path):
                if list_dir:
                    rel_dir = os.path.relpath(entry.path, root)
                    yield rel_dir
                if recursive:
                    yield from _list_dir_or_file(entry.path, list_dir,
                                                    list_file, suffix,
                                                    recursive)

    return _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                                recursive)


def find_folders(root: str):
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        Tuple[List[str], Dict[str, int]]:

        - folders: The name of sub folders under the root.
        - folder_to_idx: The map from folder name to class idx.
    """
    folders = list(
        list_dir_or_file(
            root,
            list_dir=True,
            list_file=False,
            recursive=False,
        ))
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folders, folder_to_idx


def get_samples(root: str, folder_to_idx: Dict[str, int],
                is_valid_file: Callable):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        is_valid_file (Callable): A function that takes path of a file
            and check if the file is a valid sample file.

    Returns:
        Tuple[list, set]:

        - samples: a list of tuple where each element is (image, class_idx)
        - empty_folders: The folders don't have any valid files.
    """
    samples = []
    available_classes = set()

    for folder_name in sorted(list(folder_to_idx.keys())):
        _dir = os.path.join(root, folder_name)
        files = list(
            list_dir_or_file(
                _dir,
                list_dir=False,
                list_file=True,
                recursive=True,
            ))
        for file in sorted(list(files)):
            if is_valid_file(file):
                path = os.path.join(folder_name, file)
                item = (path, folder_to_idx[folder_name])
                samples.append(item)
                available_classes.add(folder_name)

    empty_folders = set(folder_to_idx.keys()) - available_classes

    return samples, empty_folders


class CustomClsDataset(ClsDataset):
    """Custom dataset for classification.

    The dataset supports two kinds of annotation format.

    1. An annotation file is provided, and each line indicates a sample:

       The sample files: ::

           data_prefix/
           ├── folder_1
           │   ├── xxx.png
           │   ├── xxy.png
           │   └── ...
           └── folder_2
               ├── 123.png
               ├── nsdf3.png
               └── ...

       The annotation file (the first column is the image path and the second
       column is the index of category): ::

            folder_1/xxx.png 0
            folder_1/xxy.png 1
            folder_2/123.png 5
            folder_2/nsdf3.png 3
            ...

       Please specify the name of categories by the argument ``classes``.

    2. The samples are arranged in the specific way: ::

           data_prefix/
           ├── class_x
           │   ├── xxx.png
           │   ├── xxy.png
           │   └── ...
           │       └── xxz.png
           └── class_y
               ├── 123.png
               ├── nsdf3.png
               ├── ...
               └── asd932_.png

    If the ``ann_file`` is specified, the dataset will be generated by the
    first way, otherwise, try the second way.

    Args:
        data_prefix (str): The path of data directory.
        pipeline (Sequence[dict]): A list of dict, where each element
            represents a operation defined in :mod:`mmcls.datasets.pipelines`.
            Defaults to an empty tuple.
        classes (str | Sequence[str], optional): Specify names of classes.

            - If is string, it should be a file path, and the every line of
              the file is a name of a class.
            - If is a sequence of string, every item is a name of class.
            - If is None, use ``cls.CLASSES`` or the names of sub folders
              (If use the second way to arrange samples).

            Defaults to None.
        ann_file (str, optional): The annotation file. If is string, read
            samples paths from the ann_file. If is None, find samples in
            ``data_prefix``. Defaults to None.
        extensions (Sequence[str]): A sequence of allowed extensions. Defaults
            to ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif').
        test_mode (bool): In train mode or test mode. It's only a mark and
            won't be used in this class. Defaults to False.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            If None, automatically inference from the specified path.
            Defaults to None.
    """

    def __init__(self,
                #  data_prefix: str,
                #  pipeline: Sequence = (),
                #  classes: Union[str, Sequence[str], None] = None,
                #  ann_file: Optional[str] = None,
                #  extensions: Sequence[str] = ('.jpg', '.jpeg', '.png', '.ppm',
                #                               '.bmp', '.pgm', '.tif'),
                #  test_mode: bool = False,
                 
                 train_or_test='train',
                 dir=None,
                 params={
                     'data_prefix': '', 
                     'ann_file': None, 
                     'classes': None, 
                     'extensions': ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')}
                 ):
        self.extensions = tuple(set([i.lower() for i in params['extensions']]))
        super().__init__(train_or_test,dir,params)

    def _find_samples(self):
        """find samples from ``data_prefix``."""
        classes, folder_to_idx = find_folders(self.data_prefix)
        samples, empty_classes = get_samples(
            self.data_prefix,
            folder_to_idx,
            is_valid_file=self.is_valid_file
        )

        if len(samples) == 0:
            raise RuntimeError(
                f'Found 0 files in subfolders of: {self.data_prefix}. '
                f'Supported extensions are: {",".join(self.extensions)}')

        if self.CLASSES is not None:
            assert len(self.CLASSES) == len(classes), \
                f"The number of subfolders ({len(classes)}) doesn't match " \
                f'the number of specified classes ({len(self.CLASSES)}). ' \
                'Please check the data folder.'
        else:
            self.CLASSES = classes

        if empty_classes:
            warnings.warn(
                'Found no valid file in the folder '
                f'{", ".join(empty_classes)}. '
                f"Supported extensions are: {', '.join(self.extensions)}",
                UserWarning)

        self.folder_to_idx = folder_to_idx

        return samples

    def load_annotations(self):
        """Load image paths and gt_labels."""
        if self.ann_file is None:
            samples = self._find_samples()
        elif isinstance(self.ann_file, str):
            lines = list_from_file(self.ann_file)
            samples = [x.strip().rsplit(' ', 1) for x in lines]
        else:
            raise TypeError('ann_file must be a str or None')

        data_infos = []
        for filename, gt_label in samples:
            info = {}
            info['image_meta'] = {'filename': filename, 'img_prefix': self.data_prefix}
            info['image_file'] = os.path.join(self.data_prefix,filename)
            if not os.path.exists(info['image_file']):
                info['image_file'] = os.path.join(self.data_prefix, self.train_or_test, filename)            
            
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            info['category_id'] = np.array(gt_label, dtype=np.int64)
            info['id'] = len(data_infos)
            data_infos.append(info)
        return data_infos

    def is_valid_file(self, filename: str) -> bool:
        """Check if a file is a valid sample."""
        return filename.lower().endswith(self.extensions)
