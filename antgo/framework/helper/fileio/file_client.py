from curses.ascii import isdigit
from decimal import localcontext
import inspect
import os
import os.path as osp
import re
import tempfile
from threading import local
import warnings
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from aligo import Aligo
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple, Union
from antgo.framework.helper.utils.path import is_filepath
from antgo.ant import environment
import shutil
import time


class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends.

    All backends need to implement two apis: ``get()`` and ``get_text()``.
    ``get()`` reads the file as a byte stream and ``get_text()`` reads the file
    as texts.
    """

    # a flag to indicate whether the backend can create a symlink for a file
    _allow_symlink = False

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def allow_symlink(self):
        return self._allow_symlink

    @abstractmethod
    def download(self, remote_path, local_path=None):
        pass

    @abstractmethod
    def upload(self, remote_path, local_path, is_exist=False):
        pass

    @abstractmethod
    def ls(self, remote_folder):
        pass

    @abstractmethod
    def mkdir(self, remote_path, p=False):
        pass


class HardDiskBackend(BaseStorageBackend):
    """Raw hard disks storage backend."""

    _allow_symlink = True

    def get(self, filepath: Union[str, Path]) -> bytes:
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Expected bytes object.
        """
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def get_text(self,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8') -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.
        """
        with open(filepath, 'r', encoding=encoding) as f:
            value_buf = f.read()
        return value_buf

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        """Write data to a given ``filepath`` with 'wb' mode.

        Note:
            ``put`` will create a directory if the directory of ``filepath``
            does not exist.

        Args:
            obj (bytes): Data to be written.
            filepath (str or Path): Path to write data.
        """
        if not os.path.exists(osp.dirname(filepath)):
            os.makedirs(osp.dirname(filepath))
        with open(filepath, 'wb') as f:
            f.write(obj)

    def put_text(self,
                 obj: str,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8') -> None:
        """Write data to a given ``filepath`` with 'w' mode.

        Note:
            ``put_text`` will create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.
        """
        if not os.path.exists(osp.dirname(filepath)):
            os.makedirs(osp.dirname(filepath))        
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(obj)

    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove a file.

        Args:
            filepath (str or Path): Path to be removed.
        """
        os.remove(filepath)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.

        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.
        """
        return osp.exists(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
            ``False`` otherwise.
        """
        return osp.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
            otherwise.
        """
        return osp.isfile(filepath)

    def join_path(self, filepath: Union[str, Path],
                  *filepaths: Union[str, Path]) -> str:
        """Concatenate all file paths.

        Join one or more filepath components intelligently. The return value
        is the concatenation of filepath and any members of *filepaths.

        Args:
            filepath (str or Path): Path to be concatenated.

        Returns:
            str: The result of concatenation.
        """
        return osp.join(filepath, *filepaths)

    def list_dir_or_file(self,
                         dir_path: Union[str, Path],
                         list_dir: bool = True,
                         list_file: bool = True,
                         suffix: Optional[Union[str, Tuple[str]]] = None,
                         recursive: bool = False) -> Iterator[str]:
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
                    rel_path = osp.relpath(entry.path, root)
                    if (suffix is None
                            or rel_path.endswith(suffix)) and list_file:
                        yield rel_path
                elif osp.isdir(entry.path):
                    if list_dir:
                        rel_dir = osp.relpath(entry.path, root)
                        yield rel_dir
                    if recursive:
                        yield from _list_dir_or_file(entry.path, list_dir,
                                                     list_file, suffix,
                                                     recursive)

        return _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                                 recursive)

    def ls(self, remote_folder):
        return os.listdir(remote_folder)

    def mkdir(self, remote_path, p=False):
        if os.path.exists(remote_path):
            return False

        os.makedirs(remote_path)
        return True

    def download(self, remote_path, local_path):
        # donwload from remote_path to local_path
        if HardDiskBackend._allow_symlink:
            os.system(f'ln -s {remote_path} {local_path}')
        else:
            shutil.copy(remote_path, local_path)

        return True

    def upload(self, remote_path, local_path, is_exist=False):
        # upload from local_path to remote_path
        if is_exist:
            # 如果存在，则取消copy
            if os.path.exists(remote_path):
                return

        shutil.copy(local_path, remote_path)
        return True


class HDFSBackend(BaseStorageBackend):
    _allow_symlink = False

    def get(self, filepath) -> bytes:
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath str: Path to read data.

        Returns:
            bytes: Expected bytes object.
        """
        if not os.path.exists('./temp'):
            os.makedirs('./temp')

        environment.hdfs_client.get(filepath, './temp')
        filename = filepath.split('/')[-1]

        with open(os.path.join(f'./temp/{filename}'), 'rb') as f:
            value_buf = f.read()
        return value_buf

    def get_text(self,
                 filepath: str,
                 encoding: str = 'utf-8') -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath str: Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.
        """
        if not os.path.exists('./temp'):
            os.makedirs('./temp')

        environment.hdfs_client.get(filepath, './temp')
        filename = filepath.split('/')[-1]

        with open(os.path.join(f'./temp/{filename}'), 'r', encoding=encoding) as f:
            value_buf = f.read()
        return value_buf

    def put(self, obj: bytes, filepath: str) -> None:
        """Write data to a given ``filepath`` with 'wb' mode.

        Note:
            ``put`` will create a directory if the directory of ``filepath``
            does not exist.

        Args:
            obj (bytes): Data to be written.
            filepath (str or Path): Path to write data.
        """
        if not os.path.exists('./temp'):
            os.makedirs('./temp')
        filename = filepath.split('/')[-1]

        with open(os.path.join('./temp', filename), 'wb') as f:
            f.write(obj)

        p = filepath.find(filename)
        parent_path = filepath[:p]
        if parent_path != '':
            environment.hdfs_client.mkdir(parent_path, True)
        environment.hdfs_client.put(filepath, os.path.join('./temp', filename))

    def put_text(self,
                 obj: str,
                 filepath: str,
                 encoding: str = 'utf-8') -> None:
        """Write data to a given ``filepath`` with 'w' mode.

        Note:
            ``put_text`` will create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.
        """
        if not os.path.exists('./temp'):
            os.makedirs('./temp')
        filename = filepath.split('/')[-1]

        with open(os.path.join('./temp', filename), 'w', encoding=encoding) as f:
            f.write(obj)

        p = filepath.find(filename)
        parent_path = filepath[:p]
        if parent_path != '':
            environment.hdfs_client.mkdir(parent_path, True)
        environment.hdfs_client.put(filepath, os.path.join('./temp', filename))

    def remove(self, filepath: str) -> None:
        """Remove a file.

        Args:
            filepath (str or Path): Path to be removed.
        """
        environment.hdfs_client.rm(filepath)

    def exists(self, filepath: str) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.

        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.
        """
        return environment.hdfs_client.exists(filepath)

    def isdir(self, filepath: str) -> bool:
        """
        仅依靠路径名称进行判断，存在不准确
        """
        if filepath.endswith('/'):
            return True
        
        if '.' in filepath.split('/')[-1]:
            return False
        else:
            return True

    def isfile(self, filepath: str) -> bool:
        """
        仅依靠路径名称进行判断，存在不准确
        """
        return not self.isdir(filepath)

    def join_path(self, filepath: str, *filepaths: str) -> str:
        """Concatenate all file paths.

        Join one or more filepath components intelligently. The return value
        is the concatenation of filepath and any members of *filepaths.

        Args:
            filepath (str or Path): Path to be concatenated.

        Returns:
            str: The result of concatenation.
        """
        return osp.join(filepath, *filepaths)

    def list_dir_or_file(self,
                         dir_path: Union[str, Path],
                         list_dir: bool = True,
                         list_file: bool = True,
                         suffix: Optional[Union[str, Tuple[str]]] = None,
                         recursive: bool = False) -> Iterator[str]:
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
        raise NotImplementedError

    def ls(self, folder):
        return environment.hdfs_client.ls(folder)

    def mkdir(self, remote_path, p=False):
        environment.hdfs_client.mkdir(remote_path, p)
        return True

    def download(self, remote_path, local_path):
        # donwload from remote_path to local_path
        status = environment.hdfs_client.get(remote_path, local_path)
        return status

    def upload(self, remote_path, local_path, is_exist=False):
        # upload from local_path to remote_path
        status = environment.hdfs_client.put(remote_path, local_path, is_exist)
        return status


class AliBackend(BaseStorageBackend):
    def __init__(self) -> None:
        super().__init__()
        local_config_file = 'aligo.json'
        if os.path.exists(local_config_file):
            if not os.path.exists(Path.home().joinpath('.aligo')):
                os.makedirs(Path.home().joinpath('.aligo'))
                
            shutil.copy(local_config_file, Path.home().joinpath('.aligo'))

        self.ali = Aligo()  # 第一次使用，会弹出二维码，供扫描登录
        if not os.path.exists(local_config_file):
            shutil.copy(os.path.join(Path.home().joinpath('.aligo'), 'aligo.json'),'./')
        self.prefix = 'ali://'
        
    def get(self, filepath) -> bytes:
        if not os.path.exists('./temp'):
            os.makedirs('./temp')

        self.download(filepath, './temp')
        filename = filepath.split('/')[-1]

        with open(os.path.join(f'./temp/{filename}'), 'rb') as f:
            value_buf = f.read()
        return value_buf        
    
    def get_text(self,
                 filepath: str,
                 encoding: str = 'utf-8') -> str:
        if not os.path.exists('./temp'):
            os.makedirs('./temp')

        self.download(filepath, './temp')
        filename = filepath.split('/')[-1]

        with open(os.path.join(f'./temp/{filename}'), 'r', encoding=encoding) as f:
            value_buf = f.read()
        return value_buf        

    def put(self, obj: bytes, filepath: str) -> None:
        if not os.path.exists('./temp'):
            os.makedirs('./temp')
        filename = filepath.split('/')[-1]

        with open(os.path.join('./temp', filename), 'wb') as f:
            f.write(obj)

        p = filepath.find(filename)
        parent_path = filepath[:p]
        self.upload(parent_path, os.path.join('./temp', filename))

    def put_text(self,
                 obj: str,
                 filepath: str,
                 encoding: str = 'utf-8') -> None:
        if not os.path.exists('./temp'):
            os.makedirs('./temp')
        filename = filepath.split('/')[-1]

        with open(os.path.join('./temp', filename), 'w', encoding=encoding) as f:
            f.write(obj)

        p = filepath.find(filename)
        parent_path = filepath[:p]
        self.upload(parent_path, os.path.join('./temp', filename))

    def remove(self, filepath: str) -> None:
        pass

    def exists(self, filepath: str) -> bool:
        filepath = filepath.replace(self.prefix, '')
        file = self.ali.get_file_by_path(filepath)
        if file is None:
            file = self.ali.get_folder_by_path(filepath)
            if file is None:
                return False

        return True

    def isdir(self, filepath: str) -> bool:
        file = self.ali.get_folder_by_path(filepath)
        if file is None:
            return False

        return True

    def isfile(self, filepath: str) -> bool:
        file = self.ali.get_file_by_path(filepath)
        if file is None:
            return False

        return True

    def join_path(self, filepath: str, *filepaths: str) -> str:
        """Concatenate all file paths.

        Join one or more filepath components intelligently. The return value
        is the concatenation of filepath and any members of *filepaths.

        Args:
            filepath (str or Path): Path to be concatenated.

        Returns:
            str: The result of concatenation.
        """
        return osp.join(filepath, *filepaths)

    def ls(self, remote_folder):
        remote_folder = remote_folder.replace(self.prefix, '')
        file_handler = self.ali.get_folder_by_path(remote_folder)
        if file_handler is None:
            return []

        ll = self.ali.get_file_list(file_handler.file_id)
        ll = [f'{self.prefix}{os.path.join(remote_folder, l.name)}' for l in ll]
        return ll

    def mkdir(self, remote_path, p=False):
        # remote prefix
        remote_path = remote_path.replace(self.prefix, '')
        remote_path = os.path.normpath(remote_path)
        # 迭代创建目录
        levels = remote_path.split('/')[1:]
        level_num = len(levels)
        find_file = None
        find_i = 0
        for i in range(level_num,0,-1):
            check_path = '/'+'/'.join(levels[:i])            
            find_file = self.ali.get_folder_by_path(check_path)
            if find_file:
                break
            find_i = i

        if find_i == 0:
            # 已经存在，不进行重新创建
            return find_file.file_id

        sub_folder = '/'.join(levels[find_i-1:])
        ss = self.ali.create_folder(sub_folder,find_file.file_id)
        return ss.file_id

    def download(self, remote_path, local_path):
        # user = ali.get_user
        remote_path = remote_path.replace(self.prefix, '')
        download_as_multi_files = False
        if remote_path.endswith('*'):
            download_as_multi_files = True
            file = self.ali.get_folder_by_path(os.path.dirname(remote_path))
        else:
            file = self.ali.get_file_by_path(remote_path)
            if file is None:
                file = self.ali.get_folder_by_path(remote_path)

        if file is None:
            return False

        if file.type == 'file':
            self.ali.download_file(file=file, local_folder=local_path)
        else:
            if not download_as_multi_files:
                self.ali.download_folder(file.file_id, local_folder=local_path)
            else:
                ll = self.ali.get_file_list(file.file_id)
                prefix = remote_path.split('/')[-1][:-1]
                filter_ll = []
                if prefix == '':
                    filter_ll = ll
                else:
                    for l in ll:
                        if l.name.startswith(prefix):
                            filter_ll.append(l)
                self.ali.download_files(filter_ll, local_folder=local_path)

        return True

    def upload(self, remote_path, local_path, is_exist=False):
        # 检查远程目录是否存在
        file_id = self.mkdir(remote_path, True)
        if file_id is None:
            # error
            return False

        if os.path.isdir(local_path):
            # 目录
            self.ali.upload_folder(local_path, file_id)
        else:
            # 文件
            self.ali.upload_file(local_path, file_id)

        return True



class FileClient:
    """A general file client to access files in different backends.

    The client loads a file or text in a specified backend from its path
    and returns it as a binary or text file. There are two ways to choose a
    backend, the name of backend and the prefix of path. Although both of them
    can be used to choose a storage backend, ``backend`` has a higher priority
    that is if they are all set, the storage backend will be chosen by the
    backend argument. If they are all `None`, the disk backend will be chosen.
    Note that It can also register other backend accessor with a given name,
    prefixes, and backend class. In addition, We use the singleton pattern to
    avoid repeated object creation. If the arguments are the same, the same
    object will be returned.

    Args:
        backend (str, optional): The storage backend type. Options are "disk",
            "ceph", "memcached", "lmdb", "http" and "petrel". Default: None.
        prefix (str, optional): The prefix of the registered storage backend.
            Options are "s3", "http", "https". Default: None.

    Examples:
        >>> # only set backend
        >>> file_client = FileClient(backend='petrel')
        >>> # only set prefix
        >>> file_client = FileClient(prefix='s3')
        >>> # set both backend and prefix but use backend to choose client
        >>> file_client = FileClient(backend='petrel', prefix='s3')
        >>> # if the arguments are the same, the same object is returned
        >>> file_client1 = FileClient(backend='petrel')
        >>> file_client1 is file_client
        True

    Attributes:
        client (:obj:`BaseStorageBackend`): The backend object.
    """

    _backends = {
        'disk': HardDiskBackend,
        'hdfs': HDFSBackend,
        'ali': AliBackend
    }
    # This collection is used to record the overridden backends, and when a
    # backend appears in the collection, the singleton pattern is disabled for
    # that backend, because if the singleton pattern is used, then the object
    # returned will be the backend before overwriting
    _overridden_backends = set()
    _overridden_prefixes = set()

    _instances = {}

    def __new__(cls, backend=None, prefix=None, **kwargs):
        if backend is None and prefix is None:
            backend = 'disk'
        
        # concatenate the arguments to a unique key for determining whether
        # objects with the same arguments were created
        arg_key = f'{backend}:{prefix}'
        for key, value in kwargs.items():
            arg_key += f':{key}:{value}'

        # if a backend was overridden, it will create a new object
        if (arg_key in cls._instances
                and backend not in cls._overridden_backends
                and prefix not in cls._overridden_prefixes):
            _instance = cls._instances[arg_key]
        else:
            # create a new object and put it to _instance
            _instance = super().__new__(cls)
            _instance.client = cls._backends[backend](**kwargs)
            cls._instances[arg_key] = _instance

        return _instance

    @property
    def name(self):
        return self.client.name

    @property
    def allow_symlink(self):
        return self.client.allow_symlink

    @classmethod
    def infer_client(cls,
                     file_client_args: Optional[dict] = None,
                     uri: Optional[str] = None) -> 'FileClient':
        """Infer a suitable file client based on the URI and arguments.

        Args:
            file_client_args (dict, optional): Arguments to instantiate a
                FileClient. Default: None.
            uri (str | Path, optional): Uri to be parsed that contains the file
                prefix. Default: None.

        Returns:
            FileClient: Instantiated FileClient object.
        """
        assert file_client_args is not None or uri is not None
        if file_client_args is None:
            if uri.startswith('hdfs'):
                file_client_args = {'backend': 'hdfs'}
            elif uri.startswith('ali'):
                file_client_args = {'backend': 'ali'}
            elif uri.startswith('/') or uri.startswith('./'):
                file_client_args = {'backend': 'disk'}
        if file_client_args is None:
            return None

        return cls(**file_client_args)

    @classmethod
    def _register_backend(cls, name, backend, force=False, prefixes=None):
        if not isinstance(name, str):
            raise TypeError('the backend name should be a string, '
                            f'but got {type(name)}')
        if not inspect.isclass(backend):
            raise TypeError(
                f'backend should be a class but got {type(backend)}')
        if not issubclass(backend, BaseStorageBackend):
            raise TypeError(
                f'backend {backend} is not a subclass of BaseStorageBackend')
        if not force and name in cls._backends:
            raise KeyError(
                f'{name} is already registered as a storage backend, '
                'add "force=True" if you want to override it')

        if name in cls._backends and force:
            cls._overridden_backends.add(name)
        cls._backends[name] = backend

        if prefixes is not None:
            if isinstance(prefixes, str):
                prefixes = [prefixes]
            else:
                assert isinstance(prefixes, (list, tuple))
            for prefix in prefixes:
                if prefix not in cls._prefix_to_backends:
                    cls._prefix_to_backends[prefix] = backend
                elif (prefix in cls._prefix_to_backends) and force:
                    cls._overridden_prefixes.add(prefix)
                    cls._prefix_to_backends[prefix] = backend
                else:
                    raise KeyError(
                        f'{prefix} is already registered as a storage backend,'
                        ' add "force=True" if you want to override it')

    @classmethod
    def register_backend(cls, name, backend=None, force=False, prefixes=None):
        """Register a backend to FileClient.

        This method can be used as a normal class method or a decorator.

        .. code-block:: python

            class NewBackend(BaseStorageBackend):

                def get(self, filepath):
                    return filepath

                def get_text(self, filepath):
                    return filepath

            FileClient.register_backend('new', NewBackend)

        or

        .. code-block:: python

            @FileClient.register_backend('new')
            class NewBackend(BaseStorageBackend):

                def get(self, filepath):
                    return filepath

                def get_text(self, filepath):
                    return filepath

        Args:
            name (str): The name of the registered backend.
            backend (class, optional): The backend class to be registered,
                which must be a subclass of :class:`BaseStorageBackend`.
                When this method is used as a decorator, backend is None.
                Defaults to None.
            force (bool, optional): Whether to override the backend if the name
                has already been registered. Defaults to False.
            prefixes (str or list[str] or tuple[str], optional): The prefixes
                of the registered storage backend. Default: None.
                `New in version 1.3.15.`
        """
        if backend is not None:
            cls._register_backend(
                name, backend, force=force, prefixes=prefixes)
            return

        def _register(backend_cls):
            cls._register_backend(
                name, backend_cls, force=force, prefixes=prefixes)
            return backend_cls

        return _register

    def get(self, filepath: Union[str, Path]) -> Union[bytes, memoryview]:
        """Read data from a given ``filepath`` with 'rb' mode.

        Note:
            There are two types of return values for ``get``, one is ``bytes``
            and the other is ``memoryview``. The advantage of using memoryview
            is that you can avoid copying, and if you want to convert it to
            ``bytes``, you can use ``.tobytes()``.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes | memoryview: Expected bytes object or a memory view of the
            bytes object.
        """
        return self.client.get(filepath)

    def get_text(self, filepath: Union[str, Path], encoding='utf-8') -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Default: 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.
        """
        return self.client.get_text(filepath, encoding)

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        """Write data to a given ``filepath`` with 'wb' mode.

        Note:
            ``put`` should create a directory if the directory of ``filepath``
            does not exist.

        Args:
            obj (bytes): Data to be written.
            filepath (str or Path): Path to write data.
        """
        self.client.put(obj, filepath)

    def put_text(self, obj: str, filepath: Union[str, Path]) -> None:
        """Write data to a given ``filepath`` with 'w' mode.

        Note:
            ``put_text`` should create a directory if the directory of
            ``filepath`` does not exist.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str, optional): The encoding format used to open the
                `filepath`. Default: 'utf-8'.
        """
        self.client.put_text(obj, filepath)

    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove a file.

        Args:
            filepath (str, Path): Path to be removed.
        """
        self.client.remove(filepath)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.

        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.
        """
        return self.client.exists(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
            ``False`` otherwise.
        """
        return self.client.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
            otherwise.
        """
        return self.client.isfile(filepath)

    def join_path(self, filepath: Union[str, Path],
                  *filepaths: Union[str, Path]) -> str:
        """Concatenate all file paths.

        Join one or more filepath components intelligently. The return value
        is the concatenation of filepath and any members of *filepaths.

        Args:
            filepath (str or Path): Path to be concatenated.

        Returns:
            str: The result of concatenation.
        """
        return self.client.join_path(filepath, *filepaths)

    @contextmanager
    def get_local_path(self, filepath: Union[str, Path]) -> Iterable[str]:
        """Download data from ``filepath`` and write the data to local path.

        ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Note:
            If the ``filepath`` is a local path, just return itself.

        .. warning::
            ``get_local_path`` is an experimental interface that may change in
            the future.

        Args:
            filepath (str or Path): Path to be read data.

        Examples:
            >>> file_client = FileClient(prefix='s3')
            >>> with file_client.get_local_path('s3://bucket/abc.jpg') as path:
            ...     # do something here

        Yields:
            Iterable[str]: Only yield one path.
        """
        with self.client.get_local_path(str(filepath)) as local_path:
            yield local_path

    def list_dir_or_file(self,
                         dir_path: Union[str, Path],
                         list_dir: bool = True,
                         list_file: bool = True,
                         suffix: Optional[Union[str, Tuple[str]]] = None,
                         recursive: bool = False) -> Iterator[str]:
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
        yield from self.client.list_dir_or_file(dir_path, list_dir, list_file,
                                                suffix, recursive)

    def ls(self, remote_folder):
        return self.client.ls(remote_folder)

    def mkdir(self, remote_path, p=False):
        return self.client.mkdir(remote_path, p)

    def download(self, remote_path, local_path):
        return self.client.download(remote_path, local_path)

    def upload(self, remote_path, local_path, is_exist=False):
        return self.client.upload(remote_path, local_path, is_exist)


def file_client_get(remote_path, local_path):
    status = FileClient.infer_client(uri=remote_path).download(remote_path, local_path)
    return status


def file_client_put(remote_path, local_path, is_exist=False):
    status = FileClient.infer_client(uri=remote_path).upload(remote_path, local_path, is_exist)
    return status


def file_client_ls(remote_folder):
    result = FileClient.infer_client(uri=remote_folder).ls(remote_folder)
    return result


def file_client_mkdir(remote_path, p=False):
    status = FileClient.infer_client(uri=remote_path).mkdir(remote_path, p)
    return status


def file_client_rm(remote_path):
    status = FileClient.infer_client(uri=remote_path).remove(remote_path)
    return status


def file_client_exists(remote_path):
    status = FileClient.infer_client(uri=remote_path).exists(remote_path)
    return status

