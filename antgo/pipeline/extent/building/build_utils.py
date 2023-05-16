"""Building Utils"""

from ..config import config
from ..utils import makedirs
import ast
import os
import threading
import platform
import re
from subprocess import Popen, PIPE
try:
    import Queue
except ImportError:
    import queue as Queue
if not hasattr(Queue.Queue, 'clear'):
    def _queue_clear(self):
        with self.mutex:
            self.queue.clear()
    setattr(Queue.Queue, 'clear', _queue_clear)

from .build_path import get_virtual_dirname, change_exts, add_path
from .build_dependant import get_include_file
from .build_flags import Flags
from .build_hash import path_hash, get_file_hash
from .build_latest_code import file_is_changed, code_need_to_rebuild, save_latest_state

OS_NAME = platform.system()
OS_IS_WINDOWS = OS_NAME == 'Windows'
OS_IS_LINUX = OS_NAME in ['Linux', 'Darwin']
assert OS_IS_WINDOWS or OS_IS_LINUX,\
    Exception('Unsupported Operator System: {}'.format(OS_NAME))

INC_PATHS = ['./']

# Load Config File
ENV_PATH = os.path.join(os.path.dirname(__file__), '../')
if os.path.dirname(config.BUILD_PATH) == '.':
    config.BUILD_PATH = os.path.join(ENV_PATH, config.BUILD_PATH)


def mkdir(dir_name):
    if not os.path.exists(dir_name):
        if config.SHOW_BUILDING_COMMAND:
            print('mkdir -p %s' % dir_name)
        makedirs(dir_name, exist_ok=True)


if OS_IS_LINUX:
    _rmdir_command = 'rm -rf'
elif OS_IS_WINDOWS:
    _rmdir_command = 'rd /s /q'


def rmdir(dir_name):
    # we use shell command to remove the non-empty or empry directory
    if os.path.exists(dir_name):
        command = '%s %s' % (_rmdir_command, dir_name)
        run_command(command)


def run_command(command):
    if config.SHOW_BUILDING_COMMAND:
        print(command)
    return os.system(command)


def run_command_parallel(commands, allow_error=False):
    command_queue = Queue.Queue()
    info_queue = Queue.Queue()
    for c in commands:
        command_queue.put(c)
    max_worker_num = min(config.MAX_BUILDING_WORKER_NUM, len(commands))
    for _ in range(max_worker_num):
        command_queue.put(None)

    def worker(command_queue, info_queue):
        while not command_queue.empty():
            e = command_queue.get()
            if e is None:
                break
            rtn = run_command(e)
            if rtn != 0:
                # Error
                command_queue.clear()
                info_queue.put(Exception('Error, terminated :-('))
    workers = [threading.Thread(target=worker, args=(command_queue, info_queue))
               for _ in range(max_worker_num)]
    for w in workers:
        w.daemon = True
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    while not info_queue.empty():
        info = info_queue.get()
        if isinstance(info, Exception) and not allow_error:
            raise RuntimeError(info)


def command_exists(command):
    try:
        Popen([command], stdout=PIPE, stderr=PIPE, stdin=PIPE)
    except Exception:
        return False
    return True


class build_context:
    def __enter__(self):
        pass

    def __exit__(self, *dummy):
        save_latest_state()
