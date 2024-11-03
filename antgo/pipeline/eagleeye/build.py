import os
import pathlib
import sys

ANTGO_DEPEND_ROOT = os.environ.get('ANTGO_DEPEND_ROOT', f'{str(pathlib.Path.home())}/.3rd')

def build_eagleeye_env():
    if os.path.exists(os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye','py')):
        if f'{ANTGO_DEPEND_ROOT}/eagleeye/py/libs/x86-64' not in sys.path:
            sys.path.append(f'{ANTGO_DEPEND_ROOT}/eagleeye/py/libs/x86-64')
        return True

    if not os.path.exists(os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye')):
        os.system(f'cd {ANTGO_DEPEND_ROOT} && git clone https://github.com/jianzfb/eagleeye.git')

    if 'darwin' in sys.platform:
        os.system(f'cd {ANTGO_DEPEND_ROOT}/eagleeye && bash osx_build.sh BUILD_PYTHON_MODULE')
    else:
        first_comiple = False
        if not os.path.exists(os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye','py')):
            first_comiple = True
        os.system(f'cd {ANTGO_DEPEND_ROOT}/eagleeye && bash linux_x86_64_build.sh BUILD_PYTHON_MODULE')
        if first_comiple:
            # 增加搜索.so路径
            cur_abs_path = os.path.abspath(os.curdir)
            so_path = f"{ANTGO_DEPEND_ROOT}/eagleeye/py/libs/x86-64"
            os.system(f'echo "{so_path}" >> /etc/ld.so.conf')
            so_path = "/usr/lib/x86_64-linux-gnu/"
            os.system(f'echo "{so_path}" >> /etc/ld.so.conf')
            os.system('ldconfig')

    if f'{ANTGO_DEPEND_ROOT}/eagleeye/py/libs/x86-64' not in sys.path:
        sys.path.append(f'{ANTGO_DEPEND_ROOT}/eagleeye/py/libs/x86-64')

    return True