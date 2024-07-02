import os
import pathlib
import sys
ANTGO_DEPEND_ROOT = os.environ.get('ANTGO_DEPEND_ROOT', f'{str(pathlib.Path.home())}/.3rd')
if not os.path.exists(ANTGO_DEPEND_ROOT):
    os.makedirs(ANTGO_DEPEND_ROOT)

if not os.path.exists(os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye', 'py')):
    if not os.path.exists(os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye')):
        os.system(f'cd {ANTGO_DEPEND_ROOT} && git clone https://github.com/jianzfb/eagleeye.git')

    if 'darwin' in sys.platform:
        os.system(f'cd {ANTGO_DEPEND_ROOT}/eagleeye && bash osx_build.sh BUILD_PYTHON_MODULE && mv install py')
    else:
        first_comiple = False
        if not os.path.exists(os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye','py')):
            first_comiple = True
        os.system(f'cd {ANTGO_DEPEND_ROOT}/eagleeye && bash linux_build.sh BUILD_PYTHON_MODULE && mv install py')
        if first_comiple:
            # 增加搜索.so路径
            cur_abs_path = os.path.abspath(os.curdir)
            so_path = f"{ANTGO_DEPEND_ROOT}/eagleeye/py/libs/X86-64"
            os.system(f'echo "{so_path}" >> /etc/ld.so.conf')
            so_path = "/usr/lib/x86_64-linux-gnu/"
            os.system(f'echo "{so_path}" >> /etc/ld.so.conf')
            os.system('ldconfig')

if f'{ANTGO_DEPEND_ROOT}/eagleeye/py/libs/X86-64' not in sys.path:
    sys.path.append(f'{ANTGO_DEPEND_ROOT}/eagleeye/py/libs/X86-64')
import eagleeye