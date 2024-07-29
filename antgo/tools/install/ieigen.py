import os
import pathlib
ANTGO_DEPEND_ROOT = os.environ.get('ANTGO_DEPEND_ROOT', f'{str(pathlib.Path.home())}/.3rd')
if not os.path.exists(ANTGO_DEPEND_ROOT):
    os.makedirs(ANTGO_DEPEND_ROOT)

def install_eigen():
    install_path = os.path.join(ANTGO_DEPEND_ROOT, 'eigen')
    if not os.path.exists(install_path):
        # 下载源码
        os.system(f'cd {ANTGO_DEPEND_ROOT} && git clone https://gitlab.com/libeigen/eigen.git -b 3.3')
