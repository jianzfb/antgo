import os
import pathlib
ANTGO_DEPEND_ROOT = os.environ.get('ANTGO_DEPEND_ROOT', f'{str(pathlib.Path.home())}/.3rd')
if not os.path.exists(ANTGO_DEPEND_ROOT):
    os.makedirs(ANTGO_DEPEND_ROOT)

def install_opencv():
    install_path = os.path.join(ANTGO_DEPEND_ROOT, 'opencv-install')
    if not os.path.exists(install_path):
        if not os.path.exists(os.path.join(ANTGO_DEPEND_ROOT, 'opencv')):
            # 下载源码
            os.system(f'cd {ANTGO_DEPEND_ROOT} && git clone https://github.com/opencv/opencv.git -b 3.4')
            os.system(f'cd {ANTGO_DEPEND_ROOT} && git clone https://github.com/opencv/opencv_contrib.git -b 3.4')

        # 编译
        print('compile opencv')
        os.system(f'cd {ANTGO_DEPEND_ROOT} ; cd opencv ; mkdir build ; cd build ; cmake -DOPENCV_EXTRA_MODULES_PATH={ANTGO_DEPEND_ROOT}/opencv_contrib/modules -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX={install_path} -D BUILD_DOCS=OFF -D BUILD_EXAMPLES=OFF -D BUILD_opencv_apps=OFF -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=OFF -D BUILD_PERF_TESTS=OFF  -D BUILD_JAVA=OFF -D BUILD_opencv_java=OFF -D BUILD_TESTS=OFF -D WITH_FFMPEG=OFF .. ; make -j4 ; make install')
        os.system(f'cd {ANTGO_DEPEND_ROOT} ; cd opencv ; rm -rf build')

        # 添加so的搜索路径 (for linux)
        so_abs_path = os.path.join(install_path, 'lib')
        os.system(f'echo "{so_abs_path}" >> /etc/ld.so.conf && ldconfig')

