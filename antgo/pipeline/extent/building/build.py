"""Building Implementation"""
import os
import sys
import multiprocessing
try:
    from .build_utils import *
except Exception:
    from build_utils import *


ANTGO_DEPEND_ROOT = os.environ.get('ANTGO_DEPEND_ROOT', '/workspace/.3rd')
NUM_CPU_CORE = multiprocessing.cpu_count()


def source_to_o(build_path, src_obj, compiler, cflags):
    mkdir(build_path)
    existed_dirs = set()
    updated = False
    commands = []
    for src, obj in src_obj:
        dir_name, obj_name = os.path.split(obj)
        build_dir_name = os.path.join(build_path, dir_name)
        build_name = os.path.join(build_path, dir_name, obj_name)
        if not code_need_to_rebuild(src) and os.path.exists(build_name):
            continue
        updated = True
        if build_dir_name not in existed_dirs:
            mkdir(build_dir_name)
            existed_dirs.add(build_dir_name)
        if OS_IS_WINDOWS and not command_exists(compiler):
            inc_flags = Flags()
            for path in INC_PATHS:
                p = os.path.join(ENV_PATH, path)
                inc_flags.add_string('-I{}'.format(p))
            cflags_sp = str(cflags).split()
            def_flags = ' '.join(
                [s for s in cflags_sp if len(s) > 2 and s[:2] == '-D'])
            command = 'cl /EHsc /O2 %s %s -c %s -Fo%s' % (
                def_flags, inc_flags, src, build_name)
        else:
            command = '%s %s %s -c -o %s' % (compiler, src, cflags, build_name)
        commands.append(command)
    run_command_parallel(commands)
    return updated


def o_to_so(target_name, objs, linker, ldflags):
    if OS_IS_WINDOWS and not command_exists(linker):
        command = 'link -DLL %s -out:%s' % (' '.join(objs), target_name)
    else:
        command = '%s %s %s -o %s' % (linker,
                                      ' '.join(objs), ldflags, target_name)
    run_command(command)


def source_to_so(build_path, srcs, target_name, compiler, cflags, ldflags, buildin_o=None):
    objs = change_exts(srcs, [('cpp', 'o')])
    if source_to_o(build_path, zip(srcs, objs), compiler, cflags) or\
            not os.path.exists(target_name):
        if buildin_o is not None:
            objs.extend(buildin_o)
        abs_objs = add_path(build_path, objs)
        o_to_so(target_name, abs_objs, compiler, ldflags)


def get_common_flags():
    HOST_NUM_THREADS = config.HOST_NUM_THREADS if config.HOST_NUM_THREADS > 0 else NUM_CPU_CORE
    COMMON_FLAGS = Flags().add_definition('HOST_NUM_THREADS', HOST_NUM_THREADS)
    if config.USING_OPTIMIZATION:
        COMMON_FLAGS.add_string('-O3')
    if config.DEBUG:
        COMMON_FLAGS.add_string('-g')
    COMMON_FLAGS.add_definition('USING_CBLAS', config.USING_CBLAS)
    INC_PATHS.extend(['./cpp/include'])

    if config.USING_OPENCV:
        # 使用opencv库
        INC_PATHS.extend([os.path.join(ANTGO_DEPEND_ROOT,'opencv-install/include')])

    if config.USING_EIGEN:
        # 使用eigen库
        INC_PATHS.extend([os.path.join(ANTGO_DEPEND_ROOT,'eigen')])

    if config.USING_EAGLEEYE:
        # 使用eagleeye库
        INC_PATHS.extend([os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye', f'{sys.platform}-install', 'include')])

    for path in INC_PATHS:
        p = os.path.join(ENV_PATH, path)
        if p:
            COMMON_FLAGS.add_string('-I{}'.format(p))

    return COMMON_FLAGS


def get_build_flag_cpu():
    COMMON_FLAGS = get_common_flags()
    CFLAGS = Flags('-std=c++11').add_definition('USING_CUDA', 0).add_definition('USING_OPENMP', config.USING_OPENMP).\
        add_string(COMMON_FLAGS)
    if not OS_IS_WINDOWS:
        CFLAGS.add_string('-fPIC')
    LDFLAGS = Flags('-lpthread -shared')
    if config.USING_CBLAS:
        LDFLAGS.add_string('-lopenblas')
    if config.USING_OPENMP:
        CFLAGS.add_string('-fopenmp')
        LDFLAGS.add_string('-fopenmp')
    if config.USING_HIGH_LEVEL_WARNINGS:
        CFLAGS.add_string('-Werror -Wall -Wextra -pedantic -Wcast-align -Wcast-qual \
    -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-include-dirs \
    -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow \
    -Wsign-promo -Wundef -fdiagnostics-show-option')

    if config.USING_OPENCV:
        opencv_dir = os.path.join(ANTGO_DEPEND_ROOT, 'opencv-install')
        opencv_libs = ['opencv_calib3d', 'opencv_core', 'opencv_highgui', 'opencv_imgproc', 'opencv_imgcodecs']
        opencv_libs = [f'-l{v}' for v in opencv_libs]
        opencv_libs = ' '.join(opencv_libs)
        LDFLAGS.add_string(f'-L {opencv_dir}/lib {opencv_libs}')
        
    if config.USING_EAGLEEYE:
        eagleeye_lib_dir = os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye', f'{sys.platform}-install', 'libs', 'X86-64')
        LDFLAGS.add_string(f'-L {eagleeye_lib_dir} -leagleeye')
        
    return config.CXX, CFLAGS, LDFLAGS


def get_build_flag_cuda():
    COMMON_FLAGS = get_common_flags()
    CU_FLAGS = Flags('-std=c++11 -x cu -Wno-deprecated-gpu-targets -dc \
    -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES --expt-extended-lambda').\
        add_definition('USING_CUDA', 1).\
        add_definition('USING_HIP', 0).\
        add_string(COMMON_FLAGS)
    if not OS_IS_WINDOWS:
        CU_FLAGS.add_string('--compiler-options "-fPIC"')
    CU_LDFLAGS = Flags('-shared -Wno-deprecated-gpu-targets \
    -L%s/lib64 -lcuda -lcudart' % config.CUDA_DIR)
    if config.USING_CBLAS:
        CU_LDFLAGS.add_string('-lcublas')
    return config.NVCC, CU_FLAGS, CU_LDFLAGS


BUILD_FLAGS = dict(
    cpu=get_build_flag_cpu,
    cuda=get_build_flag_cuda,
)


def get_build_flag(ctx_name):
    flags = BUILD_FLAGS.get(ctx_name, None)()
    assert flags is not None, ValueError(
        'Unsupported Context: {} -('.format(ctx_name))
    return flags


def source_to_so_ctx(build_path, srcs, target_name, ctx_name):
    # 检查依赖库并下载编译
    if config.USING_OPENCV:
        install_path = os.path.join(ANTGO_DEPEND_ROOT, 'opencv-install')
        if not os.path.exists(install_path):
            if not os.path.exists(os.path.join(ANTGO_DEPEND_ROOT, 'opencv')):
                # 下载源码
                os.system(f'cd {ANTGO_DEPEND_ROOT} && git clone https://github.com/opencv/opencv.git -b 3.4')
    
            # 编译
            print('compile opencv')
            os.system(f'cd {ANTGO_DEPEND_ROOT} && cd opencv && mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX={install_path} -D BUILD_DOCS=OFF -D BUILD_EXAMPLES=OFF -D BUILD_opencv_apps=OFF -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=OFF -D BUILD_PERF_TESTS=OFF  -D BUILD_JAVA=OFF -D BUILD_opencv_java=OFF -D BUILD_TESTS=OFF -D WITH_FFMPEG=OFF .. && make -j4 && make install')
            os.system(f'cd {ANTGO_DEPEND_ROOT} && cd opencv && rm -rf build')

            # 添加so的搜索路径 (for linux)
            so_abs_path = os.path.join(install_path, 'lib')
            os.system(f'echo "{so_abs_path}" >> /etc/ld.so.conf && ldconfig')

    if config.USING_EIGEN:
        install_path = os.path.join(ANTGO_DEPEND_ROOT, 'eigen')
        if not os.path.exists(install_path):
            # 下载源码
            os.system(f'cd {ANTGO_DEPEND_ROOT} && git clone https://gitlab.com/libeigen/eigen.git -b 3.3')
    
    if config.USING_EAGLEEYE:
        src_path = os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye')
        if not os.path.exists(src_path):
            # 下载源码
            os.system(f'cd {ANTGO_DEPEND_ROOT} && git clone https://github.com/jianzfb/eagleeye.git')

        # 编译
        install_path = os.path.join(ANTGO_DEPEND_ROOT, 'eagleeye', f'{sys.platform}-install')
        if not os.path.exists(install_path):
            print('compile eagleeye')
            if 'darwin' in sys.platform:
                os.system(f'cd {src_path} && bash osx_build.sh && mv install {sys.platform}-install')
            else:
                os.system(f'cd {src_path} && bash linux_build.sh && mv install {sys.platform}-install')

                # 添加so的搜索路径 (for linux)
                so_abs_path = os.path.join(install_path, 'libs', 'X86-64')
                os.system(f'echo "{so_abs_path}" >> /etc/ld.so.conf && ldconfig')

    # 构建编译信息
    flags = get_build_flag(ctx_name)
    compiler, cflags, ldflags = flags[:3]

    buildin_path = os.path.join(
        config.BUILD_PATH, 'build', 'buildin', ctx_name)
    os.makedirs(buildin_path, exist_ok=True)

    buildin_o = []
    buildin_cpp = []
    for src in ['defines.cpp']:
        fname = os.path.join('cpp', 'src', src)
        buildin_o.append(os.path.join(buildin_path, fname))
        buildin_cpp.append(os.path.join(ENV_PATH, fname))
    buildin_o = change_exts(buildin_o, [('cpp', 'o')])

    for fname in buildin_o:
        if not os.path.exists(fname):
            with build_context():
                source_to_o('.', zip(
                    buildin_cpp, buildin_o), compiler, cflags)
    flags += (buildin_o, )
    source_to_so('.', srcs, target_name, *flags)
