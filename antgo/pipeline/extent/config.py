from .utils import with_metaclass


class DefaultConfig:
    TARGET = 'antgo_op'
    BUILD_PATH = './.temp/'
    BUILD_IN_LOCAL_PATH = True
    SHOW_BUILDING_COMMAND = False
    MAX_BUILDING_WORKER_NUM = 8

    DEBUG = False
    USING_OPENMP = False
    USING_CBLAS = False
    USING_OPENCV = False
    USING_EIGEN = False
    USING_EAGLEEYE = False
    HOST_NUM_THREADS = 0  # 0 : auto
    USING_HIGH_LEVEL_WARNINGS = False
    USING_OPTIMIZATION = True
    USING_ASYNC_EXEC = True
    GPU_BACKEND = 'cuda'

    CXX = 'g++'
    NVCC = 'nvcc'
    HIPCC = 'hipcc'
    CUDA_DIR = '/opt/cuda'
    HIP_DIR = '/opt/rocm/hip'


class Config:
    def __init__(self):
        for name in dir(DefaultConfig):
            if not name.startswith('_'):
                self.__dict__[name] = getattr(DefaultConfig, name)

    def __setattr__(self, name, value):
        data = self.__dict__.get(name, None)
        if data is None:
            raise AttributeError("Config has no attribute '{}'".format(name))
        target_type = type(data)
        value_type = type(value)
        if target_type is not value_type:
            raise TypeError('The type of config attribute `{}` is not consistent, target {} vs value {}.'.format(
                name, target_type, value_type))
        self.__dict__[name] = value


config = Config()


class TempConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.old_config = dict()

    def __enter__(self):
        for k, v in self.kwargs.items():
            if not hasattr(config, k):
                raise AttributeError(
                    "'mobula.config' object has no attribute '{}'".format(k))
            self.old_config[k] = getattr(config, k)
            setattr(config, k, v)

    def __exit__(self, *dummy):
        for k, v in self.old_config.items():
            setattr(config, k, v)


Config.TempConfig = TempConfig
