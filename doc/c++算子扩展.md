# C++算子扩展
## 目的
支持编写C++函数扩展并支持python端动态编译，从而实现快速算法验证和功能部署。在DAG算子管线下，快速实现python算子和C++算子互调用。


## 例子1 C++函数扩展
在文件夹testclass下，创建如下目录结构
```
testclass
    -demoop1
        demoop1.cpp
```

其中，demoop1.cpp文件，代码如下
```
#include <iostream>
#include "defines.h"
#include <opencv2/core.hpp>
#include "Eigen/Dense"
#include "eagleeye/basic/Matrix.h"

using namespace eagleeye;
ANTGO_FUNC void demo_func(const CUCTensor* image, CUCTensor* output){
    // 获得输入图像的高和宽
    size_t height = image->dims[0];
    size_t width = image->dims[1];

    // 验证opencv第三方库
    cv::Mat check_image(height, width, CV_8UC3);
    memcpy(check_image.data, image->data, sizeof(unsigned char)*height*width*3);
    
    // 验证eigen第三方库
    Eigen::MatrixXd m=Eigen::MatrixXd::Zero(2,3);
    std::cout<<m<<std::endl;

    // 验证eagleeye核心库
    Matrix<float> ll(3,4);
    for(int i=0; i<3; ++i){
        for (int j=0; j<4; ++j){
            ll.at(i,j) = i*4+j;
        }
    }

    // 输出结果
    output->create3d(height, width, 3);
    memcpy(output->data, image->data, sizeof(unsigned char)*height*width*3);
}
```

### 核心概念
* ANTGO_FUNC
    
    用于标记此函数是算子
* 参数说明
    
    const 修饰的变量表示输入参数，其余变量表示输出参数
    
    输入参数包括标量和tensor类型，输出参数仅包括tensor类型

## 例子2 C++类扩展
在文件夹testclass下，创建如下目录结构
```
testclass
    -demoop2
        demoop2.cpp
```

其中，demoop2.cpp文件，代码如下
```
#include <iostream>
#include "defines.h"
#include <opencv2/core.hpp>
#include "Eigen/Dense"
#include "eagleeye/basic/Matrix.h"

using namespace eagleeye;
ANTGO_CLASS DemoCls{
public:
    DemoCls(){
    }

    void run(const CUCTensor* image, CUCTensor* output){
        // do something
    }
};
```

### 核心概念
* ANTGO_CLASS
    
    用于标记此类为算子
* 构造函数参数说明    
    参数包括标量和tensor类型

* run函数参数说明
    const 修饰的变量表示输入参数，其余变量表示输出参数
    
    输入参数包括标量和tensor类型，输出参数仅包括tensor类型

## 基础类型定义
### 输入输出变量类型
支持标准Scalar类型和Tensor类型
* Scalar类型
    ```
    float,bool,int,std::string
    ```
* Tensor类型

    c++类型与numpy类型的对应关系
    
    c++变量类型
    ```
    typedef CTensor<double> CDTensor;
    typedef CTensor<float> CFTensor;
    typedef CTensor<int> CITensor;
    typedef CTensor<unsigned char> CUCTensor;
    ```
    
    numpy变量类型
    ```
    np.array((1,4), dtype=np.float64)
    np.array((1,4), dtype=np.float32)
    np.array((1,4), dtype=np.int32)
    np.array((1,4), dtype=np.uint8)
    ```

    在c++中我们可以通过如下方式创建tensor
    ```
    CFTensor* tensor = new CFTensor();
    tensor->create1d(dim_0);                            // 创建1维tensor
    tensor->create2d(dim_0, dim_1);                     // 创建2维tensor
    tensor->create3d(dim_0, dim_1, dim_2);              // 创建3维tensor
    tensor->create4d(dim_0, dim_1, dim_2, dim_3);       // 创建4维tensor
    tensor->create5d(dim_0, dim_1, dim_2, dim_3, dim_4);// 创建5维tensor
    ```

## Python调用算子
### 常规调用
```
import os
import sys
import numpy as np
from antgo.pipeline import *
from antgo.pipeline.functional.data_collection import *
from antgo.pipeline.functional import *
from antgo.pipeline.extent import op
from antgo.pipeline import extent

op.load('demoop1', '/workspace/project/testclass')
op.load('demoop2', '/workspace/project/testclass')
input = np.zeros((10,10), dtype=np.uint8)
# 函数算子
out = extent.func.demo_func(input, np.empty((), dtype=np.uint8))
# 类算子
out = extent.func.DemoCls(input, np.empty((), dtype=np.uint8))
```
