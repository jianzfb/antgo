# C++算子扩展
## 目的
支持编写C++函数对DAG算子进行扩展，从而实现快速算法验证和功能部署。

## 例子
```
#include <iostream>
#include "defines.h"
#include <opencv2/core.hpp>
#include "Eigen/Dense"
#include "eagleeye/basic/Matrix.h"

using namespace eagleeye;
EAGLEEYE_FUNC void test_func(const CUCTensor* image, CUCTensor* output){
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

## 解释
### 输出输入变量类型
支持标准scalar类型和tensor类型
* scalar类型
    ```
    float,bool,int,std::string
    ```
* tensor类型

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
    np.array((), dtype=np.float64)
    np.array((), dtype=np.float32)
    np.array((), dtype=np.int32)
    np.array((), dtype=np.uint8)
    ```

    在c++中我们可以通过如下方式创建tensor
    ```
        tensor->create1d(dim_0);                            // 创建1维tensor
        tensor->create2d(dim_0, dim_1);                     // 创建2维tensor
        tensor->create3d(dim_0, dim_1, dim_2);              // 创建3维tensor
        tensor->create4d(dim_0, dim_1, dim_2, dim_3);       // 创建4维tensor
        tensor->create5d(dim_0, dim_1, dim_2, dim_3, dim_4);// 创建5维tensor
    ```

### 函数定义说明
* EAGLEEYE_FUNC
    
    用于标记此函数是算子
* 函数参数
    
    const 修饰的变量表示输入参数，其余变量表示输出参数

    输入参数包括标量和tensor类型，输出参数仅包括tensor类型