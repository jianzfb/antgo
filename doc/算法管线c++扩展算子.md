# 算法管线C++算子扩展
## 目的
支持编写C++函数扩展并支持python端动态编译，从而实现快速算法验证和功能部署。在DAG算子管线下，快速实现python算子和C++算子互调用。

## 例子 C++类扩展
在文件夹testop下，创建如下目录结构
```
testop
    testop.hpp
```

其中，testop.hpp文件，代码如下
```
#include <iostream>
#include "defines.h"
#include <opencv2/core.hpp>

using namespace antgo;
ANTGO_CLASS class TestOp{
public:
    TestOp(){}
    ~TestOp(){}

    void run(const CUCTensor* image, CUCTensor* output){
        // 算子的输入，以const 修饰
        // 获得张量维度信息
        int h = image->dims[0];
        int w = image->dims[1];
        // 获得张量内存数据
        unsighed char* data = image->data;

        // 实现算子功能

    }
};
```

算子输入输出类型对照表
|类名称|类型|numpy类型|
|---|---|---|
|CUCTensor|uint8 张量|np.zeros((10,10), np.uint8)|
|CFTensor|float32 张量|np.zeros((10,10), np.float32)|
|CITensor|int32 张量|np.zeros((10,10), np.int32)|
|CDTensor|float64 张量|np.zeros((10,10), np.float64)|

### 核心概念
* ANTGO_CLASS
    
    用于标记此类为算子
* 构造函数参数说明    
    参数包括标量，如int,float,const char*等

* run函数参数说明
    const 修饰的变量表示输入参数，其余变量表示输出参数
    
    输入和输出参数仅支持Tensor张量类型

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

### Tensor类定义
```
template <typename T>
struct CTensor {
  int64_t dim_size;
  int64_t* dims;
  T* data;
  bool is_assign_inner;
  double timestamp;

  CTensor(){
    this->dim_size = 0;
    this->dims = NULL;
    this->data = NULL;
    this->is_assign_inner = false;
    this->timestamp = 0;
  }
  ~CTensor(){
    this->destroy();
  }

  void destroy(){
    // dim
    if(is_assign_inner && this->dims != NULL){
      delete[] this->dims;
      this->dims = NULL;
    }

    // data
    if(is_assign_inner && this->data != NULL){
      delete[] this->data;
      this->data = NULL;
    }
    is_assign_inner = false;
  }

  void mirror(const T* data, const std::vector<int64_t>& shape){
    // 尝试释放资源
    this->destroy();

    is_assign_inner = false;
    this->data = const_cast<T*>(data);
    this->dim_size = shape.size();
    this->dims = const_cast<int64_t*>(shape.data());
  }

  void create1d(int64_t dim_0){
    if(this->data != NULL){
      if(this->dims[0] == dim_0){
        // 不需要重新分配空间, 清空空间
        if(dim_0 > 0)
          memset(this->data, 0, sizeof(T)*dim_0);
        return;
      }
    }

    this->dim_size = 1;

    // dim
    if(is_assign_inner){
      delete[] this->dims;
    }    
    this->dims = new int64_t[1];
    this->dims[0] = dim_0;

    // data
    if(is_assign_inner){
      delete[] this->data;
    }
    this->data = new T[dim_0];
    if(dim_0 > 0){
      memset(this->data, 0, sizeof(T)*dim_0);
    }
    is_assign_inner = true;
  }

  void create2d(int64_t dim_0, int64_t dim_1){
    if(this->data != NULL){
      if(this->dims[0] == dim_0 && this->dims[1] == dim_1){
        // 不需要重新分配空间, 清空空间
        if(dim_0 > 0 && dim_1 > 0){
          memset(this->data, 0, sizeof(T)*dim_0*dim_1);
        }
        return;
      }
    }
    this->dim_size = 2;

    // dim
    if(is_assign_inner){
      delete[] this->dims;
    }
    this->dims = new int64_t[2];
    this->dims[0] = dim_0; this->dims[1] = dim_1; 

    // data
    if(is_assign_inner){
      delete[] this->data;
    }

    this->data = new T[dim_0*dim_1];
    if(dim_0*dim_1 > 0){
      memset(this->data, 0, sizeof(T)*dim_0*dim_1);
    }
    is_assign_inner = true;
  }

  void create3d(int64_t dim_0, int64_t dim_1, int64_t dim_2){
    if(this->data != NULL){
      if(this->dims[0] == dim_0 && this->dims[1] == dim_1 && this->dims[2] == dim_2){
        // 不需要重新分配空间, 清空空间
        if(dim_0 > 0 && dim_1 > 0 && dim_2 > 0){
          memset(this->data, 0, sizeof(T)*dim_0*dim_1*dim_2);
        }
        return;
      }
    }

    this->dim_size = 3;
    // dim
    if(is_assign_inner){
      delete[] this->dims;
    }
    this->dims = new int64_t[3];
    this->dims[0] = dim_0; this->dims[1] = dim_1; this->dims[2] = dim_2; 

    // data
    if(is_assign_inner){
      delete[] this->data;
    }
    this->data = new T[dim_0*dim_1*dim_2];
    if(dim_0*dim_1*dim_2 > 0){
      memset(this->data, 0, sizeof(T)*dim_0*dim_1*dim_2);
    }        
    is_assign_inner = true;
  }

  void create4d(int64_t dim_0, int64_t dim_1, int64_t dim_2, int64_t dim_3){
    if(this->data != NULL){
      if(this->dims[0] == dim_0 && 
          this->dims[1] == dim_1 && 
          this->dims[2] == dim_2 && 
          this->dims[3] == dim_3){
        // 不需要重新分配空间, 清空空间
        if(dim_0 > 0 && dim_1 > 0 && dim_2 > 0 && dim_3 > 0){
          memset(this->data, 0, sizeof(T)*dim_0*dim_1*dim_2*dim_3);
        }
        return;
      }
    }

    this->dim_size = 4;

    // dim
    if(is_assign_inner){
      delete[] this->dims;
    }
    this->dims = new int64_t[4];
    this->dims[0] = dim_0; this->dims[1] = dim_1; this->dims[2] = dim_2; this->dims[3] = dim_3; 

    // data
    if(is_assign_inner){
      delete[] this->data;
    }
    this->data = new T[dim_0*dim_1*dim_2*dim_3];
    if(dim_0*dim_1*dim_2*dim_3 > 0){
      memset(this->data, 0, sizeof(T)*dim_0*dim_1*dim_2*dim_3);
    }
    is_assign_inner = true;
  }  

  void create5d(int64_t dim_0, int64_t dim_1, int64_t dim_2, int64_t dim_3, int64_t dim_4){
    if(this->data != NULL){
      if(this->dims[0] == dim_0 && 
          this->dims[1] == dim_1 && 
          this->dims[2] == dim_2 && 
          this->dims[3] == dim_3 && 
          this->dims[4] == dim_4){
        // 不需要重新分配空间, 清空空间
        if(dim_0 > 0 && dim_1 > 0 && dim_2 > 0 && dim_3 > 0 && dim_4 > 0){
          memset(this->data, 0, sizeof(T)*dim_0*dim_1*dim_2*dim_3*dim_4);
        }        
        return;
      }
    }

    this->dim_size = 5;

    // dim
    if(is_assign_inner){
      delete[] this->dims;
    }
    this->dims = new int64_t[5];
    this->dims[0] = dim_0; this->dims[1] = dim_1; this->dims[2] = dim_2; this->dims[3] = dim_3; this->dims[4] = dim_4; 

    // data
    if(is_assign_inner){
      delete[] this->data;
    }
    this->data = new T[dim_0*dim_1*dim_2*dim_3*dim_4];
    if(dim_0*dim_1*dim_2*dim_3*dim_4 > 0){
      memset(this->data, 0, sizeof(T)*dim_0*dim_1*dim_2*dim_3*dim_4);
    }
    is_assign_inner = true;
  }
};
```