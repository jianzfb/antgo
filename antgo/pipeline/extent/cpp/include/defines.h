#ifndef ANTGO_INCLUDE_DEFINES_H_
#define ANTGO_INCLUDE_DEFINES_H_

#include <array>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "./ctypes.h"
#include "./context.h"
namespace antgo {
template <typename T>
struct CTensor {
  int64_t dim_size;
  int64_t* dims;
  T* data;
  bool is_assign_inner;

  CTensor(){
    this->dim_size = 0;
    this->dims = NULL;
    this->data = NULL;
    this->is_assign_inner = false;
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

typedef CTensor<double> CDTensor;
typedef CTensor<float> CFTensor;
typedef CTensor<int> CITensor;
typedef CTensor<unsigned char> CUCTensor;


template <typename F, typename T>
inline void antgo_map(F func, const T *data, const int n,
                                     const int stride = 1, T *out = nullptr) {
  if (out == nullptr) out = const_cast<T *>(data);
  for (int i = 0, j = 0; i < n; ++i, j += stride) {
    out[j] = func(data[j]);
  }
}

template <typename F, typename T>
inline void antgo_reduce(F func, const T *data, const int n,
                                        const int stride = 1,
                                        T *out = nullptr) {
  if (out == nullptr) out = const_cast<T *>(data);
  T &val = out[0];
  val = data[0];
  for (int i = 1, j = stride; i < n; ++i, j += stride) {
    val = func(val, data[j]);
  }
}

inline int get_middle_loop_offset(const int i,
                                                const int middle_size,
                                                const int inner_size) {
  // &a[outer_size][0][inner_size] = &a[j]
  const int inner_i = i % inner_size;
  const int outer_i = i / inner_size;
  return outer_i * middle_size * inner_size + inner_i;  // j
}

template <typename T>
T ADD_FUNC(const T &a, const T &b) {
  return a + b;
}

template <typename T>
T MAX_FUNC(const T &a, const T &b) {
  return a > b ? a : b;
}

}  // namespace antgo

#endif  // ANTGO_INCLUDE_DEFINES_H_
