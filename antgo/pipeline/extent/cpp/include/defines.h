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
  size_t dim_size;
  size_t* dims;
  T* data;
  bool is_assign_inner;

  T &operator[](int i) { return data[i]; }
  T &operator[](int i) const { return data[i]; }

  void destroy(){
    // dim
    if(is_assign_inner){
      delete[] this->dims;
    }

    // data
    if(is_assign_inner){
      delete[] this->data;
    }
  }

  void create1d(size_t dim_0){
    this->dim_size = 1;

    // dim
    if(is_assign_inner){
      delete[] this->dims;
    }    
    this->dims = new size_t[1];
    this->dims[0] = dim_0;

    // data
    if(is_assign_inner){
      delete[] this->data;
    }
    this->data = new T[dim_0];
    is_assign_inner = true;
  }

  void create2d(size_t dim_0, size_t dim_1){
    this->dim_size = 2;

    // dim
    if(is_assign_inner){
      delete[] this->dims;
    }    
    this->dims = new size_t[2];
    this->dims[0] = dim_0; this->dims[1] = dim_1; 

    // data
    if(is_assign_inner){
      delete[] this->data;
    }
    this->data = new T[dim_0*dim_1];
    is_assign_inner = true;
  }

  void create3d(size_t dim_0, size_t dim_1, size_t dim_2){
    this->dim_size = 3;
    // dim
    if(is_assign_inner){
      delete[] this->dims;
    }
    this->dims = new size_t[3];
    this->dims[0] = dim_0; this->dims[1] = dim_1; this->dims[2] = dim_2; 

    // data
    if(is_assign_inner){
      delete[] this->data;
    }
    this->data = new T[dim_0*dim_1*dim_2];
    is_assign_inner = true;
  }

  void create4d(size_t dim_0, size_t dim_1, size_t dim_2, size_t dim_3){
    this->dim_size = 4;

    // dim
    if(is_assign_inner){
      delete[] this->dims;
    }
    this->dims = new size_t[4];
    this->dims[0] = dim_0; this->dims[1] = dim_1; this->dims[2] = dim_2; this->dims[3] = dim_3; 

    // data
    if(is_assign_inner){
      delete[] this->data;
    }
    this->data = new T[dim_0*dim_1*dim_2*dim_3];
    is_assign_inner = true;
  }  

  void create5d(size_t dim_0, size_t dim_1, size_t dim_2, size_t dim_3, size_t dim_4){
    this->dim_size = 5;

    // dim
    if(is_assign_inner){
      delete[] this->dims;
    }
    this->dims = new size_t[5];
    this->dims[0] = dim_0; this->dims[1] = dim_1; this->dims[2] = dim_2; this->dims[3] = dim_3; this->dims[4] = dim_4; 

    // data
    if(is_assign_inner){
      delete[] this->data;
    }
    this->data = new T[dim_0*dim_1*dim_2*dim_3*dim_4];
    is_assign_inner = true;
  }

};

typedef CTensor<double> CDTensor;
typedef CTensor<float> CFTensor;
typedef CTensor<int> CITensor;
typedef CTensor<unsigned char> CUCTensor;


template <typename F, typename T>
inline void mobula_map(F func, const T *data, const int n,
                                     const int stride = 1, T *out = nullptr) {
  if (out == nullptr) out = const_cast<T *>(data);
  for (int i = 0, j = 0; i < n; ++i, j += stride) {
    out[j] = func(data[j]);
  }
}

template <typename F, typename T>
inline void mobula_reduce(F func, const T *data, const int n,
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
