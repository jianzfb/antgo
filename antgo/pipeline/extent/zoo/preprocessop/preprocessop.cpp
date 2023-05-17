#include <iostream>
#include "defines.h"

using namespace mobula;
MOBULA_FUNC void preprocess_func(
    const CFTensor* meanv, 
    const CFTensor* stdv, 
    const CITensor* permute, 
    const bool needed_expand_batch_dim, 
    const bool needed_chw, 
    const CUCTensor* image, CFTensor* output){
    // std::cout<<"here"<<std::endl;
    // std::cout<<a<<std::endl;
    // std::cout<<a->dim_num<<std::endl;
    // std::cout<<"b"<<std::endl;
    // std::cout<<"a dims "<<a->dims<<std::endl;

    // std::cout<<"mm"<<std::endl;
    // std::cout<<a->data<<std::endl;
    // std::cout<<"nn"<<std::endl;
    // std::cout<<*a->data<<std::endl;
    // std::cout<<"-- "<<a->data[0]<<" -- "<<std::endl;
    // std::cout<<"nn"<<std::endl;
    // for(int i=0; i<a->dims[0]; ++i){
    //     b->data[i] = a->data[i] + i;
    //     std::cout<<"i "<<i<<" "<<"data "<<b->data[i]<<std::endl;
    // }
    std::cout<<"meanv"<<std::endl;
    std::cout<<meanv->dim_num<<std::endl;
    std::cout<<"stdv"<<std::endl;
    std::cout<<stdv->dim_num<<std::endl;

    float* tensor_data = (float*)(malloc(sizeof(float)* 5*3*4));
    for(int i=0; i<5*3*4; ++i){
        tensor_data[i] = -i;
    }
    size_t* tensor_dims = (size_t*)(malloc(sizeof(size_t) * 3));
    tensor_dims[0] = 5;
    tensor_dims[1] = 3;
    tensor_dims[2] = 4;
    output->data = tensor_data;
    output->dims = tensor_dims;
    output->dim_num = 3;
}