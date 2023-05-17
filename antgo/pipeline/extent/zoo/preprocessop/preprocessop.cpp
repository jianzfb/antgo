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
    std::cout<<"meanv"<<std::endl;
    std::cout<<meanv->dim_size<<std::endl;
    std::cout<<meanv->dims[0]<<std::endl;
    std::cout<<meanv->data[0]<<" "<<meanv->data[1]<<" "<<meanv->data[2]<<std::endl;
    std::cout<<"stdv"<<std::endl;
    std::cout<<stdv->dim_size<<std::endl;
    std::cout<<stdv->dims[0]<<std::endl;
    std::cout<<stdv->data[0]<<" "<<stdv->data[1]<<" "<<stdv->data[2]<<std::endl;
    
    std::cout<<">>>>"<<std::endl;
    std::shared_ptr<size_t> _dim_mem = std::shared_ptr<size_t>(new size_t[3], [](size_t* arr) { delete [] arr; });
    std::cout<<"<<<<"<<std::endl;

    output->create3d(5, 4, 3);
    for(int i=0; i<5*4*3; ++i){
        output->data[i] = -i;
    }
}