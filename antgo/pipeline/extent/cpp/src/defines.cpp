#include "defines.h"

extern "C" {
using namespace antgo;

ANTGO_API void destroy_cdtensor(CDTensor* t){
    t->destroy();
}

ANTGO_API void destroy_cftensor(CFTensor* t){
    t->destroy();
}

ANTGO_API void destroy_citensor(CITensor* t){
    t->destroy();
}

ANTGO_API void destroy_cuctensor(CUCTensor* t){
    t->destroy();
}

}
