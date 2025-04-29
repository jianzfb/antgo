#ifndef _EAGLEEYE_${op_name}_OP_
#define _EAGLEEYE_${op_name}_OP_
#include "eagleeye/engine/nano/dataflow/base.h"
#include "eagleeye/basic/Tensor.h"
#include "defines.h"
#include "eagleeye/basic/DataConvert.h"
#include <string>
#include <vector>
#include <omp.h>
#include "${include_dependent}"

namespace eagleeye{
namespace dataflow{
class ${op_name}:public BaseOp<${input_num}, ${output_num}>{
public:
    using BaseOp<${input_num}, ${output_num}>::init;
    ${op_name}(){
        // for(int thread_i=0; thread_i<${parallel_num}; ++thread_i){
        //     Base* func = ${func_create};
        //     m_funcs.push_back(func);
        // }
        // 创建
        Base* func = ${func_create};
        m_funcs.push_back(func);
        // 初始化
        ${func_init}
    }
    virtual ~${op_name}(){
        // for(int thread_i=0; thread_i<${parallel_num}; ++thread_i){
        //     delete m_funcs[thread_i];
        // }
        delete m_funcs[0];
    }

    virtual int init(std::map<std::string, std::vector<float>> params){
        // for(int thread_i=0; thread_i<${parallel_num}; ++thread_i){
        //     m_funcs[thread_i]->init(params);
        // }
        m_funcs[0]->init(params);
        return 0;
    }
    virtual int init(std::map<std::string, std::vector<std::vector<float>>> params){
        // for(int thread_i=0; thread_i<${parallel_num}; ++thread_i){
        //     m_funcs[thread_i]->init(params);
        // }
         m_funcs[0]->init(params);
        return 0;
    }
    virtual int init(std::map<std::string, std::vector<std::string>> params){
        // for(int thread_i=0; thread_i<${parallel_num}; ++thread_i){
        //     m_funcs[thread_i]->init(params);
        // }
        m_funcs[0]->init(params);
        return 0;
    };

    virtual int runOnCpu(const std::vector<Tensor>& input){
        // split batch dim, as loop_num
        int loop_num = input[0].dims()[0];
        for(int i=1; i<input.size(); ++i){
            if(loop_num < input[i].dims()[0]){
                loop_num = input[i].dims()[0];
            }
        }
        std::vector<std::vector<Tensor>> loop_output(loop_num);

        // parallel loop
        for(int loop_i=0; loop_i<loop_num; ++loop_i){
            // int thread_i = omp_get_thread_num();
            int thread_i = 0;
            int input_num = input.size();
            std::vector<Tensor> slice_input;
            for(int input_i=0; input_i<input_num; ++input_i){
                int slice_i = loop_i % input[input_i].dims()[0];
                int slice_size = input[input_i].numel() / input[input_i].dims()[0];

                Tensor input_i_tensor = input[input_i];
                // 数据
                char* input_i_slice_ptr = input_i_tensor.template cpu<char>() + slice_i * slice_size * input[input_i].elemsize();
                // 形状
                std::vector<int64_t> input_i_slice_shape;
                if(input[input_i].dims().size() > 1){
                    for(int i=1; i<input[input_i].dims().size(); ++i){
                        input_i_slice_shape.push_back(input[input_i].dims()[i]);
                    }
                }
                else{
                    input_i_slice_shape.push_back(1);
                }

                Tensor slice_tensor(
                    input_i_slice_shape,
                    input[input_i].type(),
                    DataFormat::AUTO,
                    (void*)input_i_slice_ptr
                );
                slice_input.push_back(slice_tensor);
            }

            this->m_funcs[thread_i]->runOnCpu(slice_input);

            std::vector<Tensor> out;
            for(int output_i=0; output_i<this->m_funcs[thread_i]->getOutputNum(); ++output_i){
                out.push_back(this->m_funcs[thread_i]->getOutput(output_i).clone());
            }
            loop_output[loop_i] = out;
        }

        if(loop_num == 0){
            for(int output_i=0; output_i<this->m_funcs[0]->getOutputNum(); ++output_i){
                this->m_outputs[output_i] = Tensor(
                    std::vector<int64_t>{0},
                    EAGLEEYE_FLOAT32,
                    DataFormat::AUTO,
                    CPU_BUFFER
                );
            }
            return 0;
        }
        // stack all output
        for(int output_i=0; output_i<this->m_funcs[0]->getOutputNum(); ++output_i){
            // 申请空间
            std::vector<int64_t> output_i_shape;
            output_i_shape.push_back(loop_num);
            for(int i=0; i<loop_output[0][output_i].dims().size(); ++i){
                output_i_shape.push_back(loop_output[0][output_i].dims()[i]);
            }

            if(this->m_outputs[output_i].numel() != loop_num * loop_output[0][output_i].numel()){
                this->m_outputs[output_i] = Tensor(
                    output_i_shape,
                    loop_output[0][output_i].type(),
                    DataFormat::AUTO,
                    CPU_BUFFER
                );
            }
            char* output_i_ptr = this->m_outputs[output_i].cpu<char>();
            int elem_size = this->m_outputs[output_i].elemsize();
            int slice_size = loop_output[0][output_i].numel();

            // 复制内存
            for(int loop_i=0; loop_i<loop_num; ++loop_i){
                char* output_i_slice_ptr = output_i_ptr + loop_i * slice_size * elem_size;
                memcpy(output_i_slice_ptr, loop_output[loop_i][output_i].cpu<char>(), slice_size * elem_size);
            }
        }
        return 0;
    }

    virtual int runOnGpu(const std::vector<Tensor>& input){
        return 0;
    }

private:
    std::vector<Base*> m_funcs;
};
}
}

#endif