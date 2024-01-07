#ifndef _EAGLEEYE_${op_name}_OP_
#define _EAGLEEYE_${op_name}_OP_
#include "eagleeye/engine/nano/dataflow/base.h"
#include "eagleeye/basic/Tensor.h"
#include <string>
#include <vector>
#include "defines.h"
#include "eagleeye/basic/DataConvert.h"
#include "${true_func_include_dependent}"
#include "${false_func_include_dependent}"

namespace eagleeye{
namespace dataflow{
class ${op_name}:public BaseOp<${input_num}, ${output_num}>{
public:
    using BaseOp<${input_num}, ${output_num}>::init;
    ${op_name}(){
        m_true_func = ${true_func_create}          // new XXX();
        m_false_func = ${false_func_create}
    }
    virtual ~${op_name}(){
        delete m_true_func;
        delete m_false_func;
    }

    virtual int init(std::map<std::string, std::vector<float>> params){
        ${true_func_init}
        ${false_func_init}
    }
    virtual int init(std::map<std::string, std::vector<std::vector<float>>> params){return 0;};
    virtual int init(std::map<std::string, std::vector<std::string>> params){return 0;};

    virtual int runOnCpu(const std::vector<Tensor>& input){
        Tensor true_or_false_tensor = input[0];
        bool* true_or_false_ptr = true_or_false_tensor.cpu<bool>();

        std::vector<Tensor> func_input;
        for(int i=1; i<input.size(); ++i){
            func_input.push_back(input[i]);
        }
        if(true_or_false_ptr[0]){
            this->m_true_func->runOnCpu(func_input);
            for(int i=0;i<this->m_true_func->getOutputNum(); ++i){
                this->m_outputs[i]= this->m_true_func->getOutput(i);
            }
        }
        else{
            this->m_false_func->runOnCpu(func_input);
            for(int i=0;i<this->m_false_func->getOutputNum(); ++i){
                this->m_outputs[i]= this->m_false_func->getOutput(i);
            }            
        }
    }
    virtual int runOnGpu(const std::vector<Tensor>& input){
        return 0;
    }

private:
    Base* m_true_func;
    Base* m_false_func;
};
}
}

#endif