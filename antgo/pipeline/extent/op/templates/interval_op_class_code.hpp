#ifndef _EAGLEEYE_${op_name}_OP_
#define _EAGLEEYE_${op_name}_OP_
#include "eagleeye/engine/nano/dataflow/base.h"
#include "eagleeye/basic/Tensor.h"
#include <string>
#include <vector>
#include "defines.h"
#include "eagleeye/basic/DataConvert.h"
#include "${include_dependent}"

namespace eagleeye{
namespace dataflow{
class ${op_name}:public BaseOp<${input_num}, ${output_num}>{
public:
    using BaseOp<${input_num}, ${output_num}>::init;
    ${op_name}(){
        m_func = ${func_create};
        this->m_count = 0;
        this->m_interval = 1;
    }
    virtual ~${op_name}(){
        delete m_func;
    }

    virtual int init(std::map<std::string, std::vector<float>> params){
        ${func_init}
    }
    virtual int init(std::map<std::string, std::vector<std::vector<float>>> params){return 0;};
    virtual int init(std::map<std::string, std::vector<std::string>> params){
        m_func->init(params);
        return 0;
    };

    virtual int runOnCpu(const std::vector<Tensor>& input){
        if(this->m_count % this->m_interval == 0){
            this->m_func->runOnCpu(input);
            for(int output_i = 0; output_i<this->m_func->getOutputNum(); ++output_i){
                this->m_outputs[output_i] = this->m_func->getOutput(output_i);
            }
        }

        this->m_count += 1;
        return 0;
    }

    virtual int runOnGpu(const std::vector<Tensor>& input){
        return 0;
    }

private:
    Base* m_func;
    int m_count;
    int m_interval;
};
}
}

#endif