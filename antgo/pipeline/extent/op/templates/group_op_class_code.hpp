#ifndef _EAGLEEYE_${op_name}_OP_
#define _EAGLEEYE_${op_name}_OP_
#include "eagleeye/engine/nano/dataflow/base.h"
#include "eagleeye/basic/Tensor.h"
#include <string>
#include <vector>
#include "defines.h"
#include "eagleeye/basic/DataConvert.h"
#include "eagleeye/engine/nano/op/group_op.h"
${include_dependent}

namespace eagleeye{
namespace dataflow{
class ${op_name}:public GroupOp<${input_num}, ${output_num}>{
public:
    using GroupOp<${input_num}, ${output_num}>::init;
    using GroupOp<${input_num}, ${output_num}>::runOnCpu;
    using GroupOp<${input_num}, ${output_num}>::runOnGpu;

    ${op_name}(){
        // create
        this->m_ops = std::vector<Base*>{${group_op_create}};

        // init op
        ${group_op_param_init}

        // init relation
        ${group_relation_init}
    }
    virtual ~${op_name}(){}

    virtual int init(std::map<std::string, std::vector<float>> params){}
    virtual int init(std::map<std::string, std::vector<std::vector<float>>> params){return 0;};
    virtual int init(std::map<std::string, std::vector<std::string>> params){return 0;};
    virtual int init(std::map<std::string, void*> params){return 0;}    
};
}
}

#endif