#include "${inc_fname1}"
#include "defines.h"
#include "${inc_fname2}"
#include "eagleeye/basic/DataConvert.h"

namespace eagleeye{
namespace dataflow{
${op_name}::${op_name}(){
    m_${cls_name} = NULL;
    ${input_default}
    ${output_default} 
    ${const_default}
}

${op_name}::~${op_name}(){
    if(m_${cls_name} != NULL){
        delete m_${cls_name};
    }
    ${input_delete}
    ${output_delete} 
    ${const_delete}
}

int ${op_name}::init(std::map<std::string, std::vector<float>> params){
    // ignore
    ${const_init}
    return 0;
}

int ${op_name}::runOnCpu(const std::vector<Tensor>& input){
if(m_${cls_name} == NULL){
    m_${cls_name} = new ${cls_name}(${args_init});
}
// input
${input_create}
${output_create}

${input_init}
${ext_cont_init}

// run
${return_statement} m_${cls_name}->run(${args_run});

// output
${output_export}

    return 0;
}

int ${op_name}::runOnGpu(const std::vector<Tensor>& input){
    return -1;
}
} // namespace dataflow
} // namespace eagleeye
