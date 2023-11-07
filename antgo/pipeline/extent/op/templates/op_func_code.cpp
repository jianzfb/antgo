#include "${inc_fname1}"
#include "defines.h"
#include "${inc_fname2}"
#include "eagleeye/basic/DataConvert.h"

namespace eagleeye{
namespace dataflow{
${op_name}::${op_name}(){
    ${input_default}
    ${output_default} 
    ${const_default}
}
${op_name}::~${op_name}(){
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

// input
// input
${input_create}
${output_create}

${input_init}
${ext_cont_init}

// run
${return_statement} ${func_name}(${args_inst});

// output
${output_export}

    return 0;
}

int ${op_name}::runOnGpu(const std::vector<Tensor>& input){
    return -1;
}

void ${op_name}::clear(){
    // clear some info
}
} // namespace dataflow
} // namespace eagleeye
