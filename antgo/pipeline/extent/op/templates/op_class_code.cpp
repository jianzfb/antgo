#include "${inc_fname1}"
#include "defines.h"
#include "${inc_fname2}"
#include "eagleeye/basic/DataConvert.h"

namespace eagleeye{
namespace dataflow{
${op_name}::${op_name}(){
    m_${cls_name} = new ${cls_name}(${args_init});
}
${op_name}::~${op_name}(){
    delete m_${cls_name};
}

int ${op_name}::init(std::map<std::string, std::vector<float>> params){
    // ignore
    return 0;
}

int ${op_name}::runOnCpu(const std::vector<Tensor>& input){

// input
${args_convert}

// run
${return_statement} m_${cls_name}->run(${args_run});

// convert output
${output_covert}

// release resource
${args_clear}

    return 0;
}

int ${op_name}::runOnGpu(const std::vector<Tensor>& input){
    return -1;
}
} // namespace dataflow
} // namespace eagleeye
