#include "${node_name}.cpp"
#include "defines.h"
#include "eagleeye/basic/DataConvert.h"

namespace eagleeye{
namespace dataflow{
${node_name}::${node_name}(){}
${node_name}::~${node_name}(){}

int ${node_name}::init(std::map<std::string, std::vector<float>> params){
    // ignore
    return 0;
}

int ${node_name}::runOnCpu(const std::vector<Tensor>& input){

    // inputt
    // CFTensor* a = convert_cftensor_tensor(input[0]);
    // CFTensor* b = new CFTensor();
    ${args_convert}

    // run
    ${return_statement} ${func_name}(${args_inst});

    // output
    // m_outputs[0] = convert_cftensor_tensor(b)
    ${assign_output}

    // clear
    // a->destroy();
    ${args_clear}

    return 0;
}

int ${node_name}::runOnGpu(const std::vector<Tensor>& input){
    return -1;
}
} // namespace dataflow
} // namespace eagleeye
