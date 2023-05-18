#ifndef _EAGLEEYE_${node_name}_OP_
#define _EAGLEEYE_${node_name}_OP_
#include "eagleeye/engine/nano/dataflow/base.h"
#include "eagleeye/basic/Tensor.h"
#include <string>
#include <vector>

namespace eagleeye{
namespace dataflow{
class ${node_name}:public BaseOp<Tensor, ${input_num}, ${output_num}>{
public:
    ${node_name}();
    virtual ~${node_name}();
    
    virtual int init(std::map<std::string, std::vector<float>> params);
    virtual int runOnCpu(const std::vector<Tensor>& input);
    virtual int runOnGpu(const std::vector<Tensor>& input);
};

} // namespace dataflow
} // namespace eagleeye


#endif