#ifndef _EAGLEEYE_${op_name}_OP_
#define _EAGLEEYE_${op_name}_OP_
#include "eagleeye/engine/nano/dataflow/base.h"
#include "eagleeye/basic/Tensor.h"
#include <string>
#include <vector>
#include "defines.h"
#include "eagleeye/basic/DataConvert.h"

namespace eagleeye{
namespace dataflow{
class ${op_name}:public BaseOp<${input_num}, ${output_num}>{
public:
    ${op_name}();
    virtual ~${op_name}();

    virtual int init(std::map<std::string, std::vector<float>> params);
    virtual int init(std::map<std::string, std::vector<std::vector<float>>> params){return 0;};
    virtual int init(std::map<std::string, std::vector<std::string>> params){return 0;};
    virtual int runOnCpu(const std::vector<Tensor>& input);
    virtual int runOnGpu(const std::vector<Tensor>& input);

private:
    ${input_define}
    ${output_define}
    ${const_define}
};

} // namespace dataflow
} // namespace eagleeye


#endif