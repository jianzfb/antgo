#ifndef _EAGLEEYE_${op_name}_OP_
#define _EAGLEEYE_${op_name}_OP_
#include "eagleeye/engine/nano/dataflow/base.h"
#include "eagleeye/basic/Tensor.h"
#include <string>
#include <vector>
#include "defines.h"
#include "eagleeye/basic/DataConvert.h"
#include "${det_func_include_dependent}"
#include "${tracking_func_include_dependent}"

namespace eagleeye{
namespace dataflow{
class ${op_name}:public BaseOp<${input_num}, ${output_num}>{
public:
    using BaseOp<${input_num}, ${output_num}>::init;
    ${op_name}(){
        m_det_func = ${det_func_create}          // new XXX();
        m_tracking_func = ${tracking_func_create}
        this->m_count = 0;
        this->m_interval = 1;
        this->m_only_once = false;

        this->m_det_func_arg_num = ${input_num} - ${output_num};
        this->m_tracking_func_update_arg_num = ${input_num} - this->m_det_func_arg_num;
        this->m_tracking_func_res_arg_num = ${output_num};
        this->m_tracking_func_arg_num = this->m_det_func_arg_num + m_tracking_func_res_arg_num + m_tracking_func_update_arg_num;
    }

    virtual ~${op_name}(){
        delete m_det_func;
        if(this->m_tracking_func != NULL){
            delete m_tracking_func;
        }
    }

    virtual int init(std::map<std::string, std::vector<float>> params){
        ${det_func_init}
        ${tracking_func_init}

        if(params.find("interval") != params.end()){
            this->m_interval = int(params["interval"][0]);
        }
        if(params.find("only_once") != params.end()){
            this->m_only_once = bool(int(params["only_once"][0]));
        }
    }
    virtual int init(std::map<std::string, std::vector<std::vector<float>>> params){return 0;};
    virtual int init(std::map<std::string, std::vector<std::string>> params){
        m_det_func->init(params);
        if(this->m_tracking_func != NULL){
            m_tracking_func->init(params);
        }
        return 0;
    };

    virtual int runOnCpu(const std::vector<Tensor>& input){
        bool is_call_det = false;
        if(this->m_count == 0){
            is_call_det = true;
        }
        else if(this->m_interval > 0 && this->m_count % this->m_interval == 0){
            is_call_det = true;
        }
        else if(this->m_tracking_func_update_arg_num > 0){
            if(input[this->m_det_func_arg_num].dims()[0] > 0){
                is_call_det = true;
            }
        }
        else if(this->m_interval == 0){
            is_call_det = true;
        }

        if(this->m_only_once){
            is_call_det = false;
        }

        this->m_count += 1;

        if(is_call_det){
            // 使用det_func计算
            this->m_det_func->runOnCpu(input);
            for(int i=0;i<this->m_det_func->getOutputNum(); ++i){
                this->m_outputs[i] = this->m_det_func->getOutput(i);
            }

            // 更新tracking_func状态
            std::vector<Tensor> tracking_input;
            for(int i=0; i<this->m_det_func_arg_num; ++i){
                tracking_input.push_back(input[i]);
            }
            for(int i=0; i<m_tracking_func_res_arg_num; ++i){
                tracking_input.push_back(this->m_outputs[i]);
            }
            for(int i=0; i<m_tracking_func_update_arg_num; ++i){
                // 加入空tensor
                tracking_input.push_back(Tensor());
            }
            this->m_tracking_func->runOnCpu(tracking_input);
            return 0;
        }

        if(m_tracking_func != NULL){
            std::vector<Tensor> tracking_input;
            for(int i=0; i<this->m_det_func_arg_num; ++i){
                tracking_input.push_back(input[i]);
            }
            for(int i=0; i<m_tracking_func_res_arg_num; ++i){
                // 加入空tensor
                tracking_input.push_back(Tensor());
            }
            for(int i=0; i<m_tracking_func_update_arg_num; ++i){
                tracking_input.push_back(input[this->m_det_func_arg_num+i]);
            }
            this->m_tracking_func->runOnCpu(tracking_input);
            for(int i=0; i<this->m_tracking_func->getOutputNum(); ++i){
                this->m_outputs[i] = this->m_tracking_func->getOutput(i);
            }
            return 0;
        }

        // 直接使用update input作为输出
        for(int i=0; i<m_tracking_func_update_arg_num; ++i){
            this->m_outputs[i] = input[this->m_det_func_arg_num+i];
        }
        return 0;
    }

    virtual int runOnGpu(const std::vector<Tensor>& input){
        return 0;
    }

private:
    Base* m_det_func;
    Base* m_tracking_func;
    
    int m_interval;
    int m_only_once;
    int m_count;

    int m_det_func_arg_num;
    int m_tracking_func_update_arg_num;
    int m_tracking_func_res_arg_num;
    int m_tracking_func_arg_num;
};
}    
}

#endif