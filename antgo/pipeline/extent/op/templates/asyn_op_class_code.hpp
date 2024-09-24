#ifndef _EAGLEEYE_${op_name}_OP_
#define _EAGLEEYE_${op_name}_OP_
#include "eagleeye/engine/nano/dataflow/base.h"
#include "eagleeye/basic/Tensor.h"
#include <string>
#include <vector>
#include "defines.h"
#include "eagleeye/basic/DataConvert.h"
#include <functional>
#include <thread>
#include "${include_dependent}"


namespace eagleeye{
namespace dataflow{
class ${op_name}:public BaseOp<${input_num}, ${output_num}>{
public:
    using BaseOp<${input_num}, ${output_num}>::init;
    ${op_name}(){
        m_op_func = ${func_create}
        m_op_thread = std::thread(std::bind(&${op_name}::_run, this));
        m_thread_status = true;

        m_run_finish = false;
    }

    virtual ~${op_name}(){
        // stop thread
        m_thread_status = false;
        std::unique_lock<std::mutex> in_locker(this->m_in_mu);
        this->m_input_queue.push(std::vector<Tensor>{Tensor()});
        in_locker.unlock();
        this->m_in_cond.notify_all();

        if(m_op_thread.joinable()){
            m_op_thread.join();
        }

        // delete op
        delete m_op_func;
    }

    virtual int init(std::map<std::string, std::vector<float>> params){
        m_op_func->init(params);
        ${func_init}
        return 0;
    }
    virtual int init(std::map<std::string, std::vector<std::vector<float>>> params){return 0;};
    virtual int init(std::map<std::string, std::vector<std::string>> params){
        m_op_func->init(params);
        return 0;
    };

    virtual int runOnCpu(const std::vector<Tensor>& input){
        // push data into queue
        std::unique_lock<std::mutex> in_locker(this->m_in_mu);
        m_input_queue.push(input);
        in_locker.unlock();
        this->m_in_cond.notify_all();

        // no block, directly return
        return 0;
    }
    virtual int runOnGpu(const std::vector<Tensor>& input){
        return 0;
    }

    void _run(){
        while(m_thread_status){
            // step 1: get data
            std::unique_lock<std::mutex> in_locker(this->m_in_mu);
            while(this->m_input_queue.size() == 0){
                this->m_in_cond.wait(in_locker);
                if(this->m_input_queue.size() > 0){
                    break;
                }
            }
            std::vector<Tensor> in_data = m_input_queue.front();
            m_input_queue.pop();
            in_locker.unlock();

            bool is_ready = true;
            for(int i=0; i<in_data.size(); ++i){
                if(in_data[i].empty() || in_data[i].dims()[0] == 0){
                    is_ready = false;

                    // empty data, return directly
                    for(int k=0;k<this->m_op_func->getOutputNum(); ++k){
                        m_outputs[k] = Tensor(
                            std::vector<int64_t>{0},
                            EAGLEEYE_FLOAT32,
                            DataFormat::AUTO,
                            CPU_BUFFER
                        );
                    }
                    m_run_finish = true;
                    break;
                }
            }
            if(!is_ready){
                continue;
            }
            m_run_finish = false;

            // step 2: process data
            m_op_func->runOnCpu(in_data);
            for(int i=0;i<this->m_op_func->getOutputNum(); ++i){
                m_outputs[i] = this->m_op_func->getOutput(i);
            }
            m_run_finish = true;
        }
    }

    /**
     * @brief Get the Output Tensor object
     * 
     * @param index 
     * @return Tensor& 
     */
    virtual Tensor& getOutput(int index){
        while(!m_run_finish){
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
        }

        return this->m_outputs[index];
    }

    virtual int getOutputSize(int index){
        if(index >= this->m_output_num){
            EAGLEEYE_LOGD("Index out of bounds.");
            return 0;
        }

        while(!m_run_finish){
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
        }
        return this->m_outputs[index].blobsize();
    }

    /**
     * @brief get cpu data, from 
     */
    virtual int fetch(void*& data, std::vector<int64_t>& shape, EagleeyeType& type, int index, bool block){
        while(!m_run_finish){
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
        }

        if(!block){
            this->getOutput(index).transfer(EagleeyeRuntime(EAGLEEYE_CPU));
            return -1;
        }
        else{
            data = this->getOutput(index).cpu();
            shape = this->getOutput(index).dims().data();
            type = this->getOutput(index).type();
            return 0;
        }
    }

private:
    Base* m_op_func;
    std::thread m_op_thread;
    std::queue<std::vector<Tensor>> m_input_queue;
    std::queue<std::vector<Tensor>> m_output_queue;

    bool m_thread_status;

	std::mutex m_in_mu;
	std::condition_variable m_in_cond;
    bool m_run_finish;
};
}
}
#endif