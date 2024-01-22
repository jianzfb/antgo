#ifndef _EAGLEEYE_${op_name}_OP_
#define _EAGLEEYE_${op_name}_OP_
#include "eagleeye/engine/nano/dataflow/base.h"
#include "eagleeye/basic/Tensor.h"
#include "eagleeye/common/EagleeyeFile.h"
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
        m_file_prefix = "${file_prefix}";
        m_check_empty_at_index = ${check_empty_at_index};
        m_writable_path = "./";
    }
    virtual ~${op_name}(){
        delete m_func;
    }

    virtual int init(std::map<std::string, std::vector<float>> params){return 0;}
    virtual int init(std::map<std::string, std::vector<std::vector<float>>> params){return 0;};
    virtual int init(std::map<std::string, std::vector<std::string>> params){
        m_func->init(params);

        if(params.find("writable_path") != params.end()){
            if(endswith(params["writable_path"][0], "/")){
                this->m_writable_path = params["writable_path"][0];
            }
            else{
                this->m_writable_path = params["writable_path"][0] + "/";
            }
            EAGLEEYE_LOGD("Set Cache Writable folder %s", this->m_writable_path.c_str());
        }     
        return 0;
    };

    virtual int runOnCpu(const std::vector<Tensor>& input){
        Tensor key = input[0];
        int* key_ptr = key.cpu<int>();
        int key_name = key_ptr[0];

        std::string cache_file_path = m_writable_path+m_file_prefix+"_"+std::to_string(key_name)+".bin";
        if(m_cache_map.find(key_name) == m_cache_map.end() && isfileexist(cache_file_path.c_str())){
            // 从文件中读取Tensor
            // 读取 tensor个数
            // 读取每个tensor的信息，形状，类型，数据
            EagleeyeIO yard_io;
            yard_io.createReadHandle(cache_file_path, READ_BINARY_MODE);

            int64_t tensor_num_array[1];
            void* tensor_num_array_ptr = (void*)tensor_num_array;
            int tensor_num_size = 0;
            yard_io.read(tensor_num_array_ptr, tensor_num_size);
            std::vector<Tensor> tensor_list(tensor_num_array[0]);
            for(int64_t tensor_i=0; tensor_i<tensor_num_array[0]; ++tensor_i){
                // 形状
                int64_t dim_num_array[1];
                void* dim_num_array_ptr = (void*)dim_num_array;
                int dim_num_size = 0;
                yard_io.read(dim_num_array_ptr, dim_num_size);

                // 
                std::vector<int64_t> tensor_shape(dim_num_array[0]);
                void* tensor_shape_ptr = (void*)(tensor_shape.data());
                int tensor_shape_size = 0;
                yard_io.read(tensor_shape_ptr, tensor_shape_size);

                // 类型
                int64_t tensor_type_array[1];
                void* tensor_type_array_ptr = (void*)tensor_type_array;
                int tensor_type_size = 0;
                yard_io.read(tensor_type_array_ptr, tensor_type_size);

                // 数据
                tensor_list[tensor_i] = Tensor(
                    tensor_shape,
                    EagleeyeType(tensor_type_array[0]),
                    DataFormat::AUTO,
                    CPU_BUFFER
                );

                int tensor_data_size = 0;
                void* tensor_ptr = tensor_list[tensor_i].cpu();
                yard_io.read(tensor_ptr, tensor_data_size);
            }
            yard_io.destroyHandle();
            m_cache_map[key_name] = tensor_list;
        }

        std::vector<Tensor> output_list;
        if(m_cache_map.find(key_name) == m_cache_map.end()){
            // 运行
            std::vector<Tensor> func_input;
            for(int input_i=1; input_i<${input_num}; ++input_i){
                func_input.push_back(input[input_i]);
            }

            this->m_func->runOnCpu(func_input);

            // 检查输出
            bool is_ok = true;
            for(int output_i=0; output_i<${output_num}; ++output_i){
                output_list.push_back(this->m_func->getOutput(output_i));
            }
            if(this->m_func->getOutput(m_check_empty_at_index).dims()[0] == 0){
                is_ok = false;
            }

            if(is_ok){
                m_cache_map[key_name] = output_list;

                // 缓存成文件
                EagleeyeIO yard_io;
                yard_io.createWriteHandle(cache_file_path, false, WRITE_BINARY_MODE);

                int64_t tensor_num = output_list.size();
                // 数量
                yard_io.write(&tensor_num, sizeof(int64_t));
                for(int64_t tensor_i=0; tensor_i<tensor_num; ++tensor_i){
                    Tensor tt = output_list[tensor_i];

                    // 形状
                    int64_t dim_num = tt.dims().size();
                    yard_io.write(&dim_num, sizeof(int64_t));

                    std::vector<int64_t> tensor_shape = tt.dims().data();
                    yard_io.write(tensor_shape.data(), tensor_shape.size()*sizeof(int64_t));

                    // 类型
                    int64_t tensor_type = (int64_t)(tt.type());
                    yard_io.write(&tensor_type, sizeof(int64_t));

                    // 数据
                    yard_io.write(tt.cpu(), tt.blobsize());
                }
                yard_io.destroyHandle();
            }
        }

        if(m_cache_map.find(key_name) == m_cache_map.end()){
            // 没有运行成功，返回空结果
            for(int output_i=0; output_i<${output_num}; ++output_i){
                this->m_outputs[output_i] = output_list[output_i];
            }
            return 0;
        }

        // 运行成功，直接从cache里获得返回结果
        for(int output_i=0; output_i<${output_num}; ++output_i){
            this->m_outputs[output_i] = m_cache_map[key_name][output_i];
        }
        return 0;
    }
    virtual int runOnGpu(const std::vector<Tensor>& input){
        return 0;
    }

private:
    Base* m_func;
    std::map<int, std::vector<Tensor>> m_cache_map;
    std::string m_file_prefix;                      // 缓存文件名前缀
    std::string m_writable_path;                    // 缓存文件夹
    int m_check_empty_at_index;
};
}
}

#endif