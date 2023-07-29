#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "eagleeye/common/EagleeyeLog.h"
#include "eagleeye/common/EagleeyeStr.h"
#include "eagleeye/framework/pipeline/DynamicPipelineCreater.h"


namespace py = pybind11;
using namespace py::literals;

namespace eagleeye{
std::map<std::string, std::shared_ptr<AnyPipeline>> pipeline_pools;

py::list pipeline_execute(py::str exe_name, py::str pipeline_name, py::dict param, py::list input_tensors){
    py::list output_tensors;

    // 创建算子/加载算子
    bool is_new_created = false;
    if(pipeline_pools.find(py::cast<std::string>(pipeline_name)) == pipeline_pools.end()){
        AnyPipeline* op = CreatePipeline<>(py::cast<std::string>(pipeline_name));
        if(op == NULL){
            return output_tensors;
        }

        is_new_created = true;
        pipeline_pools[pipeline_name] = std::shared_ptr<AnyPipeline>(op);
    }
    std::shared_ptr<AnyPipeline> exe_pipeline = pipeline_pools[pipeline_name];
    if(is_new_created){
        // TODO, 基于传入的参数，初始化
    }

    // 转换py到Pipeline输入
    for(int input_index=0; input_index < input_tensors.size(); ++input_index){
        auto array = pybind11::array::ensure(input_tensors[input_index]);
		if (!array)
			return output_tensors;

		if (array.dtype() == pybind11::dtype::of<int32_t>()){
            py::buffer_info buf = array.request();

            std::vector<size_t> shape;
            for(int i=0; i<array.ndim(); ++i){
                shape.push_back(buf.shape[i]);
            }

            std::string input_name = "placeholder_" + tos(input_index);
            exe_pipeline->setInput(input_name.c_str(), buf.ptr, shape.data(), shape.size(), 0, 4);
		}
        else if(array.dtype() == pybind11::dtype::of<float>()){
            py::buffer_info buf = array.request();

            std::vector<size_t> shape;
            for(int i=0; i<array.ndim(); ++i){
                shape.push_back(buf.shape[i]);
            }            

            std::string input_name = "placeholder_" + tos(input_index);
            exe_pipeline->setInput(input_name.c_str(), buf.ptr, shape.data(), shape.size(), 0, 6);
        }
        else if(array.dtype() == pybind11::dtype::of<double>()){
            py::buffer_info buf = array.request();

            std::vector<size_t> shape;
            for(int i=0; i<array.ndim(); ++i){
                shape.push_back(buf.shape[i]);
            }            

            std::string input_name = "placeholder_" + tos(input_index);
            exe_pipeline->setInput(input_name.c_str(), buf.ptr, shape.data(), shape.size(), 0, 7);            
        }
        else if(array.dtype() == pybind11::dtype::of<unsigned char>()){
            py::buffer_info buf = array.request();

            std::vector<size_t> shape;
            for(int i=0; i<array.ndim(); ++i){
                shape.push_back(buf.shape[i]);
            }  
              
            std::string input_name = "placeholder_" + tos(input_index);
            exe_pipeline->setInput(input_name.c_str(), buf.ptr, shape.data(), shape.size(), 0, 1);    
        }
		else {
            EAGLEEYE_LOGE("Dont support input type. Only Support int32/float/double/unsigned char");
			return output_tensors;
		}
    }

    // run
    exe_pipeline->start(NULL, NULL);

    // 转换Pipeline输出到py
    std::vector<std::string> output_nodes;
    std::vector<std::string> output_types;
    std::vector<std::string> output_targets;
    exe_pipeline->getPipelineOutputs(output_nodes, output_types, output_targets);

    std::vector<std::string> output_name_list = {};                 // 输出节点名字（见计算管线搭建中的设置）
    for(int output_i=0; output_i<output_nodes.size(); ++output_i){
        std::string output_and_port_info = output_nodes[output_i] + "/0";

        void* out_data;         // RESULT DATA POINTER
        size_t* out_data_size;  // RESULT DATA SHAPE (IMAGE HEIGHT, IMAGE WIDTH, IMAGE CHANNEL)
        int out_data_dims=0;    // 3
        int out_data_type=0;    // RESULT DATA TYPE 

        exe_pipeline->getOutput(output_and_port_info.c_str(), out_data, out_data_size, out_data_dims, out_data_type);
        if(out_data != NULL){
            std::vector<py::ssize_t> shape;
            for(int i=0; i<out_data_dims; ++i){
                shape.push_back(out_data_size[i]);
            }

            if(out_data_type == 4){
                // int
                auto result = py::array_t<int32_t>(py::detail::any_container<ssize_t>(shape), (int32_t*)out_data);
                output_tensors.append(result);
            }
            else if(out_data_type == 6){
                // float
                auto result = py::array_t<float>(py::detail::any_container<ssize_t>(shape), (float*)out_data);
                output_tensors.append(result);
            }
            else if(out_data_type == 7){
                // double
                auto result = py::array_t<double>(py::detail::any_container<ssize_t>(shape), (double*)out_data);
                output_tensors.append(result);
            }
            else if(out_data_type == 1 || out_data_type == 8 || out_data_type == 9){
                // unsigned char
                auto result = py::array_t<unsigned char>(py::detail::any_container<ssize_t>(shape), (unsigned char*)out_data);
                output_tensors.append(result);
            }
            else{
                //
                EAGLEEYE_LOGE("Out data type abnormal (type=%d).", out_data_type);
            }
        }
        else{
            EAGLEEYE_LOGE("Out data abnormal.");
        }    
    }
    return output_tensors;
}


PYBIND11_MODULE(${project}, m) {
    m.doc() = "pybind11 pipline/node ext"; // optional module docstring
    m.def("pipeline_execute", &pipeline_execute);
}
}
