/*集成核心依赖*/
#include "${project}_plugin.h"
/*集成可选依赖，仅负责数据加载等外围功能*/
#include "eagleeye/common/EagleeyeStr.h"
#include "eagleeye/common/EagleeyeLog.h"
#include "eagleeye/common/EagleeyeFile.h"
#include "eagleeye/common/EagleeyeTime.h"
/*C++系统依赖*/
#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>
#include <memory>
#include <string>
#include <cstring>
#include <sstream>
#include <thread>
#include <stdio.h> 
#include <unistd.h> 
#include <fcntl.h> 

using namespace eagleeye;

int get_data_size(int data_type, std::vector<size_t> shape){
    int num = std::accumulate(shape.begin(), shape.end(), 1, [](int64_t a, int64_t b){return a*b;});
    if(data_type == 0 || data_type == 1 || data_type == 8 || data_type == 9){
        // char,uchar
        return sizeof(char) * num;
    }
    else if(data_type == 2 || data_type == 3){
        // short,ushort
        return sizeof(short) * num;
    }
    else if(data_type == 4 || data_type == 5){
        // int32,uint32
        return sizeof(int) * num;
    }
    else if(data_type == 6){
        // float32
        return sizeof(float) * num;
    }
    else if(data_type == 7){
        // double
        return sizeof(double) * num;
    }
    else if(data_type == 10){
        // bool
        return sizeof(bool) * num;
    }
    else if(data_type == 11){
        // std::string
        return sizeof(char) * num;
    }

    return 0;
}

int main(int argc, char** argv){
    int original_stdout = 0;
    // 运行模式 image, file, stdin
    std::string run_mode(argv[1]);
    if(!(run_mode == "file" || run_mode == "image")){
        original_stdout = dup(STDOUT_FILENO);
        close(STDOUT_FILENO);
    }

    // 1.step initialize ${project} module
    const char* config_folder = NULL;   // ${project} module configure folder
    eagleeye_${project}_initialize(config_folder);
    bool is_ok = false;

    // 2.step (optional) set module pipeline parameter
    /*
    char* node_name = "";     // NODE NAME in pipeline
    char* param_name = "";    // PARAMETER NAME of NODE in pipeline
    void* value = NULL;       // PARAMETER VALUE
    is_ok = eagleeye_${project}_set_param(node_name, param_name, value);
    if(is_ok){
        EAGLEEYE_LOGD("success to set parameter %s of node %s",param_name,node_name);
    }
    else{
        EAGLEEYE_LOGE("fail to set parameter %s of node %s",param_name,node_name);
    }
    */

    // 3.step set input data info
    std::vector<std::string> input_name_list = ${input_name_list};           // 输入节点名字（见计算管线搭建中的设置）
    std::vector<std::vector<size_t>> input_size_list = ${input_size_list};   // 输入节点数据形状
    std::vector<int> input_type_list = ${input_type_list};                   // 输入节点属性类型（见EagleeyeType类型）
    int input_port_num = input_name_list.size();

    std::vector<void*> cache_data_list;
    if(!(run_mode == "file" || run_mode == "image")){
        // stdinout mode
        for(int input_port=0; input_port<input_port_num; ++input_port){
            std::string input_name = input_name_list[input_port];
            int input_type = input_type_list[input_port];

            // std::vector<size_t> input_size = input_size_list[input_port];
            int argi = input_port + 2;
            std::string input_port_config_info = argv[argi];
            std::string config_sep = "/";
            std::vector<std::string> input_port_config = split(input_port_config_info, config_sep);
            std::string shape_info = input_port_config[1];
            std::string shape_sep = ",";
            std::vector<std::string> shape_keys = split(shape_info, shape_sep);
            std::vector<size_t> input_size;
            for(int shape_i=0; shape_i<shape_keys.size(); ++shape_i){
                input_size.push_back(tof<size_t>(shape_keys[shape_i]));
            }
            input_size_list[input_port] = input_size;

            int data_byte_size = get_data_size(input_type, input_size);
            if(cache_data_list.size() < input_port+1){
                char* data = (char*)malloc(data_byte_size);  // YOUR DATA POINTER;
                memset(data, 0, data_byte_size);
                // save to list
                cache_data_list.push_back(data);
            }
        }
    }

    while(true){
        // 3.step load pipeline input
        if(run_mode == "file"){
            // 从文件加载
            // load data from ./data/input
            // file format input_name.input_port.type.shape.bin
            const char* input_file_folder = "./data/input";
            std::vector<std::string> input_file_list;
            traverseFiles(input_file_folder, input_file_list);
            std::vector<std::string> order_input_shape_list;
            order_input_shape_list.resize(input_port_num);

            std::string file_name_sep = ".";
            for(int i=0; i<input_file_list.size(); ++i){
                std::vector<std::string> keys = split(input_file_list[i], file_name_sep);
                int input_port = atoi(keys[1].c_str());
                order_input_shape_list[input_port] = keys[3];
            }

            for(int input_port=0; input_port<input_port_num; ++input_port){
                std::string input_name = input_name_list[input_port];
                int input_type = input_type_list[input_port];
                std::string shape_sep = ",";
                std::vector<std::string> shape_str = split(order_input_shape_list[input_port], shape_sep);
                std::vector<size_t> input_size;
                for(int shape_i=0; shape_i<shape_str.size(); ++shape_i){
                    input_size.push_back(tof<size_t>(shape_str[shape_i]));
                }
                int data_byte_size = get_data_size(input_type, input_size);
                if(cache_data_list.size() < input_port+1){
                    char* data = (char*)malloc(data_byte_size);  // YOUR DATA POINTER;
                    memset(data, 0, data_byte_size);
                    // save to list
                    cache_data_list.push_back(data);
                }
                char* data = (char*)cache_data_list[input_port];

                std::string input_size_str = "";
                for(int k=0; k<input_size.size(); ++k){
                    if(k != input_size.size()-1){
                        input_size_str += tos(input_size[k]) + ",";
                    }
                    else{
                        input_size_str += tos(input_size[k]);
                    }
                }
                std::string file_path = std::string("./data/input/")+input_name+"."+ tos(input_port)+"."+tos(input_type)+"."+input_size_str+".bin";
                EAGLEEYE_LOGI("load file %s for port %d", file_path.c_str(), input_port);
                EAGLEEYE_LOGI("tensor shape %s", input_size_str.c_str());

                std::ifstream file_path_handle;
                file_path_handle.open(file_path.c_str(),std::ios::binary);
                file_path_handle.read((char*)(data), data_byte_size);
                file_path_handle.close();

                // set pipeline input
                eagleeye_${project}_set_input(input_name.c_str(), data, input_size.data(), input_size.size(), 0, input_type);
            }
        }
        else if(run_mode == "image"){
            // 从图像加载
            // load data argv
            for(int input_port=0; input_port<input_port_num; ++input_port){
                std::string input_name = input_name_list[input_port];
                int input_type = input_type_list[input_port];

                int argi = input_port + 2;
                EAGLEEYE_LOGI("load image %s for port %d", argv[argi], input_port);
                unsigned char* data = NULL;
                int height = 0;
                int width = 0;
                int channel = 0;

                loadpng(argv[argi], data, height, width, channel);
                EAGLEEYE_LOGI("image shape %d, %d, %d", height, width, channel);
                std::vector<size_t> input_size={size_t(height), size_t(width), size_t(channel)};

                // set pipeline input
                eagleeye_${project}_set_input(input_name.c_str(), data, input_size.data(), input_size.size(), 0, input_type);
                free(data);
            }
        }
        else{
            // 从stdin加载
            for(int input_port=0; input_port<input_port_num; ++input_port){
                std::string input_name = input_name_list[input_port];
                int input_type = input_type_list[input_port];
                std::vector<size_t> input_size = input_size_list[input_port];
                int data_byte_size = get_data_size(input_type, input_size);
                char* data = (char*)cache_data_list[input_port];
                fread(data, sizeof(char), data_byte_size, stdin);
                
                // set pipeline input
                eagleeye_${project}_set_input(input_name.c_str(), data, input_size.data(), input_size.size(), 0, input_type);
            }
        }

        // 4.step refresh module pipeline
        eagleeye_${project}_run();

        // 5.step get output data of ${project} module
        if(run_mode == "file"){
            if(isdirexist("./data/output/")){
                deletedir("./data/output");
            }
        }

        std::vector<std::string> output_name_list = ${output_name_list};             // 输出节点名字（见计算管线搭建中的设置）
        std::vector<int> output_port_list = ${output_port_list};                     // 输出节点端口（见计算管线搭建中的设置）
        for(int i=0; i<output_name_list.size(); ++i){
            std::string output_name = output_name_list[i];
            std::string output_port = tos(output_port_list[i]);
            std::string output_and_port_info = output_name + "/" + output_port;

            void* out_data;         // RESULT DATA POINTER
            size_t* out_data_size;  // RESULT DATA SHAPE (IMAGE HEIGHT, IMAGE WIDTH, IMAGE CHANNEL)
            int out_data_dims=0;    // 3
            int out_data_type=0;    // RESULT DATA TYPE 
            is_ok = eagleeye_${project}_get_output(output_and_port_info.c_str(),out_data, out_data_size, out_data_dims, out_data_type);   
            if(is_ok){
                int data_byte_size = get_data_size(out_data_type, std::vector<size_t>(out_data_size, out_data_size+out_data_dims));
                // file format output_name.output_port.type.size.bin
                std::string output_size_str = "";
                for(int k=0; k<out_data_dims; ++k){
                    if(k != out_data_dims-1){
                        output_size_str += tos(out_data_size[k]) + ",";
                    }
                    else{
                        output_size_str += tos(out_data_size[k]);
                    }
                }

                std::string out_data_id = output_name+"."+output_port+"."+tos(out_data_type)+"."+output_size_str+".bin";               
                if(run_mode == "file" || run_mode == "image"){
                    // save data to ./data/output/
                    std::string file_path = std::string("./data/output/")+out_data_id;
                    if(!isdirexist("./data/output")){
                        createdirectory("./data/output");
                    }
                    if((run_mode == "image") && (out_data_dims == 3 && out_data_type == 0 && (out_data_size[2] == 3 || out_data_size[2] == 4))){
                        // only for image data format
                        file_path = std::string("./data/output/")+output_name+"."+output_port+".png";
                        unsigned char* image_data = (unsigned char*)out_data;
                        savepng(file_path.c_str(), image_data, out_data_size[0], out_data_size[1], out_data_size[1], out_data_size[2]);
                    }
                    else{
                        // for any tensor
                        std::ofstream file_path_handle;
                        file_path_handle.open(file_path.c_str(),std::ios::binary);
                        file_path_handle.write((char*)out_data, data_byte_size);
                        file_path_handle.close();
                    }
                }
                else{
                    // 恢复stdout
                    fflush(stdout);
                    dup2(original_stdout, STDOUT_FILENO);

                    char* temp_ptr = (char*)(malloc(sizeof(size_t)*(2+out_data_dims) + data_byte_size));
                    size_t t = out_data_type;
                    int offset = 0;
                    memcpy(temp_ptr+offset, &t, sizeof(size_t));
                    t = out_data_dims;
                    offset += sizeof(size_t);
                    memcpy(temp_ptr+offset, &t, sizeof(size_t));
                    offset += sizeof(size_t);
                    memcpy(temp_ptr+offset, out_data_size, sizeof(size_t)*out_data_dims);
                    offset += sizeof(size_t)*out_data_dims;
                    if(data_byte_size > 0){
                        memcpy(temp_ptr+offset, out_data, data_byte_size);
                    }

                    setbuf(stdout, NULL);
                    fwrite(temp_ptr, sizeof(char), sizeof(size_t)*(2+out_data_dims) + data_byte_size, stdout);
                    free(temp_ptr);

                    // 关闭stdout
                    close(STDOUT_FILENO);
                }
            }
        }

        // 6.step (optional) sometimes, could call this function to reset all intermedianl state in pipeline
        /*
        is_ok = eagleeye_${project}_reset();
        if(is_ok){
            EAGLEEYE_LOGD("success to reset pipeline");
        }
        else{
            EAGLEEYE_LOGE("fail to reset pipeline");
        }
        */

        if(run_mode == "file" || run_mode == "image"){
            break;
        }
    }

    // 7.step release ${project} module
    for(int i=0; i<cache_data_list.size(); ++i){
        free(cache_data_list[i]);
    }    
    eagleeye_${project}_release();

    return 0;
}