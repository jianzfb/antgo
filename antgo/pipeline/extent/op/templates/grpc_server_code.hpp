#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <functional>
#include "eagleeye/common/EagleeyeModule.h"
#include "eagleeye/common/EagleeyeTime.h"
#include "eagleeye/common/CJsonObject.hpp"
#include "eagleeye/common/base64.h"
#include "eagleeye/hardware/rk.h"
#include "${project}.grpc.pb.h"
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/health_check_service_interface.h>
#include "opencv2/opencv.hpp"

using namespace grpc;
using namespace ${package};


std::mutex g_create_or_stop_mu;
std::map<std::string, std::mutex> g_call_mu_map;
class ${servername}ServiceImpl final : public ${servername}::Service{
public:
    // sync interface
    ::grpc::Status ${servername}SyncStart(::grpc::ServerContext* context, const ::${package}::${servername}SyncStartRequest* request, ::${package}::${servername}SyncStartReply* response){
        // 从本地家在服务管线配置配置
        std::string server_pipeline = request->serverpipeline();
        std::string server_id = request->serverid();
        std::string server_config = request->servercfg();

        std::ifstream i_file_handle;
        i_file_handle.open("config/plugin_config.json");
        i_file_handle.seekg(0, std::ios::end);
        int file_buffer_size = i_file_handle.tellg();
        i_file_handle.seekg(0, std::ios::beg);
        char* file_buffer = new char[file_buffer_size+1];
        memset(file_buffer, '\0', sizeof(char)*(file_buffer_size+1));
        i_file_handle.read(file_buffer, file_buffer_size);
        std::string content = file_buffer;
        i_file_handle.close();
        delete[] file_buffer;
        neb::CJsonObject config_obj(content);
        neb::CJsonObject pipeline_config_obj;
        config_obj.Get(server_pipeline, pipeline_config_obj);
        if(pipeline_config_obj.IsEmpty()){
            EAGLEEYE_LOGE("No %s pipeline config", server_pipeline.c_str());
            return Status::CANCELLED;
        }
        pipeline_config_obj.Add("server_id", server_id);
        pipeline_config_obj.Add("server_timestamp", eagleeye::EagleeyeTime::getTimeStamp());

        // server_params，data_source
        if(server_config != ""){
            neb::CJsonObject server_config_obj(server_config);
            // 更新参数
            neb::CJsonObject exist_server_param_obj;
            pipeline_config_obj.Get("server_params", exist_server_param_obj);
            neb::CJsonObject request_server_params_obj;
            server_config_obj.Get("server_params", request_server_params_obj);
            if(request_server_params_obj.IsArray() && !request_server_params_obj.IsEmpty()){
                // TODO
            }

            // 更新数据源
            neb::CJsonObject request_data_source_obj;
            server_config_obj.Get("data_source", request_data_source_obj);
            if(request_data_source_obj.IsArray() && !request_data_source_obj.IsEmpty()){
                pipeline_config_obj.Delete("data_source");
                pipeline_config_obj.Add("data_source", request_server_params_obj);
            }
        }

        std::string server_pipeline_config = pipeline_config_obj.ToFormattedString();
        EAGLEEYE_LOGD("server pipeline config %s", server_pipeline_config.c_str());
        std::string server_key;
        std::unique_lock<std::mutex> locker(this->g_create_or_stop_mu);
        // 启动服务
        eagleeye::ServerStatus result = eagleeye::eagleeye_pipeline_server_start(server_pipeline_config, server_key, nullptr);

        if(result == eagleeye::SERVER_SUCCESS){
            response->set_code(0);
            response->set_serverkey(server_key);

            // 分配服务锁
            g_call_mu_map[server_key] = std::mutex();
            locker.unlock();
            return Status::OK;
        }

        locker.unlock();
        return Status::CANCELLED;
    }
    ::grpc::Status ${servername}SyncCall(::grpc::ServerContext* context, const ::${package}::${servername}SyncCallRequest* request, ::${package}::${servername}SyncCallReply* response){
        std::string server_key = request->serverkey();
        std::string server_request = request->serverrequest();
        if(g_call_mu_map.find(server_key) == g_call_mu_map.end()){
            EAGLEEYE_LOGD("No server key");
            return Status::CANCELLED
        }

        // step 1: 解析服务请求，并处理输入数据
        // image -> base64解码 -> opencv read -> memory
        // float -> memory
        // int32 -> memory
        std::vector<cv::Mat> mat_data_list;
        std::vector<std::shared_ptr<float>> float_data_list;
        std::vector<std::shared_ptr<int>> int_data_list;
        std::vector<eagleeye::RequestData> server_request_data;
        if(server_request != ""){
            neb::CJsonObject server_request_obj(server_request);
            neb::CJsonObject data_info;
            server_request_obj.Get("data", data_info);
            for(int data_i=0; data_i<data_info.GetArraySize(); ++data_i){
                neb::CJsonObject data_cfg;
                data_info.Get(data_i, data_cfg);
                std::string data_type = "";
                data_cfg.Get("type", data_type);

                if(data_type == "image"){
                    std::string data_content = "";
                    data_cfg.Get("content", data_content);

                    // step 1: base64 解码
                    std::string image_content = base64_decode(data_content);
                    // step 2: imread
                    char* data_c = const_cast<char*>(image_content.c_str());
                    std::vector<char> mem_buffer_enc_img(data_c, data_c + image_content.size());
                    cv::Mat image = cv::imdecode(mem_buffer_enc_img, cv::IMREAD_ANYCOLOR);
                    mat_data_list.push_back(image);

                    eagleeye::RequestData request_data;
                    request_data.data = image.data;
                    request_data.width = image.cols;
                    request_data.height = image.rows;
                    request_data.channel = 3;
                    request_data.type = "image";

                    server_request_data.push_back(request_data);
                }
                else if(data_type == "string"){
                    std::string data_content = "";
                    data_cfg.Get("content", data_content);

                    eagleeye::RequestData request_data;
                    request_data.data = (void*)(const_cast<char*>(data_content.c_str()));
                    request_data.width = data_content.size();
                    request_data.height = 1;
                    request_data.channel = 0;
                    request_data.type = "string";

                    server_request_data.push_back(request_data);                    
                }
                else if(data_type == "float"){
                    int width = 0;
                    int height = 0;
                    data_cfg.Get("width", width);
                    data_cfg.Get("height", height);

                    neb::CJsonObject array;
                    data_cfg.Get("content", array);
                    std::shared_ptr<float> float_share_ptr(new float[array.GetArraySize()], [](float* arr) { delete[] arr;});
                    float* float_ptr = float_share_ptr.get();
                    for(int index=0; index<array.GetArraySize(); ++index){
                        float value;
                        array.Get(index, value);
                        float_ptr[index] = value;
                    }
                    float_data_list.push_back(float_share_ptr);

                    eagleeye::RequestData request_data;
                    request_data.data = float_ptr;
                    request_data.width = width;
                    request_data.height = height;
                    request_data.channel = 0;
                    request_data.type = "float";

                    server_request_data.push_back(request_data);        
                }
                else if(data_type == "int32"){
                    int width = 0;
                    int height = 0;
                    data_cfg.Get("width", width);
                    data_cfg.Get("height", height);

                    neb::CJsonObject array;
                    data_cfg.Get("content", array);
                    std::shared_ptr<int> int_share_ptr(new int[array.GetArraySize()], [](int* arr) { delete[] arr;}); 
                    int* int_ptr = int_share_ptr.get();
                    for(int index=0; index<array.GetArraySize(); ++index){
                        int value;
                        array.Get(index, value);
                        int_ptr[index] = value;
                    }
                    int_data_list.push_back(int_share_ptr);

                    eagleeye::RequestData request_data;
                    request_data.data = int_ptr;
                    request_data.width = width;
                    request_data.height = height;
                    request_data.channel = 0;
                    request_data.type = "int32";

                    server_request_data.push_back(request_data); 
                }
            }
        }

        // step 2: 执行服务管线
        std::string server_reply;
        std::unique_lock<std::mutex> locker(this->g_call_mu_map[server_key]);
        eagleeye::eagleeye_pipeline_server_call(server_key, server_request_data, server_reply);
        locker.unlock();

        response->set_code(0);
        response->set_data(server_reply);
        return Status::OK;
    }
    ::grpc::Status ${servername}SyncStop(::grpc::ServerContext* context, const ::${package}::${servername}SyncStopRequest* request, ::${package}::${servername}SyncStopReply* response){
        std::string server_key = request->serverkey();
        if(g_call_mu_map.find(server_key) == g_call_mu_map.end()){
            EAGLEEYE_LOGD("No server key");
            return Status::CANCELLED
        }

        std::unique_lock<std::mutex> locker(this->g_create_or_stop_mu);
        // 停止服务
        eagleeye::eagleeye_pipeline_server_stop(server_key);

        // 删除服务锁
        g_call_mu_map.erase(server_key);
        locker.unlock();

        response->set_code(0);
        response->set_message("success");
        return Status::OK;
    }

    // async interface
    ::grpc::Status ${servername}AsynStart(::grpc::ServerContext* context, const ::${package}::${servername}AsynStartRequest* request, ::${package}::${servername}AsynStartReply* response){
        // 从本地家在服务管线配置配置
        std::string server_pipeline = request->serverpipeline();
        std::string server_id = request->serverid();
        std::string server_config = request->servercfg();

        std::ifstream i_file_handle;
        i_file_handle.open("config/plugin_config.json");
        i_file_handle.seekg(0, std::ios::end);
        int file_buffer_size = i_file_handle.tellg();
        i_file_handle.seekg(0, std::ios::beg);
        char* file_buffer = new char[file_buffer_size+1];
        memset(file_buffer, '\0', sizeof(char)*(file_buffer_size+1));
        i_file_handle.read(file_buffer, file_buffer_size);
        std::string content = file_buffer;
        i_file_handle.close();
        delete[] file_buffer;
        neb::CJsonObject config_obj(content);
        neb::CJsonObject pipeline_config_obj;
        config_obj.Get(server_pipeline, pipeline_config_obj);
        if(pipeline_config_obj.IsEmpty()){
            EAGLEEYE_LOGE("No %s pipeline config", server_pipeline.c_str());
            return Status::CANCELLED;
        }
        pipeline_config_obj.Add("server_id", server_id);
        pipeline_config_obj.Add("server_timestamp", eagleeye::EagleeyeTime::getTimeStamp());

        // server_params，data_source
        if(server_config != ""){
            neb::CJsonObject server_config_obj(server_config);
            // 更新参数
            neb::CJsonObject exist_server_param_obj;
            pipeline_config_obj.Get("server_params", exist_server_param_obj);
            neb::CJsonObject request_server_params_obj;
            server_config_obj.Get("server_params", request_server_params_obj);
            if(request_server_params_obj.IsArray() && !request_server_params_obj.IsEmpty()){
                // TODO
            }

            // 更新数据源
            neb::CJsonObject request_data_source_obj;
            server_config_obj.Get("data_source", request_data_source_obj);
            if(request_data_source_obj.IsArray() && !request_data_source_obj.IsEmpty()){
                pipeline_config_obj.Delete("data_source");
                pipeline_config_obj.Add("data_source", request_server_params_obj);
            }
        }

        std::string server_pipeline_config = pipeline_config_obj.ToFormattedString();
        EAGLEEYE_LOGD("server pipeline config %s", server_pipeline_config.c_str());
        std::string server_key;
        std::unique_lock<std::mutex> locker(this->g_create_or_stop_mu);
        // 启动服务
        eagleeye::ServerStatus result = eagleeye::eagleeye_pipeline_server_start(server_pipeline_config, server_key, nullptr);
        locker.unlock();

        if(result == eagleeye::SERVER_SUCCESS){
            response->set_code(0);
            response->set_serverkey(server_key);
            return Status::OK;
        }
        return Status::CANCELLED;
    }
    ::grpc::Status ${servername}AsynData(::grpc::ServerContext* context, const ::${package}::${servername}AsynDataRequest* request, ::${package}::${servername}AsynDataReply* response){
        // 适合于复杂的请求数据
        std::string server_key = request->serverkey();
        std::string server_request = request->serverrequest();

        // step 1: 解析服务请求，并处理输入数据
        // image -> base64解码 -> opencv read -> memory
        // float -> memory
        // int32 -> memory
        std::vector<cv::Mat> mat_data_list;
        std::vector<std::shared_ptr<float>> float_data_list;
        std::vector<std::shared_ptr<int>> int_data_list;
        std::vector<eagleeye::RequestData> server_request_data;
        if(server_request != ""){
            neb::CJsonObject server_request_obj(server_request);
            neb::CJsonObject data_info;
            server_request_obj.Get("data", data_info);
            for(int data_i=0; data_i<data_info.GetArraySize(); ++data_i){
                neb::CJsonObject data_cfg;
                data_info.Get(data_i, data_cfg);
                std::string data_type = "";
                data_cfg.Get("type", data_type);

                if(data_type == "image"){
                    std::string data_content = "";
                    data_cfg.Get("content", data_content);

                    // step 1: base64 解码
                    std::string image_content = base64_decode(data_content);
                    // step 2: imread
                    char* data_c = const_cast<char*>(image_content.c_str());
                    std::vector<char> mem_buffer_enc_img(data_c, data_c + image_content.size());
                    cv::Mat image = cv::imdecode(mem_buffer_enc_img, cv::IMREAD_ANYCOLOR);
                    mat_data_list.push_back(image);

                    eagleeye::RequestData request_data;
                    request_data.data = image.data;
                    request_data.width = image.cols;
                    request_data.height = image.rows;
                    request_data.channel = 3;
                    request_data.type = "image";

                    server_request_data.push_back(request_data);
                }
                else if(data_type == "string"){
                    std::string data_content = "";
                    data_cfg.Get("content", data_content);

                    eagleeye::RequestData request_data;
                    request_data.data = (void*)(const_cast<char*>(data_content.c_str()));
                    request_data.width = data_content.size();
                    request_data.height = 1;
                    request_data.channel = 0;
                    request_data.type = "string";

                    server_request_data.push_back(request_data);                    
                }
                else if(data_type == "float"){
                    int width = 0;
                    int height = 0;
                    data_cfg.Get("width", width);
                    data_cfg.Get("height", height);

                    neb::CJsonObject array;
                    data_cfg.Get("content", array);
                    std::shared_ptr<float> float_share_ptr(new float[array.GetArraySize()], [](float* arr) { delete[] arr;});
                    float* float_ptr = float_share_ptr.get();
                    for(int index=0; index<array.GetArraySize(); ++index){
                        float value;
                        array.Get(index, value);
                        float_ptr[index] = value;
                    }
                    float_data_list.push_back(float_share_ptr);

                    eagleeye::RequestData request_data;
                    request_data.data = float_ptr;
                    request_data.width = width;
                    request_data.height = height;
                    request_data.channel = 0;
                    request_data.type = "float";
                    server_request_data.push_back(request_data);        
                }
                else if(data_type == "int32"){
                    int width = 0;
                    int height = 0;
                    data_cfg.Get("width", width);
                    data_cfg.Get("height", height);

                    neb::CJsonObject array;
                    data_cfg.Get("content", array);
                    std::shared_ptr<int> int_share_ptr(new int[array.GetArraySize()], [](int* arr) { delete[] arr;}); 
                    int* int_ptr = int_share_ptr.get();
                    for(int index=0; index<array.GetArraySize(); ++index){
                        int value;
                        array.Get(index, value);
                        int_ptr[index] = value;
                    }
                    int_data_list.push_back(int_share_ptr);

                    eagleeye::RequestData request_data;
                    request_data.data = int_ptr;
                    request_data.width = width;
                    request_data.height = height;
                    request_data.channel = 0;
                    request_data.type = "int32";
                    server_request_data.push_back(request_data); 
                }
            }
        }

        // 推送数据到管线队列
        eagleeye::eagleeye_pipeline_server_push(server_key, server_request_data);

        response->set_code(0);
        return Status::OK;
    }
    ::grpc::Status ${servername}AsynPacket(::grpc::ServerContext* context, const ::${package}::${servername}AsynPacketRequest* request, ::${package}::${servername}AsynPacketReply* response){
        // 仅用于对视频流分析
        std::string server_key = request->serverkey();

        // 推送数据到管线队列
        char* package_data = const_cast<char*>(request->package_data().c_str());
        eagleeye::eagleeye_pipeline_server_stream(server_key, (uint8_t*)package_data, request->package_size());
    
        response->set_code(0);
        return Status::OK;
    }
    ::grpc::Status ${servername}AsynMessage(::grpc::ServerContext* context, const ::${package}::${servername}AsynMessageRequest* request, ::grpc::ServerWriter< ::${package}::${servername}AsynMessageReply>* writer){
        std::string server_key = request->serverkey();
        std::string server_request = request->serverrequest();
        neb::CJsonObject server_request_obj(server_request);
        int timeout = -1;
        if(!(server_request_obj.Get("timeout", timeout))){
            // 默认 -1
            timeout = -1;
        }

        while(1){
            std::string server_reply;
            eagleeye::ServerStatus result = eagleeye::eagleeye_pipeline_server_call(server_key, std::vector<eagleeye::RequestData>{}, server_reply, timeout);

            ${servername}AsynMessageReply reply;
            if(result == eagleeye::SERVER_NOT_EXIST){
                reply.set_code(1);
                writer->Write(reply);
                return Status::OK;
            }

            reply.set_code(0);
            reply.set_data(server_reply);
            writer->Write(reply);
            if(result == eagleeye::SERVER_TIMEOUT){
                reply.set_code(1);
                break;
            }
        }

        return Status::OK;
    }
    ::grpc::Status ${servername}AsynStop(::grpc::ServerContext* context, const ::${package}::${servername}AsynStopRequest* request, ::${package}::${servername}AsynStopReply* response){
        std::string server_key = request->serverkey();
        if(g_call_mu_map.find(server_key) == g_call_mu_map.end()){
            EAGLEEYE_LOGD("No server key");
            return Status::CANCELLED
        }

        std::unique_lock<std::mutex> locker(this->g_create_or_stop_mu);
        // 停止服务
        eagleeye::eagleeye_pipeline_server_stop(server_key);
        locker.unlock();

        response->set_code(0);
        response->set_message("success");
        return Status::OK;
    }
};
