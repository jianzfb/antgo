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
#include "${project}.grpc.pb.h"
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/health_check_service_interface.h>
#include "opencv2/opencv.hpp"

using namespace grpc;
using namespace ${package};
class ${servername}ServiceImpl final : public ${servername}::Service{
public:
    ::grpc::Status ${servername}Start(::grpc::ServerContext* context, const ::${package}::${servername}StartRequest* request, ::${package}::${servername}StartReply* response){
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
        std::cout<<"server_pipeline_config"<<std::endl;
        std::cout<<server_pipeline_config<<std::endl;
        std::string server_key;
        eagleeye::ServerStatus result = eagleeye::eagleeye_pipeline_server_start(server_pipeline_config, server_key, nullptr);

        if(result == eagleeye::SERVER_SUCCESS){
            response->set_code(0);
            response->set_serverkey(server_key);
            return Status::OK;
        }
        return Status::CANCELLED;
    }
    ::grpc::Status ${servername}Push(::grpc::ServerContext* context, const ::${package}::${servername}PushRequest* request, ::${package}::${servername}PushReply* response){
        std::string server_key = request->serverkey();
        std::string server_request = request->serverrequest();

        // step 1: 解析服务请求，并处理输入数据
        // image -> base64解码 -> opencv read -> memory
        // matrix/float -> memory
        // matrix/int32 -> memory
        std::vector<cv::Mat> data_list;
        neb::CJsonObject recon_data_list;
        if(server_request != ""){
            neb::CJsonObject server_request_obj(server_request);
            neb::CJsonObject data_info;
            server_request_obj.Get("data", data_info);
            for(int data_i=0; data_i<data_info.GetArraySize(); ++data_i){
                neb::CJsonObject data_cfg;
                data_info.Get(data_i, data_cfg);
                std::string data_type = "";
                data_cfg.Get("type", data_type);
                std::string data_content = "";
                data_cfg.Get("content", data_content);

                if(data_type == "image"){
                    // step 1: base64 解码
                    std::string image_content = base64_decode(data_content);
                    // step 2: imread
                    char* data_c = const_cast<char*>(image_content.c_str());
                    std::vector<char> mem_buffer_enc_img(data_c, data_c + image_content.size());
                    cv::Mat image = cv::imdecode(mem_buffer_enc_img, cv::IMREAD_ANYCOLOR);
                    data_list.push_back(image);

                    neb::CJsonObject info_obj;
                    info_obj.Add("type", "image");
                    info_obj.Add("content", long(image.data));
                    info_obj.Add("width", image.cols);
                    info_obj.Add("height", image.rows);
                    info_obj.Add("channel", 3);

                    recon_data_list.Add(info_obj);
                }
                else if(data_type == "string"){
                    // 
                    neb::CJsonObject info_obj;
                    info_obj.Add("type", "string");
                    info_obj.Add("content", data_content);

                    recon_data_list.Add(info_obj);
                }
                else if(data_type == "matrix/float"){
                    // TODO, 即将支持
                }
                else if(data_type == "matrix/int32"){
                    // TODO, 即将支持
                }
            }
        }

        neb::CJsonObject recon_server_request_obj;
        recon_server_request_obj.Add("data", recon_data_list);

        // step 2: 执行服务管线
        eagleeye::eagleeye_pipeline_server_push(server_key, recon_server_request_obj.ToString());
        response->set_code(0);
        return Status::OK;
    }
    ::grpc::Status ${servername}Message(::grpc::ServerContext* context, const ::${package}::${servername}MessageRequest* request, ::grpc::ServerWriter< ::${package}::${servername}MessageReply>* writer){
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
            eagleeye::ServerStatus result = eagleeye::eagleeye_pipeline_server_call(server_key, "", server_reply, timeout);

            ${servername}MessageReply reply;
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
    ::grpc::Status ${servername}Stop(::grpc::ServerContext* context, const ::${package}::${servername}StopRequest* request, ::${package}::${servername}StopReply* response){
        std::string server_key = request->serverkey();
        eagleeye::eagleeye_pipeline_server_stop(server_key);
        response->set_code(0);
        response->set_message("success");
        return Status::OK;
    }
};
