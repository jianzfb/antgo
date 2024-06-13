#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <functional>
#include "eagleeye/common/EagleeyeModule.h"
#include "eagleeye/common/EagleeyeTime.h"
#include "${project}.grpc.pb.h"
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/health_check_service_interface.h>
#include "eagleeye/common/CJsonObject.hpp"


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
    ::grpc::Status ${servername}Message(::grpc::ServerContext* context, const ::${package}::${servername}MessageRequest* request, ::grpc::ServerWriter< ::${package}::${servername}MessageReply>* writer){
        std::string server_key = request->serverkey();

        bool is_pipeline_stop = false;
        while(1){
            std::string server_reply;
            eagleeye::ServerStatus result = eagleeye::eagleeye_pipeline_server_call(server_key, "", server_reply, 5);
            // TODO, 区分超时结束，还是管线停止结束
            if(result == eagleeye::SERVER_TIMEOUT){
                continue;
            }

            ${servername}MessageReply reply;
            reply.set_code(0);
            reply.set_data(server_reply);
            writer->Write(reply);
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