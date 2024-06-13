#include <iostream>
#include <fstream>
#include "eagleeye/common/EagleeyeStr.h"
#include "eagleeye/common/EagleeyeLog.h"
#include "grpc_server.hpp"
using namespace eagleeye;

int main(int argc, char** argv){
    std::stringstream ss(argv[1]);
    int port = 0;
	ss>>port;

    // 初始化插件系统
    eagleeye::eagleeye_pipeline_server_init("${plugin_root}");

    // 初始化服务
    std::string server_address = "0.0.0.0:"+std::to_string(port);
    std::cout<<"Server listening on " << server_address << std::endl;

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;

    ${servername}ServiceImpl service;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::MAX_POLLERS, 10);
    builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIME_MS, 30000);
    builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
    builder.AddChannelArgument(GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA, 2);

    builder.RegisterService(&service);
    std::unique_ptr<Server> server(builder.BuildAndStart());

    server->Wait();
    return 0;
}
