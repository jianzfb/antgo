from __future__ import print_function

import logging, sys
sys.path.insert(0, './proto')
import grpc
import ${project}_pb2_grpc
import ${project}_pb2
import argparse


def run(ip, port):
	with grpc.insecure_channel(f'{ip}:{port}') as channel:
		stup = ${project}_pb2_grpc.${servername}Stub(channel)
        
		response = stup.${servername}Start(${project}_pb2.${servername}StartRequest(serverpipeline="",serverid="", servercfg=""))
		print(response)

		response = stup.${servername}Call(${project}_pb2.${servername}CallRequest(serverkey=""))
		print(response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='grpc')
    parser.add_argument('--ip', default="127.0.0.1", type=str, help ='ip')
    parser.add_argument('--port', default=9002, type=int, help ='port')
    args = parser.parse_args()
    run(args.ip, args.port)
