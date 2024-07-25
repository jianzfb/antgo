from __future__ import print_function

import logging, sys
sys.path.insert(0, './proto')
import grpc
import ${project}_pb2_grpc
import ${project}_pb2

def run():
	with grpc.insecure_channel('127.0.0.1:PORT') as channel:
		stup = ${project}_pb2_grpc.${servername}Stub(channel)
        
		response = stup.${servername}Start(${project}_pb2.${servername}StartRequest(serverpipeline="",serverid="", servercfg=""))
		print(response)

		response = stup.${servername}Call(${project}_pb2.${servername}CallRequest(serverkey=""))
		print(response)


if __name__ == '__main__':
	logging.basicConfig()
	run()
