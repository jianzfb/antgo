# -*- coding: UTF-8 -*-
# @Time    : 17-8-23
# @File    : dataflow_server.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import plyvel
import threading
import zmq
import time
from antgo.dataflow.daemon import *

db_lock = threading.Lock()
db_pool = {}
db_reference_count = {}
db_client = {}


class DFServerThread(threading.Thread):
  """Dataflow Server Thread"""
  def __init__(self, context):
    threading.Thread.__init__(self)
    self.context = context
    self.running = True
    self.processing = False
    self.socket = self.context.socket(zmq.XREQ)
  
  def _open_db(self, db_path, id):
    db_lock.acquire()
    try:
      if id not in db_client:
        if db_path not in db_pool:
          db_pool[db_path] = plyvel.DB(db_path)
          db_reference_count[db_path] = 0

        db_reference_count[db_path] += 1
        db_client[id] = db_path
    finally:
      db_lock.release()
      return True
    
  def _close_db(self, id):
    db_lock.acquire()
    try:
      if id in db_client:
        db_path = db_client[id]
        db_reference_count[db_path] -= 1
        if db_reference_count[db_path] == 0:
          db_pool[db_path].close()
          db_pool.pop(db_path)
          db_reference_count.pop(db_path)
          
        db_client.pop(id)
    finally:
      db_lock.release()
      return True
  
  def run(self):
    self.socket.connect('inproc://mltalker-df-backend')
    while self.running:
      try:
        msg = self.socket.recv_multipart()
      except zmq.ZMQError:
        self.running = False
        continue
      
      self.processing = True
      if len(msg) != 3:
        value = b'None'
        reply = [msg[0], value]
        self.socket.send_multipart(reply)
        continue
      id = msg[0]
      op = msg[1]
      data = msg[2]
      reply = [id]
      if op == b'get':
        try:
          value = db_pool[db_client[id]].get(data)
        except:
          value = b""
        reply.append(value)
      elif op == b'open':
        try:
          if self._open_db(data, id):
            reply.append(b"OK")
          else:
            reply.append(b"")
        except:
          reply.append(b"")
      elif op == b'close':
        try:
          if self._close_db(id):
            reply.append(b"OK")
          else:
            reply.append(b"")
        except:
          reply.append(b"")
      else:
        value = b""
        reply.append(value)
        
      self.socket.send_multipart(reply)
      self.processing = False
  
  def close(self):
    self.running = False
    while self.processing:
      time.sleep(1)
    self.socket.close()
    
    
def dataflow_server(threads_num=3, host='tcp://127.0.0.1:9999'):
  context = zmq.Context()
  frontend = context.socket(zmq.XREP)
  frontend.bind(host)
  
  backend = context.socket(zmq.XREQ)
  backend.bind('inproc://mltalker-df-backend')
  
  poll = zmq.Poller()
  poll.register(frontend, zmq.POLLIN)
  poll.register(backend, zmq.POLLIN)
  
  workers = []
  for i in range(threads_num):
    worker = DFServerThread(context)
    worker.start()
    workers.append(worker)
  
  try:
    count = 0
    while True:
      sockets = dict(poll.poll())
      if frontend in sockets:
        if sockets[frontend] == zmq.POLLIN:
          msg = frontend.recv_multipart()
          backend.send_multipart(msg)
      
      if backend in sockets:
        if sockets[backend] == zmq.POLLIN:
          msg = backend.recv_multipart()
          frontend.send_multipart(msg)
          
      count += 1

  except KeyboardInterrupt:
    for worker in workers:
      worker.close()
    frontend.close()
    backend.close()
    context.term()


class DataflowClient(object):
  """dataflow client"""
  
  def __init__(self, host="tcp://127.0.0.1:9999", timeout=10):
    self.host = host
    self.timeout = timeout
    self.context = zmq.Context()
    self.socket = self.context.socket(zmq.XREQ)
    self.socket.connect(self.host)

  def get(self, key):
    self.socket.send_multipart([b'get', key])
    data = self.socket.recv_multipart()[0]
    return data
  
  def open(self, db):
    self.socket.send_multipart([b'open', db])
    data = self.socket.recv_multipart()[0]
    return data
  
  def close(self, db):
    # self.socket.send_multipart([b'close', db])
    # data =  self.socket.recv_multipart()[0]
    # return data
    pass
  
  def dataflow_close(self):
    self.socket.close()
    self.context.term()


class DataflowServerDaemon(Daemon):
  def __init__(self, threads, host, pidfile):
    super(DataflowServerDaemon, self).__init__(pidfile)
    self._threads = threads
    self._host = host
  
  def run(self):
    dataflow_server(self._threads, self._host)


if __name__ == "__main__":
    # daemon = DataflowServerDaemon(1,'tcp://127.0.0.1:9999',"/home/mi/dataflow_demodaemon.pid")
    # daemon.start()

    dataflow_server(1)
