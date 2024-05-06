import pika
import multiprocessing
import os
import logging
from multiprocessing import current_process
import time
from tfrecord import example_pb2


_worker_info = dict()


def worker_sender(loader, server, port, queue_prefix, consumer_size, worker_id, worker_size, rank_id, seed, max_queue_size):
    global _worker_info
    _worker_info[os.getpid()] = dict(
        num_workers=worker_size,
        id=worker_id,
        seed=seed,
        rank=rank_id
    )

    # 创建链接
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(server, port)
    )
    channel = connection.channel()

    # 推送数据
    queue_list = []
    for consumer_queue_i in range(consumer_size):
        queue_name = f'{queue_prefix}/{consumer_queue_i}'
        channel.queue_declare(queue=queue_name)
        queue_list.append(queue_name)

    sample_i = 0
    for data in loader:
        consumer_queue_i = sample_i % consumer_size
        channel.basic_publish(
            exchange='',
            routing_key=queue_list[consumer_queue_i],
            body=data.tobytes()
        )

        queue_state = channel.queue_declare(queue=queue_list[consumer_queue_i], passive=True)
        while queue_state.method.message_count > max_queue_size:
            time.sleep(1)
            queue_state = channel.queue_declare(queue=queue_list[consumer_queue_i], passive=True)

        sample_i += 1

    # 关闭链接
    connection.close()


def get_data_worker_info():
    global _worker_info
    pid = os.getpid()
    info = dict(
        num_workers=1,
        id=0,
        rank=0,
        seed=0
    )
    if pid in _worker_info:
        info = _worker_info[pid]

    class InfoS(object):
        def __init__(self, body):
            for k,v in body.items():
                setattr(self, k, v)

    return InfoS(info)


class DataServer(object):
    def __init__(self, name, loader, server, port=5672, consumer_size=1, worker_num=1, epoch_num=1, max_queue_size=1024):
        self.name = name
        self.loader = loader
        self.worker_num = worker_num
        self.rank_size = self.loader.rank_size
        self.process_objs = None
        self.epoch_num = epoch_num
        self.server = server
        self.port = port
        self.max_queue_size = max_queue_size
        self.consumer_size = consumer_size

    def start(self):
        # 创建线程，并开始服务
        for epoch_i in range(self.epoch_num):
            print(f'dataset {self.name} epoch {epoch_i} generate')
            self.loader.epoch = epoch_i
            self.process_objs = []

            # 加载数据
            for process_i in range(self.worker_num*self.rank_size):
                worker_i = process_i // self.rank_size
                rank_i = process_i - int(worker_i * self.rank_size)
                p = multiprocessing.Process(
                    target=worker_sender, 
                    args=(
                        self.loader, 
                        self.server, self.port, 
                        f'{self.name}/rank{rank_i}', 
                        self.consumer_size, 
                        worker_i, 
                        self.worker_num, 
                        rank_i, 
                        0, 
                        self.max_queue_size
                    )
                )
                p.start()
                self.process_objs.append(p)

            for thread_obj in self.process_objs:
                thread_obj.join()

            # 结束标记
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(self.server, self.port)
            )
            channel = connection.channel()
            for rank_i in range(self.rank_size):
                for consumer_queue_i in range(self.consumer_size):
                    channel.queue_declare(queue=f'{self.name}/rank{rank_i}/{consumer_queue_i}')
                    channel.basic_publish(
                        exchange='',
                        routing_key=f'{self.name}/rank{rank_i}/{consumer_queue_i}',
                        body=b''
                    )
            connection.close()


def worker_receiver(name, server, port, rank, consumer_id, callback):
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(server, port)
    )

    channel = connection.channel()
    channel.queue_declare(queue=f'{name}/rank{rank}/{consumer_id}')
    channel.basic_consume(
        queue=f'{name}/rank{rank}/{consumer_id}',
        on_message_callback=callback,
        auto_ack=False)

    channel.start_consuming()


class DataReceiver(object):
    def __init__(self, name, server, port=5672, rank=0, consumer_id=0, consumer_num=1, waiting_num=1, maxsize=1024):
        self.name = name
        self.rank = rank
        self.consumer_id = consumer_id
        self.consumer_num = consumer_num
        self.waiting_num = waiting_num
        self.waiting_i = 0

        self.queue = multiprocessing.Queue(maxsize=maxsize)
        p = multiprocessing.Process(target=worker_receiver, args=(name, server, port, rank, consumer_id, self.callback))
        p.daemon = True
        p.start()

    def callback(self, ch, method, properties, body):
        self.queue.put(body)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def __iter__(self):
        return self

    def __next__(self):
        val = self.queue.get()
        if val == b'':
            self.waiting_i += 1
            if self.waiting_i == self.waiting_num:
                self.waiting_i = 0
                raise StopIteration

        return val
