import pika
import multiprocessing
import os
import logging
from multiprocessing import current_process
import time
import torch.distributed as dist
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
    def __init__(self, name, loader, server, port=5672, consumer_size=1, worker_num=1, epoch_num=1, max_queue_size=1024, is_distribute=False):
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
        self.is_distribute = is_distribute
        if self.is_distribute:
            dist.init_process_group(backend='mpii')

    def get_dist_info(self):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        return rank, world_size

    def start(self):
        # 分布式环境信息
        dist_rank, dist_world_size = self.get_dist_info()

        # 数据基本信息 (sample num/epoch)
        sample_num = len(self.loader)
        if self.is_distribute:
            tensor_list = [torch.zeros(1, dtype=torch.int64) for _ in range(dist_world_size)]
            tensor = torch.from_numpy(np.array([sample_num], dtype=np.int64))
            dist.all_gather(tensor_list, tensor)
            sample_num = int(torch.sum(torch.concat(tensor_list[dist_rank])).cpu().numpy())

        if dist_rank == 0:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(self.server, self.port)
            )

            channel = connection.channel()
            for rank_i in range(self.rank_size):
                channel.queue_declare(queue=f'{self.name}/rank{rank_i}/basic')
                channel.basic_publish(
                    exchange='',
                    routing_key=f'{self.name}/rank{rank_i}/basic',
                    body=f'{sample_num}'
                )
            connection.close()

        # 启动数据服务
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

            for process_obj in self.process_objs:
                process_obj.join()

            if self.is_distribute:
                dist.barrier()

            # 结束标记
            if dist_rank == 0:
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


def worker_receiver(name, server, port, receiver_queue_name, callback):
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(server, port)
    )

    channel = connection.channel()
    channel.queue_declare(queue=receiver_queue_name)
    channel.basic_consume(
        queue=receiver_queue_name,
        on_message_callback=callback,
        auto_ack=False)

    channel.start_consuming()


class DataReceiver(object):
    sample_num_in_epoch = 0
    def __init__(self, name, server, port=5672, rank=0, consumer_id=0, consumer_num=1, maxsize=1024):
        self.name = name
        self.rank = rank
        self.consumer_id = consumer_id
        self.consumer_num = consumer_num
        self.server = server
        self.port = port

        self.receiver_queue = multiprocessing.Queue(maxsize=maxsize)
        self.receiver_queue_name = f'{name}/rank{rank}/{consumer_id}'
        p = multiprocessing.Process(target=worker_receiver, args=(name, server, port, self.receiver_queue_name, self.callback))
        p.daemon = True
        p.start()

    def callback(self, ch, method, properties, body):
        self.receiver_queue.put(body)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def __iter__(self):
        return self

    def __next__(self):
        val = self.receiver_queue.get()
        if val == b'':
            raise StopIteration

        return val

    @classmethod
    def sample_num_receiver_callback(cls, ch, method, properties, body):
        DataReceiver.sample_num_in_epoch = int(body)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        ch.stop_consuming()

    @classmethod
    def sample_num(cls, name, server, port, rank):
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(server, port)
        )

        channel = connection.channel()
        channel.queue_declare(queue=f'{name}/rank{rank}/basic')

        channel.basic_consume(
            queue=f'{name}/rank{rank}/basic',
            on_message_callback=DataReceiver.sample_num_receiver_callback,
            auto_ack=False)

        channel.start_consuming()
        return DataReceiver.sample_num_in_epoch