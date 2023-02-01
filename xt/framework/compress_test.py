import logging
import os
from threading import Thread, Lock
from typing import List
import setproctitle
import zmq
from multiprocessing import Process
from psutil import Process as ProcManager
from subprocess import Popen, PIPE
from queue import Queue
import shlex
from time import sleep, time


class CompressWeights:
    def __init__(self, **kwargs):
        # worker settings
        self.num_workers = kwargs.get("num_workers", 1)
        self.workers = []
        self.hidden_workers = []

        # compress tools
        self.python_interpreter = "/home/tank/miniconda3/envs/tf2/bin/python"
        self.compress_tool_file = "user/src/compress_tool.py"

        # comm parameters
        self.context_list = []
        self.socket_list = []

        self.quene_visit_lock = Lock()
        self.task_queue = Queue(maxsize=10)
        self.result_queue = Queue()

        # compression parameters
        self.compress_results = []
        self.compress_tasks = []

    def create_comm_channel(self, port):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:{}".format(port))

        self.context_list.append(context)
        self.socket_list.append(socket)

    def create_compress_worker(self, port):

        sp = Popen(shlex.split("{} {} -p {}".format(self.python_interpreter, self.compress_tool_file, port)),
                   shell=False, stdout=PIPE, stderr=PIPE)

        self.hidden_workers.append(sp)

        sleep(0.1)

        print("worker status: {}".format(sp.poll()))

        self.create_comm_channel(port)

    def start_transfer(self, worker_id=0):

        count = 0
        while True:

            cmd = self.task_queue.get()

            if cmd.startswith("exit"):
                self.socket_list[worker_id].send(b'exit')
                # self.socket_list[worker_id].recv()
                break
            else:
                weights_and_target = cmd

            self.socket_list[worker_id].send(weights_and_target.encode())

            result = self.socket_list[worker_id].recv().decode()

            target, create_time, restore_time, convert_time = result.split()

            self.result_queue.put(target)

            # print("worker_{} in {} step".format(worker_id, count))
            # count += 1
            # print("worker_{} complete status:".format(worker_id))
            # print("create_time:{}\nrestore_time:{}\nconvert_time:{}"
            #       .format(create_time, restore_time, convert_time))

    def start(self):
        for worker_id in range(self.num_workers):
            self.create_compress_worker(5555+worker_id)

        for worker_id in range(self.num_workers):
            self.workers.append(
                Thread(target=self.start_transfer, args=(worker_id,)))

        for worker in self.workers:
            worker.start()

    def compress(self, weights, target):
        compress_task = weights + " " + target
        if self.task_queue.full():
            self.task_queue.get()
            
        # while (not self.task_queue.empty()):
        #     self.task_queue.get()

        self.task_queue.put(compress_task)
        # self.status()

    def get_result(self, block=False):
        if block:
            while (self.result_queue.empty()):
                sleep(0.1)
            assert not self.result_queue.empty(), "result queue is empty"

        ret = None
        if not self.result_queue.empty():

            ret = self.result_queue.get()

        if block:
            assert ret is not None, "RWN {}".format(self.result_queue.qsize())

        return ret

    def stop(self):
        for worker_id in range(self.num_workers):
            self.task_queue.put('exit')

        sleep(0.5)

        for hidden_worker_id, hidden_worker in enumerate(self.hidden_workers):
            print("hidden_worker_{}".format(hidden_worker_id),
                  "exit", hidden_worker.poll())
            if hidden_worker.poll() is None:
                hidden_worker.terminate()

            sleep(0.5)
            print("hidden_worker_{}".format(hidden_worker_id),
                  "exit", hidden_worker.poll())

    def status(self):
        task_pile = self.task_queue.qsize()
        worker_alive = [hidden_worker.poll()
                        for hidden_worker in self.hidden_workers]
        state_str = "{} ".format(task_pile) + \
            ("{} "*self.num_workers).format(*worker_alive)
        print("status", state_str)


def test():
    testfiledir = "/home/data/dxa/xingtian_revise/impala_compose/user/model"
    compress_tool_file = "user/src/compress_tool.py"
    weight_file = testfiledir+"/weights.h5"
    tflite_model_file = testfiledir+"/test_model.tflite"

    # start compressing
    sp = Popen(shlex.split("/home/tank/miniconda3/envs/tf2/bin/python {}".format(
        compress_tool_file)), shell=False, stdout=PIPE, stderr=PIPE)  # nvidia-smi dmon

    sleep(0.5)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    #  Do 10 requests, waiting each time for a response
    for request in range(10):
        message = weight_file + " " + tflite_model_file
        socket.send(message.encode())

        #  Get the reply.

        message = socket.recv().decode()
        print("tflite: ", message)

    socket.send(b'terminated')
    # sp.terminate()
    print("wait for terminal")
    sleep(0.5)
    print("status: ", sp.poll())
    # sp.kill()
    sp.wait()
    result = sp.stdout.read().decode()+sp.stderr.read().decode()
    print(result)

    pass


def main():
    testfiledir = "/home/data/dxa/xingtian_revise/impala_compose/user/model"
    compress_tool_file = "user/src/compress_tool.py"
    weight_file = testfiledir+"/weights.h5"
    tflite_model_file = testfiledir+"/test_model.tflite"

    compress_weight_manager = CompressWeights(num_workers=4)
    compress_weight_manager.start()

    task_count = 0
    i = 0
    while True:
        if i < 10:
            compress_weight_manager.compress(
                weight_file, testfiledir+"/test_model_{}.tflite".format(i))
            i += 1
        else:
            i += 1

        tflite_file = compress_weight_manager.get_result()
        if tflite_file is None:
            sleep(0.2)
            print("Compressing continue...{}...{}".format(
                i, compress_weight_manager.task_queue.qsize()))
        else:
            task_count += 1
            print("tflite_file: {} exists {}".format(
                tflite_file, os.path.exists(tflite_file)))

        if task_count == 10:
            break

    compress_weight_manager.stop()


if __name__ == '__main__':
    main()
