import os
from multiprocessing import Process, Queue
from threading import Thread
from queue import Queue as qQueue
from setproctitle import setproctitle, setthreadtitle
from time import time, sleep
from psutil import Process as ProcManager
import logging


class CompressWeights:
    def __init__(self, **kwargs):

        self.num_workers = kwargs.get("num_workers", 4)
        shared_queue = kwargs.get("shared_queue")
        self.raw_weights_queue = shared_queue[0]  # type:Queue
        self.compress_weights_queue = shared_queue[1]  # type:Queue
        self.compress_tool = None
        self.running_proc = []
        self.schedule_comm = Queue() if kwargs.get("schedule", False) else None
        self.pending_workers = []
        self.running_workers = []
        self.schedule_threshold = 4

    def register_weights_process_function(self, func):
        self.compress_tool = func

    @staticmethod
    def worker(carrier, raw_weights_, compress_weights_, compress_tool, schedule_comm, cid):
        setproctitle("xt_compress_{}".format(cid))
        print("xt_compress_worker_{} start...".format(cid))
        idle_times = 0
        while True:
            if not raw_weights_.empty():
                raw_weight = raw_weights_.get()
                try:
                    start_0 = time()
                    compress_weight_ = compress_tool(raw_weight)
                    compress_time = time() - start_0
                    print("==================================\n"
                          "Compress_{} Time: {:.2f} ms\n"
                          "==================================\n"
                          .format(cid, compress_time * 1000))
                except ValueError as err_v:
                    print(err_v)
                    continue
                compress_weights_.put(compress_weight_)
            else:
                if idle_times > 100:
                    if schedule_comm is not None:
                        schedule_comm.put((cid, os.getpid()))
                    sleep(10)
                    idle_times = 0
                sleep(0.2)
                idle_times += 1

    # carrier type: "P" : process, "T" : thread
    def multi_task(self, num_workers, carrier="P"):
        worker_carrier = Process if carrier == "P" else Thread
        raw_weights_ = self.raw_weights_queue
        compress_weights_ = self.compress_weights_queue

        self.running_proc = [worker_carrier(target=self.worker,
                                            args=(carrier, raw_weights_,
                                                  compress_weights_, self.compress_tool,
                                                  self.schedule_comm, cid,))
                             for cid in range(num_workers)]

        # for proc in self.running_proc:
        #     proc.setDaemon(True)
        return self

    def __start_(self):
        for proc in self.running_proc:
            proc.start()

    def start(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str("-1")
        self.multi_task(self.num_workers).__start_()
        # Process(target=self.task_loop).start()

    def schedule(self):
        input_size = self.raw_weights_queue.qsize()
        output_size = self.compress_weights_queue.qsize()
        if input_size > self.schedule_threshold and len(self.pending_workers) > 0:
            (cid, pid) = self.pending_workers.pop()
            ProcManager(pid).resume()

        while not self.schedule_comm.empty():
            (cid, pid) = self.schedule_comm.get()
            ProcManager(pid).suspend()
            self.pending_workers.append((cid, pid))

        return len(self.pending_workers)


def empty_compress_tool(weight):
    sleep(4)
    return "C"


if __name__ == '__main__':
    raw_weights = Queue()
    compress_weights = Queue()
    compress_workers = CompressWeights(shared_queue=[raw_weights, compress_weights])
    compress_workers.register_weights_process_function(empty_compress_tool)
    compress_workers.start()

    print("Start Test Multi-Task Compress Tool.")

    last_transfer_time = time()
    for i in range(1000):
        sleep(2)
        raw_weights.put("C")
        print("Putting Raw Weight...")
        print("R: {}\tC: {}".format(raw_weights.qsize(), compress_weights.qsize()))
        # pending_workers = compress_workers.schedule()
        pending_workers = 0
        if not compress_weights.empty():
            compress_weight = compress_weights.get()
            current_time = time()
            print("Start Transferring Compressed Weight...W : {}\tT : {:.2f} ms\tP : {}"
                  .format(compress_weight, (current_time-last_transfer_time)*1000, pending_workers))
            last_transfer_time = current_time


    pass
