import logging
import google.protobuf.message
import subprocess
from typing import List, Dict
from tensorflow.compat.v1 import Graph, GraphDef, import_graph_def, Session
from tensorflow.compat.v1.gfile import GFile
import tensorflow.compat.v1 as tf
import graph_def_editor as gde
import re
from typing import Tuple
import os
from multiprocessing import Process, Queue
from threading import Thread
from queue import Queue as qQueue
from setproctitle import setproctitle, setthreadtitle
from time import time, sleep
from psutil import Process as ProcManager

os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)
DEFAULT_TFL_NAME = "model.tflite"


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

    def task_loop(self):
        setproctitle("xt_compress")
        os.environ["CUDA_VISIBLE_DEVICES"] = str("-1")
        while True:
            if not self.raw_weights_queue.empty():
                raw_weight = self.raw_weights_queue.get()
                try:
                    start_0 = time()

                    compress_weight_ = self.compress_tool(raw_weight)

                    compress_time = time() - start_0
                    logging.info("==================================\n"
                                 "Compress Time: {:.2f} ms\n"
                                 "==================================\n"
                                 .format(compress_time * 1000))
                except google.protobuf.message.DecodeError as err:
                    print("\"{}\" has been overlaid")
                    continue
                except ValueError as err_v:
                    print(err_v)
                    continue
                self.compress_weights_queue.put(compress_weight_)
            # else:
            #     time.sleep(0.2)

    @staticmethod
    def worker(carrier, raw_weights_, compress_weights_, compress_tool, schedule_comm, cid):
        setproctitle("xt_compress_{}".format(cid))
        print("xt_compress_worker_{} start...".format(cid))
        idle_times = 0
        while True:
            # print("TYPE:{}, {}, {}".format(type(raw_weights_), raw_weights_.empty(), raw_weights_.qsize()))
            if not raw_weights_.empty():
                raw_weight = raw_weights_.get()
                try:
                    start_0 = time()

                    compress_weight_ = compress_tool(raw_weight)
                    # compress_weight_ = None

                    compress_time = time() - start_0
                    print("==================================\n"
                          "Compress_{} Time: {:.2f} ms\n"
                          "==================================\n"
                          .format(cid, compress_time * 1000))
                except ValueError as err_v:
                    logging.info("===================GET BUG {}====================".format(err_v))
                    continue
                compress_weights_.put(compress_weight_)
            else:
                if idle_times > 100:
                    if schedule_comm is not None:
                        schedule_comm.put((cid, os.getpid()))
                    sleep(2)
                    idle_times = 0
                sleep(0.5)
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

    def start_multi_task(self):
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

    def start(self):
        Process(target=self.task_loop).start()

    @property
    def instance(self) -> List[Process]:
        return self.running_proc


# backup
# def serialize_bolt(model_file_path):
#     model1 = tf.keras.models.load_model(model_file_path, custom_objects={"CustomModel": CustomModel})
#     converter = tf.lite.TFLiteConverter.from_keras_model(model1)
#     # converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     tflite_model = converter.convert()
#     with open('./tmp/model/ppo_cnn.tflite', 'wb') as f:
#         f.write(tflite_model)
#
#     p = subprocess.Popen('/home/tank/dxa/pure_bolt/bolt/install_linux-x86_64_avx2/tools/X2bolt -d ./tmp/model/'
#                          ' -m ppo_cnn -i FP32', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def empty_weights_proc_func(weights):
    return weights


def experiment_1_proc_func(weight):
    logging.info("===================Compress Weight=====================")
    sleep(4)
    return weight


def save_as_tflite(pb_file: str, input_names: List[str], output_names: List[str], save_path: str):
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file=pb_file,
        input_arrays=input_names,
        output_arrays=output_names
    )
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    if save_path.endswith(".tflite"):
        tflite_file = save_path
    else:
        tflite_file = os.path.join(save_path, DEFAULT_TFL_NAME)
    with tf.io.gfile.GFile(tflite_file, 'wb') as f:
        f.write(tflite_model)

    return tflite_file


def save_as_tflite_resize_fix(pb_file: str, input_names: List[str], output_names: List[str], save_path: str,
                              new_batch_size=3):
    with GFile(pb_file, "rb") as f:
        graph_def = GraphDef()
        graph_def.ParseFromString(f.read())

    with Graph().as_default() as graph:
        import_graph_def(graph_def,
                         input_map=None,
                         return_elements=None,
                         name=""
                         )
    g = gde.Graph(graph.as_graph_def())
    gde.rewrite.change_batch_size(g, new_size=new_batch_size, inputs=[g["state_input"]])

    graph_revised = g.to_graph_def()
    with Graph().as_default() as graph_r:
        import_graph_def(graph_revised,
                         input_map=None,
                         return_elements=None,
                         name=""
                         )
    x = graph_r.get_tensor_by_name("state_input:0")
    y = graph_r.get_tensor_by_name("explore_agent/pi_logic_outs:0")
    z = graph_r.get_tensor_by_name("explore_agent/baseline:0")
    sess = tf.Session(graph=graph_r)

    # print("===================CONVERTER 0 ===================")
    converter = tf.lite.TFLiteConverter.from_session(sess, [x], [y, z])

    # converter.inference_input_type = tf.uint8
    # converter.quantized_input_stats = {'state_input': (128, 127)}
    # print("===================CONVERTER 1 ===================")
    tflite_model = converter.convert()
    # print("===================CONVERTER C ===================")
    if save_path.endswith(".tflite"):
        tflite_file = save_path
    else:
        tflite_file = os.path.join(save_path, DEFAULT_TFL_NAME)
    # print("===================CONVERTER W ===================")
    with tf.io.gfile.GFile(tflite_file, 'wb') as f:
        f.write(tflite_model)
    # print("===================CONVERTER E ===================")
    return tflite_file


def spilt_path_file(path_like: str) -> Tuple[str, str, str]:
    pattern1 = r"[/]"
    pattern2 = r"[.]"

    def find_last(p, s):
        ret = re.finditer(p, s)
        split_position = list(ret)[-1].span()[0]
        return split_position

    sp1 = find_last(pattern1, path_like)
    path = path_like[:sp1]
    file = path_like[sp1 + 1:]
    sp2 = find_last(pattern2, file)
    file_name = file[:sp2]
    suffix = file[sp2:]
    return path, file_name, suffix


def save_tflite_as_bolt(config: Dict):
    tflite_to_bolt = config.get("X2bolt")
    tflite_file = config.get("tflite_file")
    path, file, suffix = spilt_path_file(tflite_file)

    bolt_flag = "FP32"
    raw_proc_cmd = [tflite_to_bolt, "-d", path, "-m", file, "-i", bolt_flag]
    p = subprocess.run(raw_proc_cmd, capture_output=True, check=True, encoding="utf-8")
    # p = subprocess.Popen(
    #     raw_proc_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    info = p.stdout
    bolt_path = re.findall("{}.*".format(path), info)[-1][:-1]
    # print(bolt_path)
    return bolt_path


# bolt cmd
# /home/data/dxa/bolt/install_linux-x86_64_avx512_vnni/tools/X2bolt -d
# /home/data/dxa/xingtian_revise/impala_opt/user/data/model -m model_0 -i FP32

# config example: { "pb_file" : "/home/tank/dxa/xingtian_revise/impala_opt/user/data/model/imp25.pb"
#   "input_names" : ["state_input"]
#   "output_names" : ["explore_agent/pi_logic_outs"]
#   "save_path" : "/home/tank/dxa/xingtian_revise/impala_opt/user/data/model/" }
def exp2_p_f(config_: Dict):
    pb_file = config_.get("pb_file")
    input_names = config_.get("input_names")
    output_names = config_.get("output_names")
    save_path = config_.get("save_path")
    resize_batch_size = config_.get("resize_batch_size", 3)
    # fixme: experimental revise
    resize_batch_size = 1

    # tflite_file = save_as_tflite(pb_file, input_names, output_names, save_path)
    try:
        tflite_file = save_as_tflite_resize_fix(pb_file, input_names, output_names, save_path, resize_batch_size)
    except:
        logging.info("=======================FLAG ENCOUNTER BUG=================================")
        raise RuntimeError("UNKNOWN BUG")
    logging.info("=====================Complete Weight Compress========================")
    return tflite_file


def exp3_p_f(config_: Dict):
    default_config = {
        "X2bolt": "/home/data/dxa/bolt/install_linux-x86_64_avx512_vnni/tools/X2bolt",
    }
    tflite_file = exp2_p_f(config_)
    config_.update({"tflite_file": tflite_file})
    config_.update(default_config)
    bolt_file = save_tflite_as_bolt(config_)  # type:str
    assert bolt_file.endswith(".bolt"), bolt_file
    # from termcolor import colored
    # print(colored("{}".format(bolt_file), "red"))
    return bolt_file


# get fp32 from uint8 (input type)
def pb_to_bolt_from_template(config_: Dict):
    model_store = "/home/data/dxa/xingtian_revise/impala_opt/user/data/model"
    pb_file = "/home/data/dxa/xingtian_revise/impala_opt/user/data/model_0.pb"
    model_fp32_template = "model_fp32_template.pb"

    pass


if __name__ == '__main__':
    # config = {
    #     "X2bolt": "/home/data/dxa/bolt/install_linux-x86_64_avx512_vnni/tools/X2bolt",
    #     "tflite_file": "/home/data/dxa/xingtian_revise/impala_opt/user/data/model/model_10.tflite",
    # }
    # bolt_path = save_tflite_as_bolt(config)
    # print(bolt_path)

    # config = {
    #     "pb_file": "/home/data/dxa/xingtian_revise/impala_opt/user/data/model/model_0.pb",
    #     "input_names": ["state_input"],
    #     "output_names": ["explore_agent/pi_logic_outs", "explore_agent/baseline"],
    #     "save_path": "/home/data/dxa/xingtian_revise/impala_opt/user/data/model/model_0.tflite"
    # }
    # start_0 = time()
    # bolt_path = exp2_p_f(config)
    # tt = time() - start_0
    # print(bolt_path)
    # print("time is : {:.2f} ms".format(tt*1000))

    raw_weights = Queue()
    compress_weights = Queue()
    compress_workers = CompressWeights(shared_queue=[raw_weights, compress_weights])
    compress_workers.register_weights_process_function(exp2_p_f)
    compress_workers.start_multi_task()

    print("Start Test Multi-Task Compress Tool.")
    config = {
        "pb_file": "/home/data/dxa/xingtian_revise/impala_opt/user/data/model/model_0.pb",
        "input_names": ["state_input"],
        "output_names": ["explore_agent/pi_logic_outs", "explore_agent/baseline"],
        "save_path": "/home/data/dxa/xingtian_revise/impala_opt/user/data/model/model_0.tflite"
    }
    last_transfer_time = time()
    for i in range(1000):
        sleep(2)
        raw_weights.put(config)
        print("Putting Raw Weight...")
        print("R: {}\tC: {}".format(raw_weights.qsize(), compress_weights.qsize()))
        # pending_workers = compress_workers.schedule()
        pending_workers = 0
        if not compress_weights.empty():
            compress_weight = compress_weights.get()
            current_time = time()
            print("Start Transferring Compressed Weight...W : {}\tT : {:.2f} ms\tP : {}"
                  .format(compress_weight, (current_time - last_transfer_time) * 1000, pending_workers))
            last_transfer_time = current_time
