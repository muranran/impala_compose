import logging
import google.protobuf.message
from tensorflow_core.python.framework import graph_util
import subprocess
from typing import List, Dict
from tensorflow.compat.v1 import Graph, GraphDef, import_graph_def, Session
from tensorflow.compat.v1.gfile import GFile
import tensorflow as tf
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
                    compress_weight = self.compress_tool(raw_weight)
                    compress_time = time() - start_0
                    logging.info("==================================\n"
                                 "Compress Time: {:.2f} ms\n"
                                 "==================================\n"
                                 .format(compress_time*1000))
                except google.protobuf.message.DecodeError as err:
                    print("\"{}\" has been overlaid")
                    continue
                except ValueError as err_v:
                    print(err_v)
                    continue
                self.compress_weights_queue.put(compress_weight)
            # else:
            #     time.sleep(0.2)

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
    time.sleep(3)
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


def save_as_tflite_resize_fix(pb_file: str, input_names: List[str], output_names: List[str], save_path: str):
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
    gde.rewrite.change_batch_size(g, new_size=3, inputs=[g["state_input"]])
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
    sess = Session(graph=graph_r)
    converter = tf.lite.TFLiteConverter.from_session(sess, [x], [y, z])
    tflite_model = converter.convert()
    if save_path.endswith(".tflite"):
        tflite_file = save_path
    else:
        tflite_file = os.path.join(save_path, DEFAULT_TFL_NAME)
    with tf.io.gfile.GFile(tflite_file, 'wb') as f:
        f.write(tflite_model)
    return tflite_file


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""

        frozen_graph = graph_util.convert_variables_to_constants(session, input_graph_def, output_names,
                                                                 freeze_var_names)
        if not clear_devices:
            for node in frozen_graph.node:
                node.device = "/GPU:0"
        return frozen_graph


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


# config example:
#   {
#       "pb_file" : "/home/tank/dxa/xingtian_revise/impala_opt/user/data/model/imp25.pb"
#       "input_names" : ["state_input"]
#       "output_names" : ["explore_agent/pi_logic_outs"]
#       "save_path" : "/home/tank/dxa/xingtian_revise/impala_opt/user/data/model/"
#   }
def exp2_p_f(config: Dict):
    pb_file = config.get("pb_file")
    input_names = config.get("input_names")
    output_names = config.get("output_names")
    save_path = config.get("save_path")

    # tflite_file = save_as_tflite(pb_file, input_names, output_names, save_path)
    tflite_file = save_as_tflite_resize_fix(pb_file, input_names, output_names, save_path)
    logging.info("=====================Complete Weight Compress========================")
    return tflite_file


def exp3_p_f(config: Dict):
    default_config = {
        "X2bolt": "/home/data/dxa/bolt/install_linux-x86_64_avx512_vnni/tools/X2bolt",
    }
    tflite_file = exp2_p_f(config)
    config.update({"tflite_file": tflite_file})
    config.update(default_config)
    bolt_file = save_tflite_as_bolt(config)
    return bolt_file


if __name__ == '__main__':
    config = {
        "X2bolt": "/home/data/dxa/bolt/install_linux-x86_64_avx512_vnni/tools/X2bolt",
        "tflite_file": "/home/data/dxa/xingtian_revise/impala_opt/user/data/model/model_10.tflite",
    }
    bolt_path = save_tflite_as_bolt(config)
    print(bolt_path)
