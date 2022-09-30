import logging
import os
import time
from multiprocessing import Process, Queue
from tensorflow.keras import backend as K
from tensorflow_core.python.framework import graph_util
import subprocess
from setproctitle import setproctitle
from typing import List, Dict
from tensorflow.compat.v1 import Graph, GraphDef, import_graph_def, Session
from tensorflow.compat.v1.gfile import GFile
import tensorflow as tf
import graph_def_editor as gde

os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)
DEFAULT_TFL_NAME = "model.tflite"


class CompressWeights:
    def __init__(self, **kwargs):
        shared_queue = kwargs.get("shared_queue")
        self.raw_weights_queue = shared_queue[0]  # type:Queue
        self.compress_weights_queue = shared_queue[1]  # type:Queue
        self.compress_tool = None

    def register_weights_process_function(self, func):
        self.compress_tool = func

    def task_loop(self):
        setproctitle("xt_compress")
        os.environ["CUDA_VISIBLE_DEVICES"] = str("-1")
        while True:
            if not self.raw_weights_queue.empty():
                raw_weight = self.raw_weights_queue.get()
                compress_weight = self.compress_tool(raw_weight)
                self.compress_weights_queue.put(compress_weight)

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


def save_tflite_as_bolt(config: Dict):
    tflite_to_bolt = config.get("X2bolt")
    tflite_file = config.get("tflite_file")

    p = subprocess.Popen(
        '/home/xys/bolt/install_linux-x86_64_avx2/tools/X2bolt -d /home/xys/bolt/test'
        ' -m ppo_cnn -i FP32', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return "Bolt File"


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
    tflite_file = exp2_p_f(config)
    config.update({"tflite_file": tflite_file})
    bolt_file = save_tflite_as_bolt(config)
    return bolt_file
