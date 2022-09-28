import os
import time
from multiprocessing import Process, Queue

import subprocess
from setproctitle import setproctitle

os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)
import tensorflow as tf


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
    time.sleep(3)
    return weight
