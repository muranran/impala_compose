import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from functools import partial, update_wrapper
from multiprocessing import Process, Queue
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf2
import tensorflow.compat.v1 as tf
import zmq
from absl import logging
from tensorflow.compat.v1 import (global_variables_initializer, graph_util,
                                  keras)
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1.keras.layers import Conv2D, Flatten, Lambda
# import keras
# from keras import custom_objects
from tensorflow.compat.v1.keras.models import load_model
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1.keras.utils import CustomObjectScope
from tensorflow.compat.v1.train import (AdamOptimizer, Saver,
                                        linear_cosine_decay)
# import xt.model.impala.vtrace as vtrace
from tensorflow.python.util import deprecation

from xt.model import XTModel
from xt.model.atari_model import get_atari_filter
from xt.model.impala.default_config import GAMMA, LR
from xt.model.model_utils import custom_norm_initializer, state_transform
from xt.model.multi_trainer import allreduce_optimizer, syn_init_model
from xt.model.tf_utils import TFVariables, restore_tf_variable
from zeus.common.util.common import import_config
from zeus.common.util.register import Registers

# import tensorflow.contrib.eager as tfe


# tf.compat.v1.enable_eager_execution()
tf.enable_eager_execution()

deprecation._PRINT_DEPRECATION_WARNINGS = False


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# common parameters
parser = argparse.ArgumentParser(description="compress_tool")
parser.add_argument("--weights", "-w", type=str,
                    default="/home/data/dxa/xingtian_revise/impala_compose/user/model/weights.h5")
parser.add_argument("--output", "-o", type=str,
                    default="/home/data/dxa/xingtian_revise/impala_compose/user/model/test_model2.h5")

parser.add_argument("--port", "-p", type=int, default=5555)

args = parser.parse_args()

DTYPE_MAP = {
    "float32": tf.float32,
    "float16": tf.float16,
}


class CompressModelV2:
    def __init__(self) -> None:
        self.model = None
        self._transform = partial(state_transform,
                                  mean=0.01,
                                  std=255.0,
                                  input_dtype=tf.float32)
        update_wrapper(self._transform, state_transform)
        self.create_tf2_model()

    def create_tf2_model(self, model_info=None):
        from tensorflow import keras
        self.ph_state2 = keras.Input(
            (84, 84, 4), dtype=tf.float32, name="state_input")
        state_input = keras.layers.Lambda(
            self._transform, name="state_input_fp32")(self.ph_state2)
        last_layer = state_input
        conv1 = keras.layers.Conv2D(16, (8, 8), strides=(
            4, 4), activation="relu", padding="same")(last_layer)
        conv2 = keras.layers.Conv2D(32, (4, 4), strides=(
            2, 2), activation="relu", padding="same")(conv1)
        conv3 = keras.layers.Conv2D(256, (11, 11), strides=(
            1, 1), activation="relu", padding="valid")(conv2)

        conv4 = keras.layers.Conv2D(
            4, (1, 1), padding="same")(conv3)

        self.pi_logic_outs2 = keras.layers.Lambda(
            lambda x: tf.squeeze(x, [1, 2]), name="pi_logic_outs")(conv4)
        flat = keras.layers.Flatten()(conv3)
        dense = keras.layers.Dense(
            units=1, activation=None, kernel_initializer=custom_norm_initializer(0.01))(flat)
        self.baseline2 = keras.layers.Lambda(
            lambda x: tf.squeeze(x, 1), name="baseline")(dense)
        self.fix_oa2 = keras.layers.Lambda(lambda x: tf.multinomial(x, num_samples=1,
                                                                    output_dtype=tf.int32))(self.pi_logic_outs2)
        self.out_actions2 = keras.layers.Lambda(
            lambda x: tf.squeeze(x, 1), name="out_action")(self.fix_oa2)
        model1 = keras.Model(inputs=[self.ph_state2],
                             outputs=[self.pi_logic_outs2, self.baseline2])

        self.model = model1

        return self.model


def compress_weights(weights, tflite_saved_file):
    T0 = time.time()
    model = CompressModelV2()
    T1 = time.time()
    model.model.load_weights(weights)
    T2 = time.time()

    converter = tf2.lite.TFLiteConverter.from_keras_model(
        model.model)

    tflite_model = converter.convert()
    T3 = time.time()
    with open(tflite_saved_file, 'wb') as f:
        f.write(tflite_model)

    print("create time: {:.2f}ms".format((T1-T0)*1000))
    print("restore time: {:.2f}ms".format((T2-T1)*1000))
    print("convert time: {:.2f}ms".format((T3-T2)*1000))


def main():
    testfiledir = "/home/data/dxa/xingtian_revise/impala_compose/user/model"
    T0 = time.time()
    model = CompressModelV2()
    model.model.load_weights(testfiledir+"/weights.h5")
    # print(model.model.summary())

    T1 = time.time()

    converter = tf2.lite.TFLiteConverter.from_keras_model(
        model.model)

    tflite_model = converter.convert()

    T2 = time.time()
    print("restore time: {:.2f}ms".format((T1-T0)*1000))
    print("convert time: {:.2f}ms".format((T2-T1)*1000))


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
    # bolt_flag = "PTQ"

    raw_proc_cmd = [tflite_to_bolt, "-d", path, "-m", file, "-i", bolt_flag]
    p = subprocess.run(raw_proc_cmd, capture_output=True,
                       check=True, encoding="utf-8")
    # p = subprocess.Popen(
    #     raw_proc_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    info = p.stdout
    bolt_path = re.findall("{}.*".format(path), info)[-1][:-1]
    # print(bolt_path)
    return bolt_path


def save_tflite_as_bolt_quant(config: Dict):
    tflite_to_bolt = config.get("X2bolt")
    bolt_quant = config.get("PTQ")
    tflite_file = config.get("tflite_file")
    path, file, suffix = spilt_path_file(tflite_file)

    # bolt_flag = "FP32"
    bolt_flag = "PTQ"

    # model convert
    raw_proc_cmd = [tflite_to_bolt, "-d", path, "-m", file, "-i", bolt_flag]
    p = subprocess.run(raw_proc_cmd, capture_output=True,
                       check=True, encoding="utf-8")
    info = p.stdout
    bolt_path = re.findall("{}.*".format(path), info)[-1][:-1]

    # quantize the model
    precision = "INT8_FP32"
    storage_format = "INT8"
    raw_proc_cmd = [bolt_quant, "-p", bolt_path,
                    "-i", precision, '-q', storage_format]
    p = subprocess.run(raw_proc_cmd, capture_output=True,
                       check=True, encoding="utf-8")
    info = p.stdout
    bolt_path = re.findall("{}.*".format(path), info)[-1][:-1]
    # print(bolt_path)
    return bolt_path


def compress_proc(port=5555, to_bolt=False):
    # create communication pipeline
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:{}".format(port))
    T0 = time.time()
    model = CompressModelV2()
    create_time = time.time() - T0
    while True:
        #  Wait for next request from client
        # print("Waiting for compress task")
        message = socket.recv().decode()
        # print("message:", message)
        if message == "exit":
            break

        weights, tflite_saved_file = message.split()
        if not os.path.exists(weights):
            continue

        # if tflite_saved_file.endswith(".bolt"):
        #     to_bolt = True
        #     tflite_saved_file = tflite_saved_file.replace(".bolt", ".tflite")
        # else:
        #     to_bolt = False

        T0 = time.time()
        model.model.load_weights(weights)
        restore_time = time.time() - T0

        T0 = time.time()
        converter = tf2.lite.TFLiteConverter.from_keras_model(
            model.model)

        tflite_model = converter.convert()

        with open(tflite_saved_file, 'wb') as f:
            f.write(tflite_model)

        assert os.path.exists(tflite_saved_file), "TFLite saved file not exist"
        convert_time = time.time() - T0

        time.sleep(1)

        saved_file = tflite_saved_file

        to_bolt = True
        quant = True

        if to_bolt:
            config_ = {
                "X2bolt": "/home/data/dxa/bolt/install_linux-x86_64_avx512_vnni/tools/X2bolt",
                "PTQ": "/home/data/dxa/bolt/install_linux-x86_64_avx512_vnni/tools/post_training_quantization",
                "tflite_file": tflite_saved_file,
            }
            if quant:
                bolt_saved_file = save_tflite_as_bolt_quant(config_)
            else:
                bolt_saved_file = save_tflite_as_bolt(config_)  # type:str
            assert bolt_saved_file.endswith(".bolt"), bolt_saved_file
            saved_file = bolt_saved_file

        message = "{} {:.2f} {:.2f} {:.2f}".format(saved_file,
                                                   create_time*1000,
                                                   restore_time*1000,
                                                   convert_time*1000)
        socket.send(message.encode())


def quant_test():
    config_ = {
        "X2bolt": "/home/data/dxa/bolt/install_linux-x86_64_avx512_vnni/tools/X2bolt",
        "PTQ": "/home/data/dxa/bolt/install_linux-x86_64_avx512_vnni/tools/post_training_quantization",
        "tflite_file": "/home/data/dxa/xingtian_revise/impala_compose/user/model/test_model_0.tflite",
    }

    save_tflite_as_bolt_quant(config_)


if __name__ == "__main__":
    # compress_proc(args.port)
    quant_test()
