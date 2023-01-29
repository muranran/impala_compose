import argparse
from tensorflow.compat.v1.train import AdamOptimizer, Saver, linear_cosine_decay
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1 import global_variables_initializer
from tensorflow.compat.v1.keras.layers import Lambda
from tensorflow.compat.v1.keras.layers import Flatten
from tensorflow.compat.v1.keras.layers import Conv2D
# import keras
# from keras import custom_objects
from tensorflow.compat.v1.keras.models import load_model
from tensorflow.compat.v1.keras.utils import CustomObjectScope
from tensorflow.compat.v1.keras.backend import set_session
import tensorflow.compat.v1 as tf

# import tensorflow.contrib.eager as tfe

import tensorflow as tf2
import os
import re
import sys
import time

import numpy as np

from functools import partial, update_wrapper
import xt.model.impala.vtrace as vtrace
from tensorflow.python.util import deprecation
from xt.model.multi_trainer import allreduce_optimizer, syn_init_model
from zeus.common.util.register import Registers
from xt.model import XTModel
from xt.model.impala.default_config import GAMMA, LR
# from xt.model.tf_compat import (
#     DTYPE_MAP,
#     AdamOptimizer,
#     Conv2D,
#     Flatten,
#     Lambda,
#     Saver,
#     global_variables_initializer,
#     linear_cosine_decay,
#     tf,
# )
# from keras import (AdamOptimizer,Conv2D,)
# import keras as K

from xt.model.atari_model import get_atari_filter
from xt.model.tf_utils import TFVariables, restore_tf_variable
from xt.model.model_utils import state_transform, custom_norm_initializer
from zeus.common.util.common import import_config
from absl import logging
# from tensorflow_core.python.framework import graph_util
# from tensorflow import graph_util
from tensorflow.compat.v1 import graph_util
from multiprocessing import Queue, Process
tf.compat.v1.enable_eager_execution()

deprecation._PRINT_DEPRECATION_WARNINGS = False


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# common parameters
parser = argparse.ArgumentParser(description="compress_tool")
parser.add_argument("--weights", "-w", type=str)
parser.add_argument("--output", "-o", type=str)

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
    model = CompressModelV2()
    model.model.load_weights(weights)
    converter = tf2.lite.TFLiteConverter.from_keras_model(
        model.model)

    tflite_model = converter.convert()

    with open(tflite_saved_file, 'wb') as f:
        f.write(tflite_model)


def main():
    testfiledir = "/home/data/dxa/xingtian_revise/impala_compose/user/model"

    model = CompressModelV2()
    model.model.load_weights(testfiledir+"/weights.h5")
    print(model.model.summary())

    T0 = time.time()

    converter = tf2.lite.TFLiteConverter.from_keras_model(
        model.model)

    tflite_model = converter.convert()

    T1 = time.time()
    print("convert time: {:.2f}ms".format((T1-T0)*1000))


if __name__ == "__main__":
    # main()
    T0 = time.time()
    weights = args.weights
    tflite_model_file = args.output
    compress_weights(weights,tflite_model_file)
    T1 = time.time()
    
    print("compress time: {:.2f}ms".format((T1-T0)*1000))
    
