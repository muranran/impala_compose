# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
Implement the impala cnn network with tensorflow.

The Implement of Vtrace_loss refers to deepmind/scalable_agent.
https://github.com/deepmind/scalable_agent

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

See https://arxiv.org/abs/1802.01561 for the full paper.
"""
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
            (84,84,4), dtype=tf.float32, name="state_input")
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

@Registers.model
class ImpalaCnnOptLiteQt(XTModel):
    """Docstring for ActorNetwork."""

    def __init__(self, model_info):
        self.inference = None
        self.interpreter = None
        model_config = model_info.get("model_config", dict())
        import_config(globals(), model_config)
        self.dtype = DTYPE_MAP.get(model_info.get("default_dtype", "float32"))
        self.input_dtype = model_info.get("input_dtype", "float32")
        self.sta_mean = model_info.get("state_mean", 0.)
        self.sta_std = model_info.get("state_std", 255.)

        # def state_transform2(x, mean=1e-5, std=255., input_dtype="float32"):
        #     """Normalize data."""
        #     # if input_dtype in ("float32", "float", "float64"):
        #     #     return x
        #
        #     # only cast non-float32 state
        #     if np.abs(mean) < 1e-4:
        #         return tf.cast(x, dtype='float32') / std
        #     else:
        #         return (tf.cast(x, dtype="float32") - mean) / std

        self._transform = partial(state_transform,
                                  mean=self.sta_mean,
                                  std=self.sta_std,
                                  input_dtype=self.input_dtype)
        update_wrapper(self._transform, state_transform)

        self.state_dim = model_info["state_dim"]
        self.action_dim = model_info["action_dim"]
        self.filter_arch = get_atari_filter(self.state_dim)

        # lr schedule with linear_cosine_decay
        self.lr_schedule = model_config.get("lr_schedule", None)
        self.opt_type = model_config.get("opt_type", "adam")
        self.lr = None

        self.ph_state = None
        self.ph_adv = None
        self.out_actions = None
        self.pi_logic_outs, self.baseline = None, None

        # placeholder for behavior policy logic outputs
        self.ph_bp_logic_outs = None
        self.ph_actions = None
        self.ph_dones = None
        self.ph_rewards = None
        self.loss, self.optimizer, self.train_op = None, None, None

        self.grad_norm_clip = model_config.get("grad_norm_clip", 40.0)
        self.sample_batch_steps = model_config.get("sample_batch_step", 50)

        self.saver = None
        self.explore_paras = None
        self.actor_var = None  # store weights for agent

        self.bolt_interpreter = None
        self.backend = model_info.get("backend", "tf")
        self.inference_batchsize = model_info.get("inference_batchsize", 1)
        self.using_multi_learner = model_info.get("using_multi_learner", False)

        self.type = model_info.get('type', 'actor')
        self.infer_model = None
        self.tf2_model = None
        self.compress_model = CompressModelV2()

        super().__init__(model_info)

    def create_model(self, model_info):
        self.ph_state = keras.Input(
            self.state_dim, dtype=self.input_dtype, name="state_input")
        state_input = keras.layers.Lambda(
            self._transform, name="state_input_fp32")(self.ph_state)
        last_layer = state_input
        conv1 = keras.layers.Conv2D(16, (8, 8), strides=(
            4, 4), activation="relu", padding="same")(last_layer)
        conv2 = keras.layers.Conv2D(32, (4, 4), strides=(
            2, 2), activation="relu", padding="same")(conv1)
        conv3 = keras.layers.Conv2D(256, (11, 11), strides=(
            1, 1), activation="relu", padding="valid")(conv2)

        conv4 = keras.layers.Conv2D(
            self.action_dim, (1, 1), padding="same")(conv3)

        self.pi_logic_outs = keras.layers.Lambda(
            lambda x: tf.squeeze(x, [1, 2]), name="pi_logic_outs")(conv4)
        flat = keras.layers.Flatten()(conv3)
        dense = keras.layers.Dense(
            units=1, activation=None, kernel_initializer=custom_norm_initializer(0.01))(flat)
        self.baseline = keras.layers.Lambda(
            lambda x: tf.squeeze(x, 1), name="baseline")(dense)
        self.fix_oa = keras.layers.Lambda(lambda x: tf.multinomial(x, num_samples=1,
                                                                   output_dtype=tf.int32))(self.pi_logic_outs)
        self.out_actions = keras.layers.Lambda(
            lambda x: tf.squeeze(x, 1), name="out_action")(self.fix_oa)
        model1 = keras.Model(inputs=[self.ph_state],
                             outputs=[self.pi_logic_outs, self.baseline])

        self.infer_model = model1

        print(model1.summary())

        # create learner
        self.ph_bp_logic_outs = tf.placeholder(self.dtype, shape=(None, self.action_dim),
                                               name="ph_b_logits")

        self.ph_actions = tf.placeholder(
            tf.int32, shape=(None,), name="ph_action")
        self.ph_dones = tf.placeholder(tf.bool, shape=(None,), name="ph_dones")
        self.ph_rewards = tf.placeholder(
            self.dtype, shape=(None,), name="ph_rewards")

        batch_step = self.sample_batch_steps

        def split_batches(tensor, drop_last=False):
            batch_count = tf.shape(tensor)[0] // batch_step
            reshape_tensor = tf.reshape(
                tensor,
                tf.concat([[batch_count, batch_step],
                          tf.shape(tensor)[1:]], axis=0),
            )

            # swap B and T axes
            res = tf.transpose(
                reshape_tensor,
                [1, 0] + list(range(2, 1 + int(tf.shape(tensor).shape[0]))),
            )

            if drop_last:
                return res[:-1]
            return res

        self.loss = vtrace_loss(
            bp_logic_outs=split_batches(self.ph_bp_logic_outs, drop_last=True),
            tp_logic_outs=split_batches(self.pi_logic_outs, drop_last=True),
            actions=split_batches(self.ph_actions, drop_last=True),
            discounts=split_batches(
                tf.cast(~self.ph_dones, tf.float32) * GAMMA, drop_last=True),
            rewards=split_batches(tf.clip_by_value(
                self.ph_rewards, -1, 1), drop_last=True),
            values=split_batches(self.baseline, drop_last=True),
            bootstrap_value=split_batches(self.baseline)[-1],
        )

        global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        if self.opt_type == "adam":
            if self.lr_schedule:
                learning_rate = self._get_lr(global_step)
            else:
                learning_rate = LR
            optimizer = AdamOptimizer(learning_rate)
        elif self.opt_type == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(
                LR, decay=0.99, epsilon=0.1, centered=True)
        else:
            raise KeyError("invalid opt_type: {}".format(self.opt_type))

        grads_and_vars = optimizer.compute_gradients(self.loss)

        # global norm
        grads, var = zip(*grads_and_vars)
        grads, _ = tf.clip_by_global_norm(grads, self.grad_norm_clip)
        clipped_gvs = list(zip(grads, var))

        self.train_op = optimizer.apply_gradients(
            clipped_gvs, global_step=global_step)

        # fixme: help to show the learning rate among training processing
        self.lr = optimizer._lr

        self.actor_var = TFVariables(self.out_actions, self.sess)

        self.sess.run(global_variables_initializer())

        return True

    def create_tf2_model(self, model_info=None):
        self.ph_state2 = keras.Input(
            self.state_dim, dtype=self.input_dtype, name="state_input")
        state_input = keras.layers.Lambda(
            self._transform, name="state_input_fp32")(self.ph_state)
        last_layer = state_input
        conv1 = keras.layers.Conv2D(16, (8, 8), strides=(
            4, 4), activation="relu", padding="same")(last_layer)
        conv2 = keras.layers.Conv2D(32, (4, 4), strides=(
            2, 2), activation="relu", padding="same")(conv1)
        conv3 = keras.layers.Conv2D(256, (11, 11), strides=(
            1, 1), activation="relu", padding="valid")(conv2)

        conv4 = keras.layers.Conv2D(
            self.action_dim, (1, 1), padding="same")(conv3)

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

        self.tf2_model = model1

        print(model1.summary())
        pass

    def compress_test(self, weight_queue):
        os.environ["CUDA_DEVICES_VISIBLE"] = ""
        tf.compat.v1.enable_eager_execution()
        print("eager start")
        print("create model")
        # compress_model = CompressModelV2()
        compress_model = self.compress_model
        print("Waiting for weight")
        cmd, weight = weight_queue.get()
        print("get weight")
        if cmd == "weight":
            print("start compress")
            compress_model.model.set_weights(weight)
            # converter = tf.lite.TFLiteConverter.from_keras_model_file(
            #     keras_model, custom_objects={
            #         "state_transform": model1._transform, "_initializer": None}
            # )
            T0 = time.time()
            converter = tf2.lite.TFLiteConverter.from_keras_model(
                compress_model.model)

            # converter.inference_input_type = tf.uint8
            # converter.quantized_input_stats = {'state_input': (128, 127)}

            tflite_model = converter.convert()

            T1 = time.time()
            print("convert time: {:.2f}ms".format((T1-T0)*1000))

        pass
    
    def start_compress(self,weight_quene):
        weight_compress_process = Process(target=self.compress_test,args=(weight_quene,))
        weight_compress_process.start()
        return weight_compress_process
        

    def _get_lr(self, global_step, decay_step=20000.):
        """Make decay learning rate."""
        lr_schedule = self.lr_schedule
        if len(lr_schedule) != 2:
            logging.warning("Need 2 elements in lr_schedule!\n, "
                            "likes [[0, 0.01], [20000, 0.000001]]")
            logging.fatal("lr_schedule invalid: {}".format(lr_schedule))

        if lr_schedule[0][0] != 0:
            logging.info("lr_schedule[0][1] could been init learning rate")

        learning_rate = linear_cosine_decay(lr_schedule[0][1],
                                            global_step, decay_step,
                                            beta=lr_schedule[1][1] / float(decay_step))

        return learning_rate

    def train(self, state, label):
        """Train with sess.run."""

        bp_logic_outs, actions, dones, rewards = label
        with self.graph.as_default():
            _, loss = self.sess.run(
                [self.train_op, self.loss],
                feed_dict={
                    self.ph_state: state,
                    self.ph_bp_logic_outs: bp_logic_outs,
                    self.ph_actions: actions,
                    self.ph_dones: dones,
                    self.ph_rewards: rewards,
                },
            )
        return loss
        # return 0.

    def predict_(self, state):
        """
        Do predict use the newest model.

        :param: state
        :return: action_logits, action_val, value
        """
        # with self.graph.as_default():
        #     set_session(self.sess)
        #     return self.infer_model.predict([state])
        with self.graph.as_default():
            feed_dict = {self.ph_state: state}
            return self.sess.run([self.pi_logic_outs, self.baseline, self.out_actions],
                                 feed_dict)

    def save_model(self, file_name):
        """Save model without meta graph."""
        ck_name = self.saver.save(
            self.sess, save_path=file_name, write_meta_graph=False)
        return ck_name

    def predict(self, state):
        if self.backend == "tf" or self.backend == "tensorflow":
            return self.predict_(state)

        # if self.input_dtype == "float32":
        #     state = [s.astype(np.float32) for s in state]
        # else:
        #     state = [s.astype(np.uint8) for s in state]

        def state_transform2(x, mean=1e-5, std=255., input_dtype="float32"):
            if np.abs(mean) < 1e-4:
                return x.astype(np.float32) / std
            else:
                return (x.astype(np.float32) - mean) / std

        # state = [state_transform2(s, self.sta_mean, self.sta_std)
        #          for s in state]

        batch_size = self.interpreter["input_shape"][0]
        real_batch_size = len(state)
        # state = [np.zeros((84, 84, 4), dtype=np.float32) for i in range(5)]
        # real_batch_size = 5
        if real_batch_size > batch_size:
            state_zero = np.zeros(state[0].shape, dtype=np.float32)
            num_predict = real_batch_size // batch_size
            rest_num = real_batch_size % batch_size
            state_batch_resize = []
            for i in range(num_predict):
                state_batch_resize.append(
                    state[i * batch_size: (i + 1) * batch_size])
            state_batch_resize.append(
                [*state[num_predict * batch_size:], *[state_zero for i in range(batch_size - rest_num)]])
            pi_logic_outs = []
            baseline = []
            for s in state_batch_resize:
                try:
                    p, b = self.inference(s)
                    # pt, bt = self.invoke(s)
                    # logging.info("======================================\n"
                    #              "tflite result:\n{}\n"
                    #              "bolt result:\n{}\n"
                    #              "======================================\n"
                    #              .format((pt, bt), (pb, bb)))
                    #
                    # p = pt
                    # b = bt
                    pi_logic_outs.extend(p)
                    baseline.extend(b)
                except ValueError as err:
                    raise ValueError(
                        err.args + ("| {} | {}".format((len(p), len(b)), (p, b)),))
            # print(pi_logic_outs, baseline)
            pi_logic_outs = pi_logic_outs[:real_batch_size]
            baseline = baseline[:real_batch_size]
            # raise RuntimeError("debug...| p:\n{} \n b:\n{}".format(len(pi_logic_outs), len(baseline)))
        elif real_batch_size == batch_size:
            pi_logic_outs, baseline = self.inference(state)

            # extra_p, extra_b = self.extra_inference(state)
            # logging.info("======================================\n"
            #              "tflite result:\n{}\n"
            #              "bolt result:\n{}\n"
            #              "======================================\n"
            #              .format((pi_logic_outs, baseline), (extra_p, extra_b)))
            # pi_logic_outs = pt
            # baseline = bt
        else:
            raise NotImplementedError("state_batch_size < inference_batch_size | {} < {}"
                                      .format(real_batch_size, batch_size))

        t0 = time.time()

        def softmax(logits):
            e_x = np.exp(logits)
            probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
            return probs

        try:
            actions = np.asarray(
                [np.argmax(np.random.multinomial(1, softmax(plo))) for plo in pi_logic_outs])
        except ValueError as err:
            print("pi_logic_outs:\n{}\nmay contains negative.{}".format(
                pi_logic_outs, err))
            actions = np.random.randint(0, 4, (len(pi_logic_outs),))
            pi_logic_outs = np.zeros(shape=np.asarray(pi_logic_outs).shape)
        at = time.time() - t0
        # print("====================Action Time : {:.2f}=========================".format(at * 1000))
        return pi_logic_outs, baseline, actions

    def create_tflite_interpreter(self, tflite_model_path):
        # Load the TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        assert interpreter is not None, "Interpreter is None."
        interpreter.allocate_tensors()
        # print(output)
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']

        # logging.info("================output details {}==================".format(output_details))
        # raise RuntimeError("debug...")

        # update interpreter info
        self.tflite_interpreter = {
            "interpreter": interpreter,
            "input_index": input_details[0]['index'],
            "input_shape": input_shape,
            "pi_logic_outs_index": output_details[0]['index'],
            "baseline_index": output_details[1]['index'],
        }

    def create_bolt_interpreter(self, bolt_model_path):
        # Load the bolt model and allocate tensors.
        module_path = "/home/data/cypo/bolt"
        try:
            sys.path.append(module_path)
            import batch_infer as bolt
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "bolt module not found under path:{}".format(module_path))
        bolt_interpreter = bolt.Dog.get_instance()
        bolt_interpreter.prepare(bolt_model_path)

        # update interpreter info
        self.bolt_interpreter = {
            "interpreter": bolt_interpreter,
            # fixme: experimental revise | pipeline 3 | default 1
            "input_shape": (self.inference_batchsize, 4, 84, 84),
            "pi_logic_outs_index": 1,
            "baseline_index": 0,
        }

    def create_bolt_interpreter_fix(self, bolt_model_path):
        # Load the bolt model and allocate tensors.
        module_path = "/home/data/cypo/bolt"
        try:
            sys.path.append(module_path)
            import bolt_python_interface as bolt
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "bolt module not found under path:{}".format(module_path))
        bolt_interpreter = bolt.BoltModelWrapper.get_instance()
        bolt_interpreter.prepare(
            bolt_model_path, self.inference_batchsize, self.action_dim)

        # update interpreter info
        self.bolt_interpreter = {
            "interpreter": bolt_interpreter,
            # fixme: experimental revise | pipeline 3 | default 1
            "input_shape": (self.inference_batchsize, 4, 84, 84),
            "pi_logic_outs_index": 1,
            "baseline_index": 0,
        }

    def invoke_tflite(self, state):
        input_data = state
        interpreter = self.tflite_interpreter["interpreter"]
        interpreter.set_tensor(
            self.tflite_interpreter["input_index"], input_data)
        interpreter.invoke()
        pi_logic_outs = interpreter.get_tensor(
            self.tflite_interpreter["pi_logic_outs_index"])
        baseline = interpreter.get_tensor(
            self.tflite_interpreter["baseline_index"])
        return pi_logic_outs.tolist(), baseline.tolist()

    def invoke_bolt(self, state):
        input_data = np.expand_dims(np.transpose(
            np.array(state), (0, 3, 1, 2)).flatten(), 1)
        interpreter = self.bolt_interpreter["interpreter"]
        interpreter.inference(input_data)
        pi_logic_outs = interpreter.get_result(
            self.bolt_interpreter["pi_logic_outs_index"])
        baseline = interpreter.get_result(
            self.bolt_interpreter["baseline_index"])

        # time.sleep(0.002)

        return pi_logic_outs.tolist(), baseline.tolist()

    def save_as_h5(self, filename: str):
        if filename is None:
            self.infer_model.save("./mymodel.h5")
        else:
            if filename.endswith(".h5"):

                self.infer_model.save(filename)
            else:
                raise NotImplementedError(
                    "unsupport file type {}".format(filename.split("[\/]")[-1]))

    def load_model(self, model_name, by_name=False):
        """Load model with inference variables."""

        restore_tf_variable(self.sess, self.explore_paras, model_name)

    def set_weights(self, weights):
        if self.backend == "bolt":
            if isinstance(weights, str):
                if weights.endswith(".bolt"):
                    self.inference = self.invoke_bolt
                    self.set_bolt_weight(weights)
                    self.interpreter = self.bolt_interpreter
                else:
                    raise TypeError(
                        "{} doesn't end with .bolt".format(weights))
            else:
                raise TypeError("{} is not path-like".format(weights))

        elif self.backend == "tflite":
            if isinstance(weights, str):
                if weights.endswith(".tflite"):
                    self.inference = self.invoke_tflite
                    self.set_tflite_weights(weights)
                    self.interpreter = self.tflite_interpreter
                else:
                    raise TypeError(
                        "{} doesn't end with .tflite".format(weights))
            else:
                raise TypeError("{} is not path-like".format(weights))

        elif self.backend == "tf" or self.backend == "tensorflow":
            self.set_tf_weights(weights)

        elif self.backend == "bolt_":
            if isinstance(weights, str):
                if weights.endswith(".bolt"):
                    self.inference = self.invoke_tflite
                    self.extra_inference = self.invoke_bolt
                    self.set_bolt_weight(weights)
                    result = re.search(
                        "model_([0-9]+[0-9]*).*?(\.[a-z]*)", weights)
                    tflite_weights = "model_{}.tflite".format(result.group(1))
                    tflite_weights = weights.replace(
                        result.group(0), tflite_weights)
                    self.set_tflite_weights(tflite_weights)
                    self.interpreter = self.tflite_interpreter
                    self.extra_interpreter = self.bolt_interpreter
                else:
                    raise TypeError(
                        "{} doesn't end with .tflite".format(weights))
            else:
                raise TypeError("{} is not path-like".format(weights))

        else:
            raise NotImplementedError(
                "{} has not been implemented".format(self.backend))

    def set_tflite_weights(self, weights):
        # logging.info(
        #     "====================Create TFLite Interpreter======================")
        self.create_tflite_interpreter(weights)

    def set_bolt_weight(self, weights):
        # logging.info(
        #     "====================Create Bolt Interpreter======================")
        self.create_bolt_interpreter_fix(weights)

    def set_tf_weights(self, weights):
        """Set weight with memory tensor."""
        with self.graph.as_default():
            set_session(self.sess)
            # self.actor_var.set_weights(weights)
            self.infer_model.set_weights(weights)

    def get_weights(self, backend="default"):
        """Get weights."""
        if backend == "default":
            with self.graph.as_default():
                set_session(self.sess)
                return self.infer_model.get_weights()

        if backend == "default_":
            backend = self.backend

        if backend == "tf" or backend == "tensorflow":
            with self.graph.as_default():
                return self.actor_var.get_weights()

        elif backend == "bolt" or backend == "tflite":
            if not hasattr(self, "model_num"):
                setattr(self, "model_num", 0)
            else:
                setattr(self, "model_num", (getattr(self, "model_num") + 1) % 25)
            save_path = "/home/data/dxa/xingtian_revise/impala_opt/user/data/model/model_{}.pb". \
                format(getattr(self, "model_num"))
            pb_file = self.freeze_graph(save_path)
            return pb_file


def calc_baseline_loss(advantages):
    """Calculate the baseline loss."""
    return 0.5 * tf.reduce_sum(tf.square(advantages))


def calc_entropy_loss(logic_outs):
    """Calculate entropy loss."""
    pi = tf.nn.softmax(logic_outs)
    log_pi = tf.nn.log_softmax(logic_outs)
    entropy_per_step = tf.reduce_sum(-pi * log_pi, axis=-1)
    return -tf.reduce_sum(entropy_per_step)


def calc_pi_loss(logic_outs, actions, advantages):
    """Calculate policy gradient loss."""
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=actions, logits=logic_outs)
    advantages = tf.stop_gradient(advantages)
    pg_loss_per_step = cross_entropy * advantages
    return tf.reduce_sum(pg_loss_per_step)


def vtrace_loss(
        bp_logic_outs, tp_logic_outs, actions,
        discounts, rewards, values, bootstrap_value):
    """
    Compute vtrace loss for impala algorithm.

    :param bp_logic_outs: behaviour_policy_logic_outputs
    :param tp_logic_outs: target_policy_logic_outputs
    :param actions:
    :param discounts:
    :param rewards:
    :param values:
    :param bootstrap_value:
    :return: total loss
    """
    with tf.device("/cpu"):
        value_of_state, pg_advantages = vtrace.from_logic_outputs(
            behaviour_policy_logic_outputs=bp_logic_outs,
            target_policy_logic_outputs=tp_logic_outs,
            actions=actions,
            discounts=discounts,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
        )

    pi_loss = calc_pi_loss(tp_logic_outs, actions, pg_advantages)
    val_loss = calc_baseline_loss(value_of_state - values)
    entropy_loss = calc_entropy_loss(tp_logic_outs)

    return pi_loss + 0.5 * val_loss + 0.01 * entropy_loss


if __name__ == "__main__":
    model_info = {
        "model_name": "ImpalaCnnOptLite",
        "state_dim": [84, 84, 4],
        "input_dtype": "uint8",  # default is uint8
        "state_mean": 0.0,
        "state_std": 255.0,
        "action_dim": 4,
        "model_config": {
            "LR": 0.0005,
            "sample_batch_step": 160,
            "grad_norm_clip": 40.0
        },
        "backend": "tf",  # tf, tflite, bolt
        "inference_batchsize": 160,
        "using_multi_learner": False
    }
    # model1 = ImpalaCnnOptLiteQt(model_info)
    # weight_quene = Queue()
    # weight_compress_proc = model1.start_compress(weight_quene)
    # weight = model1.infer_model.get_weights()
    # weight_quene.put(("weight",weight))
    
    # weight_compress_proc.join()
    testfiledir = "/home/data/dxa/xingtian_revise/impala_compose/user/model"

    model = CompressModelV2()
    model.model.load_weights(testfiledir+"/weights.h5")
    print(model.model.summary())

    # testfiledir = "/home/data/dxa/xingtian_revise/impala_compose/user/model"
    # keras_model = os.path.join(testfiledir, "test_model.h5")
    # model1.save_as_h5(keras_model)
    # # model1.infer_model.save(keras_model)
    # T0 = time.time()
    # model2 = tf.keras.models.load_model(keras_model, compile=False, custom_objects={
    #                                     "state_transform": model1._transform, "_initializer": custom_norm_initializer(0.01)})

    # with CustomObjectScope({"state_transform":model1._transform}):
    #     model2 = keras.models.load_model(keras_model)
    # model2 = tf.keras.models.load_model(keras_model, compile=False, custom_objects={
    #                                     "state_transform": model1._transform, "_initializer": None})

    # converter = tf.lite.TFLiteConverter.from_keras_model_file(
    #     keras_model, custom_objects={
    #         "state_transform": model1._transform, "_initializer": None}
    # )
    T0 = time.time()

    converter = tf2.lite.TFLiteConverter.from_keras_model(
        model.model)

    # converter.inference_input_type = tf.uint8
    # converter.quantized_input_stats = {'state_input': (128, 127)}

    tflite_model = converter.convert()

    T1 = time.time()
    print("convert time: {:.2f}ms".format((T1-T0)*1000))
