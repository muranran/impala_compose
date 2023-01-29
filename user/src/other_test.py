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
import os
import re
import sys
import time


import tensorflow as tf
from tensorflow.compat.v1.train import AdamOptimizer, linear_cosine_decay
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1 import global_variables_initializer
import tensorflow.compat.v1 as tf1


import numpy as np

from functools import partial, update_wrapper
import xt.model.impala.vtrace as vtrace
from tensorflow.python.util import deprecation
from xt.model.multi_trainer import allreduce_optimizer, syn_init_model
from zeus.common.util.register import Registers
from xt.model import XTModel
from xt.model.impala.default_config import GAMMA, LR
from xt.model.atari_model import get_atari_filter
from xt.model.tf_utils import TFVariables, restore_tf_variable
from xt.model.model_utils import state_transform, custom_norm_initializer
from zeus.common.util.common import import_config
from absl import logging
from multiprocessing import Queue, Process

deprecation._PRINT_DEPRECATION_WARNINGS = False


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# common parameters


DTYPE_MAP = {
    "float32": tf.float32,
    "float16": tf.float16,
}


class ImpalaCnnOptLiteQt:
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
    print("Excuting eagerly: {}".format(tf.executing_eagerly()))
    
    T0 = time.time()    
    model = ImpalaCnnOptLiteQt(model_info)
    compress_model = model.infer_model
    converter = tf.lite.TFLiteConverter.from_keras_model(compress_model)
    converter.convert()
    
    T1 = time.time()
    s2ms = lambda t1,t0:(t1 - t0)*1000
    
    print("Compress Time : {:.2f}".format(s2ms(T1,T0)))
