import os
import time

import tensorflow as tf
from tensorflow import keras
import numpy as np

from functools import partial, update_wrapper
LR = 0.0003
ENTROPY_LOSS = 0.01
HIDDEN_SIZE = 128
NUM_LAYERS = 1
GAMMA = 0.99


def get_atari_filter(shape):
    """Get default model set for atari environments."""
    shape = list(shape)
    # (out_size, kernel, stride)
    filters_84x84 = [
        [16, 8, 4],
        [32, 4, 2],
        [256, 11, 1],
    ]
    filters_42x42 = [
        [16, 4, 2],
        [32, 4, 2],
        [256, 11, 1],
    ]
    if len(shape) == 3 and shape[:2] == [84, 84]:
        return filters_84x84
    elif len(shape) == 3 and shape[:2] == [42, 42]:
        return filters_42x42
    else:
        raise ValueError(
            "Without default architecture for obs shape {}".format(shape))


def custom_norm_initializer(std=0.5):
    """Perform Customize norm initializer for op."""
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def import_config(global_para, config):
    """
    Import config.

    :param global_para
    :param config
    :return: None
    """
    if not config:
        return
    for key in config.keys():
        if key in global_para:
            global_para[key] = config[key]


def state_transform(x, mean=1e-5, std=255., input_dtype="float32"):
    """Normalize data."""
    if input_dtype in ("float32", "float", "float64"):
        return x

    # only cast non-float32 state
    if np.abs(mean) < 1e-4:
        return tf.cast(x, dtype='float32') / std
    else:
        return (tf.cast(x, dtype="float32") - mean) / std


DTYPE_MAP = {
    "float32": tf.float32,
    "float16": tf.float16,
}


class QuantizedModel:
    def __init__(self, model_info) -> None:
        model_config = model_info.get("model_config", dict())
        import_config(globals(), model_config)
        self.dtype = DTYPE_MAP.get(model_info.get("default_dtype", "float32"))
        self.input_dtype = model_info.get("input_dtype", "float32")
        self.sta_mean = model_info.get("state_mean", 0.)
        self.sta_std = model_info.get("state_std", 255.)

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
                             outputs=[self.pi_logic_outs, self.baseline, self.out_actions])

        self.model = model1

        print(model1.summary())

        # create learner
        # self.ph_bp_logic_outs = tf.placeholder(self.dtype, shape=(None, self.action_dim),
        #                                        name="ph_b_logits")

        self.ph_bp_logic_outs = keras.Input(
            (None, self.action_dim), dtype=self.dtype, name="ph_bp_logits")

        # self.ph_actions = tf.placeholder(
        #     tf.int32, shape=(None,), name="ph_action")

        self.ph_actions = keras.Input(
            (None,), dtype=tf.int32, name="ph_actions")

        # self.ph_dones = tf.placeholder(tf.bool, shape=(None,), name="ph_dones")

        self.ph_dones = keras.Input((None,), dtype=tf.bool, name="ph_dones")

        # self.ph_rewards = tf.placeholder(
        #     self.dtype, shape=(None,), name="ph_rewards")

        self.ph_dones = keras.Input(
            (None,), dtype=self.dtype, name="ph_rewards")

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
            # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
            optimizer = tf.optimizers.Adam(learning_rate)
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

        return True

    def split_batches(self, tensor, drop_last=False):
        batch_step = self.sample_batch_steps
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

    def train(self,state,label):
        split_batches = self.split_batches
        bp_logic_outs, actions, dones, rewards = label
        
        with tf.GradientTape as tape:
            self.pi_logic_outs,self.baseline,self.out_actions = self.model(state)

            self.loss = vtrace_loss(
                bp_logic_outs=split_batches(
                    bp_logic_outs, drop_last=True),
                tp_logic_outs=split_batches(
                    self.pi_logic_outs, drop_last=True),
                actions=split_batches(actions, drop_last=True),
                discounts=split_batches(
                    tf.cast(~dones, tf.float32) * GAMMA, drop_last=True),
                rewards=split_batches(tf.clip_by_value(
                    rewards, -1, 1), drop_last=True),
                values=split_batches(self.baseline, drop_last=True),
                bootstrap_value=split_batches(self.baseline)[-1],
            )
            


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


if __name__ == '__main__':
    print(tf.executing_eagerly(), tf.__version__)
