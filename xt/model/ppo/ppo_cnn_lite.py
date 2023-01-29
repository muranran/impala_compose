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

from xt.model.model_utils import ACTIVATION_MAP, get_cnn_backbone, get_cnn_default_settings, get_default_filters
from xt.model.multi_trainer import allreduce_optimizer, syn_init_model
from xt.model.ppo.default_config import CNN_SHARE_LAYERS
from xt.model.ppo.ppo import PPO
from xt.model.tf_compat import tf
from zeus.common.util.register import Registers
from xt.model import XTModel
from xt.model.ppo import actor_loss_with_entropy, critic_loss
from xt.model.tf_utils import TFVariables


@Registers.model
class PpoCnnLite(PPO):
    """Build PPO CNN network."""

    def __init__(self, model_info):
        model_config = model_info.get('model_config')

        self.vf_share_layers = model_config.get(
            'VF_SHARE_LAYERS', CNN_SHARE_LAYERS)
        self.hidden_sizes = model_config.get(
            'hidden_sizes', get_cnn_default_settings('hidden_sizes'))
        activation = model_config.get(
            'activation', get_cnn_default_settings('activation'))
        try:
            self.activation = ACTIVATION_MAP[activation]
        except KeyError:
            raise KeyError('activation {} not implemented.'.format(activation))

        # hrl parameters:
        self.backend = model_info.get("backend", "tf")
        self.inference_batchsize = model_info.get("inference_batchsize", 1)
        self.using_multi_learner = model_info.get("using_multi_learner", False)
        self.type = model_info.get('type', 'actor')
        
        super().__init__(model_info)

    def create_model(self, model_info):
        filter_arches = get_default_filters(self.state_dim)
        model = get_cnn_backbone(self.state_dim, self.action_dim, self.hidden_sizes, self.activation, filter_arches,
                                 self.vf_share_layers, self.verbose, dtype=self.input_dtype)
        self.build_graph(self.input_dtype, model)
        return model

    def build_graph(self, input_type, model):
        # pylint: disable=W0201
        self.state_ph = tf.placeholder(
            input_type, name='state', shape=(None, *self.state_dim))
        self.old_logp_ph = tf.placeholder(
            tf.float32, name='old_log_p', shape=(None, 1))
        self.adv_ph = tf.placeholder(
            tf.float32, name='advantage', shape=(None, 1))
        self.old_v_ph = tf.placeholder(
            tf.float32, name='old_v', shape=(None, 1))
        self.target_v_ph = tf.placeholder(
            tf.float32, name='target_value', shape=(None, 1))

        pi_latent, self.out_v = model(self.state_ph)

        if self.action_type == 'Categorical':
            self.behavior_action_ph = tf.placeholder(
                tf.int32, name='behavior_action', shape=(None,))
            dist_param = pi_latent
        elif self.action_type == 'DiagGaussian':
            # fixme: add input dependant log_std logic
            self.behavior_action_ph = tf.placeholder(
                tf.float32, name='real_action', shape=(None, self.action_dim))
            log_std = tf.get_variable('pi_logstd', shape=(
                1, self.action_dim), initializer=tf.zeros_initializer())
            dist_param = tf.concat(
                [pi_latent, pi_latent * 0.0 + log_std], axis=-1)
        else:
            raise NotImplementedError(
                'action type: {} not match any implemented distributions.'.format(self.action_type))

        self.dist.init_by_param(dist_param)
        self.action = self.dist.sample()
        self.action_log_prob = self.dist.log_prob(self.action)
        self.actor_var = TFVariables(
            [self.action_log_prob, self.out_v], self.sess)

        self.actor_loss = actor_loss_with_entropy(self.dist, self.adv_ph, self.old_logp_ph, self.behavior_action_ph,
                                                  self.clip_ratio, self.ent_coef)
        self.critic_loss = critic_loss(
            self.target_v_ph, self.out_v, self.old_v_ph, self.vf_clip)
        self.loss = self.actor_loss + self.critic_loss_coef * self.critic_loss
        self.train_op = self.build_train_op(self.loss)

        self.sess.run(tf.initialize_all_variables())
        if self.type is 'learner' and self.using_multi_learner:
            self.sess = syn_init_model(self.sess)

    def build_train_op(self, loss):
        if self.type is 'learner' and self.using_multi_learner:
            # self.optimizer = allreduce_optimizer(self._lr, tf.train.AdamOptimizer)
            trainer = allreduce_optimizer(
                self._lr, tf.train.AdamOptimizer)
        else:
            trainer = tf.train.AdamOptimizer(learning_rate=self._lr)
        grads_and_var = trainer.compute_gradients(loss)
        grads, var = zip(*grads_and_var)
        grads, _ = tf.clip_by_global_norm(grads, self._max_grad_norm)
        return trainer.apply_gradients(zip(grads, var))
