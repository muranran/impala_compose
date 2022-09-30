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
"""Build vectorized multi-environment in Atari agent for impala algorithm."""
import logging
from time import sleep, time
import numpy as np
from collections import defaultdict, deque

# revised by ZZX
from xt.agent import Agent
from zeus.common.ipc.message import message, set_msg_info
from zeus.common.util.register import Registers


# import time


@Registers.agent
class AtariImpalaOpt(Agent):  # revised by ZZX. previous: AtariImpalaOpt(CartpoleImpala)
    """Build Atari agent with IMPALA algorithm."""

    def __init__(self, env, alg, agent_config, **kwargs):
        self.vector_env_size = kwargs.pop("vector_env_size")

        # added by ZZX
        self.wait_num = kwargs.pop("wait_num")

        super().__init__(env, alg, agent_config, **kwargs)
        self.keep_seq_len = True  # to keep max sequence length in explorer.
        self.next_logit = None
        self.broadcast_weights_interval = agent_config.get("sync_model_interval", 1)
        self.sync_weights_count = self.broadcast_weights_interval  # 0, sync with start

        # vector environment will auto reset in step
        self.transition_data["done"] = False
        self.sample_vector = dict()
        for env_id in range(self.vector_env_size):
            self.sample_vector[env_id] = defaultdict(list)

        self.reward_track = deque(
            maxlen=self.vector_env_size * self.broadcast_weights_interval)
        self.reward_per_env = defaultdict(float)

        # added by ZZX
        self.last_info = [{'env_id': _} for _ in range(self.wait_num)]

    def get_explore_mean_reward(self):
        """Calculate explore reward among limited trajectory."""
        return np.nan if not self.reward_track else np.nanmean(self.reward_track)

    def infer_action(self, state, use_explore):
        """
        Infer an action with `state`.

        :param state:
        :param use_explore:
        :return: action value
        """
        predict_val = self.alg.predict(state)
        logit = predict_val[0]
        value = predict_val[1]
        action = predict_val[2]

        # update transition data

        # revised by ZZX *begin
        # for env_id in range(self.vector_env_size):
        #     self.sample_vector[env_id]["cur_state"].append(state[env_id])
        #     self.sample_vector[env_id]["logit"].append(logit[env_id])
        #     self.sample_vector[env_id]["action"].append(action[env_id])

        assert len(self.last_info) == len(state) == self.wait_num, print(
            '[GGLC] === {} {} {}'.format(len(self.last_info), len(state), self.wait_num))
        if self.wait_num != self.vector_env_size:
            for i in range(self.wait_num):
                env_id = self.last_info[i]['env_id']
                self.sample_vector[env_id]["cur_state"].append(state[i])
                self.sample_vector[env_id]["logit"].append(logit[i])
                self.sample_vector[env_id]["action"].append(action[i])
        else:
            for env_id in range(self.vector_env_size):
                self.sample_vector[env_id]["cur_state"].append(state[env_id])
                self.sample_vector[env_id]["logit"].append(logit[env_id])
                self.sample_vector[env_id]["action"].append(action[env_id])
        # revised by ZZX *end

        return action

    # added by ZZX
    def run_one_episode(self, use_explore, need_collect, lock=None, gid=-1, eid=-1):
        """
        Do interaction with max steps in each episode.

        :param use_explore:
        :param need_collect: if collect the total transition of each episode.
        :return:
        """
        # clear the old trajectory data
        self.clear_trajectory()
        state = self.env.get_init_state(self.id)

        for _ in range(self.max_step):
            _start = time()
            self.clear_transition()

            state, self.last_info = self.do_one_interaction(state, use_explore, lock, gid, eid, need_info=True)

            if need_collect:
                self.add_to_trajectory(self.transition_data)

            if self.transition_data["done"]:
                self.env.reset()
                state = self.env.get_init_state(self.id)

        traj = self.get_trajectory()
        return traj

    # added by ZZX
    def data_proc(self):
        episode_data = self.trajectory
        states = episode_data["cur_state"]

        actions = np.asarray(episode_data["real_action"])
        actions = np.eye(self.action_dim)[actions.reshape(-1)]

        pred_a = np.asarray(episode_data["action"])
        states.append(episode_data["last_state"])

        states = np.asarray(states)
        self.trajectory["cur_state"] = states
        self.trajectory["real_action"] = actions
        self.trajectory["action"] = pred_a

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        """Handle next state, reward and info."""

        # revise by ZZX *begin
        if self.vector_env_size != self.wait_num:
            for i in range(self.wait_num):
                env_id = info[i]['env_id']
                info[i].update({'eval_reward': reward[i]})
                self.reward_per_env[env_id] += reward[i]

                if info[i].get('real_done'):
                    self.reward_track.append(self.reward_per_env[env_id])
                    self.reward_per_env[env_id] = 0

            for i in range(self.wait_num):
                env_id = info[i]['env_id']
                self.sample_vector[env_id]['reward'].append(reward[i])
                self.sample_vector[env_id]['done'].append(done[i])
                self.sample_vector[env_id]['info'].append(info[i])

            return self.transition_data
        # revise by ZZX *end

        for env_id in range(self.vector_env_size):
            info[env_id].update({'eval_reward': reward[env_id]})
            self.reward_per_env[env_id] += reward[env_id]

            if info[env_id].get('real_done'):  # real done
                self.reward_track.append(self.reward_per_env[env_id])
                self.reward_per_env[env_id] = 0

        for env_id in range(self.vector_env_size):
            self.sample_vector[env_id]["reward"].append(reward[env_id])
            self.sample_vector[env_id]["done"].append(done[env_id])
            self.sample_vector[env_id]["info"].append(info[env_id])

        return self.transition_data

    def get_trajectory(self, last_pred=None):
        for env_id in range(self.vector_env_size):
            for _data_key in ("cur_state", "logit", "action", "reward", "done", "info"):
                self.trajectory[_data_key].extend(self.sample_vector[env_id][_data_key])

        # merge data into env_num * seq_len
        for _data_key in self.trajectory:
            self.trajectory[_data_key] = np.stack(self.trajectory[_data_key])

        self.trajectory["action"].astype(np.int32)

        trajectory = message(self.trajectory.copy())
        set_msg_info(trajectory, agent_id=self.id)
        return trajectory

    def sync_model(self):
        if not hasattr(self, "first_recv_weight"):
            logging.info("=================waiting model exp_{}==================".format(self.id))
            model_name = self.recv_explorer.recv(block=True)
            logging.info("=================received model exp_{}==================".format(self.id))
            setattr(self, "first_recv_weight", None)
            return model_name

        model_name = None
        self.sync_weights_count += 1
        if self.sync_weights_count >= self.broadcast_weights_interval:
            model_name = self.recv_explorer.recv(block=False)
            self.sync_weights_count = 0

        model_successor = model_name
        while model_successor:
            model_successor = self.recv_explorer.recv(block=False)
            if model_successor is not None:
                model_name = model_successor

        return model_name

    def sync_model_(self):
        """Block wait one [new] model when sync need."""
        model_name = None
        self.sync_weights_count += 1
        if self.sync_weights_count >= self.broadcast_weights_interval:
            model_name = self.recv_explorer.recv(block=True)
            self.sync_weights_count = 0

            model_successor = self.recv_explorer.recv(block=False)
            while model_successor:
                model_successor = self.recv_explorer.recv(block=False)
                sleep(0.002)

            if model_successor:
                print("getsuccessor: {}".format(model_successor))
                model_name = model_successor

        return model_name

    def reset(self):
        """Clear the sample_vector buffer."""
        self.sample_vector = dict()
        for env_id in range(self.vector_env_size):
            self.sample_vector[env_id] = defaultdict(list)

    def add_to_trajectory(self, transition_data):
        pass

    def do_one_interaction(self, raw_state, use_explore=True, lock=None, gid=-1, eid=-1, need_info=False):
        """
        Use the Agent do one interaction.

        User could re-write the infer_action and handle_env_feedback functions.
        :param raw_state:
        :param use_explore:
        :return:
        """
        _start0 = time()
        action = self.infer_action(raw_state, use_explore)
        self._stats.inference_time += time() - _start0
        if lock is not None:
            lock[gid].acquire()
        _start0 = time()

        next_raw_state, reward, done, info = self.env.step(action, self.id)
        if lock is not None:
            lock[gid].release()
            self.step = (self.step + 1) % 128
        self._stats.env_step_time += time() - _start0
        self._stats.iters += 1
        self.handle_env_feedback(next_raw_state, reward, done, info, use_explore)
        if need_info:
            return next_raw_state, info
        return next_raw_state
