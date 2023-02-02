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
"""Build base agent for PPO algorithm."""

from time import time
import numpy as np
from collections import defaultdict, deque

from xt.agent import Agent
from xt.agent.ppo.default_config import GAMMA, LAM
from zeus.common.util.register import Registers
from zeus.common.ipc.message import message, set_msg_info


@Registers.agent
class PPOLite(Agent):
    """Build base agent with PPO algorithm."""

    def __init__(self, env, alg, agent_config, **kwargs):
        # pipeline parameters
        self.vector_env_size = kwargs.pop("vector_env_size")
        self.wait_num = kwargs.pop("wait_num")

        super().__init__(env, alg, agent_config, **kwargs)
        # non-block parameters
        if hasattr(alg, "prefetch"):
            self.using_prefetch = alg.prefetch
        else:
            self.using_prefetch = False

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

    def infer_action(self, state, use_explore):

        predict_val = self.alg.predict(state)
        action = predict_val[0]
        logp = predict_val[1]
        value = predict_val[2]

        assert len(self.last_info) == len(state) == self.wait_num, print(
            '[GGLC] === {} {} {}'.format(len(self.last_info), len(state), self.wait_num))
        if self.wait_num != self.vector_env_size:
            for i in range(self.wait_num):
                env_id = self.last_info[i]['env_id']
                self.sample_vector[env_id]["cur_state"].append(state[i])
                self.sample_vector[env_id]["logp"].append(logp[i])
                self.sample_vector[env_id]["action"].append(action[i])
                self.sample_vector[env_id]["value"].append(value[i])
        else:
            for env_id in range(self.vector_env_size):
                self.sample_vector[env_id]["cur_state"].append(state[env_id])
                self.sample_vector[env_id]["logp"].append(logp[env_id])
                self.sample_vector[env_id]["action"].append(action[env_id])
                self.sample_vector[env_id]["value"].append(value[env_id])
        return action

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
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
        for env_id, value in enumerate(last_pred[2]):
            self.sample_vector[env_id]["value"].append(value)
            self.data_proc(self.sample_vector[env_id])
        
        
        
        for env_id in range(self.vector_env_size):
            for _data_key in self.sample_vector[0].keys():
                self.trajectory[_data_key].extend(
                    self.sample_vector[env_id][_data_key])

        # merge data into env_num * seq_len
        for _data_key in self.trajectory:
            self.trajectory[_data_key] = np.stack(self.trajectory[_data_key])

        self.trajectory["action"].astype(np.int32)
        

        trajectory = message(self.trajectory.copy())
        set_msg_info(trajectory, agent_id=self.id)
        return trajectory

    def data_proc(self,traj):
        """Process data."""
        # traj = self.trajectory
        state = np.asarray(traj['cur_state'])
        action = np.asarray(traj['action'])
        logp = np.asarray(traj['logp'])
        value = np.asarray(traj['value'])
        reward = np.asarray(traj['reward'])
        done = np.asarray(traj['done'])

        next_value = value[1:]
        value = value[:-1]

        done = np.expand_dims(done, axis=1)
        reward = np.expand_dims(reward, axis=1)
        discount = ~done * GAMMA
        delta_t = reward + discount * next_value - value
        adv = delta_t

        for j in range(len(adv) - 2, -1, -1):
            adv[j] += adv[j + 1] * discount[j] * LAM

        traj['cur_state'] = list(state)
        traj['action'] = list(action)
        traj['logp'] = list(logp)
        traj['adv'] = list(adv)
        traj['old_value'] = list(value)
        traj['target_value'] = list(adv + value)

        del traj['value']

    def run_one_episode(self, use_explore, need_collect, *args, **kwargs):
        """
        Do interaction with max steps in each episode.

        :param use_explore:
        :param need_collect: if collect the total transition of each episode.
        :return:
        """
        # clear the old trajectory data
        self.clear_trajectory()
        state = self.env.get_init_state(self.id)

        self._stats.reset()

        for _ in range(self.max_step):
            self.clear_transition()
            state = self.do_one_interaction(state, use_explore)

            if need_collect:
                self.add_to_trajectory(self.transition_data)

            if self.transition_data["done"]:
                if not self.keep_seq_len:
                    break
                self.env.reset()
                state = self.env.get_init_state()

        last_state = []
        for env_id in range(self.vector_env_size):
            last_state.append(self.sample_vector[env_id]["cur_state"][-1])
        
        last_state = np.stack(last_state)
        
        last_pred = self.alg.predict(last_state)
        
        return self.get_trajectory(last_pred)

    def do_one_interaction(self, raw_state, use_explore=True):
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

        _start1 = time()
        next_raw_state, reward, done, info = self.env.step(action, self.id)
        self._stats.env_step_time += time() - _start1
        self._stats.iters += 1

        self.handle_env_feedback(
            next_raw_state, reward, done, info, use_explore)
        return next_raw_state

    def add_to_trajectory(self, transition_data):
        pass
    
    def reset(self):
        """Clear the sample_vector buffer."""
        self.sample_vector = dict()
        for env_id in range(self.vector_env_size):
            self.sample_vector[env_id] = defaultdict(list)
