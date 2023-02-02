import random
import time
import envpool
import numpy as np
from typing import List, Tuple

from xt.environment.environment import Environment
from xt.environment.gym import infer_action_type
from xt.environment.gym.atari_wrappers import make_atari
from zeus.common.util.register import Registers


@Registers.env
class SinglePool(Environment):
    def init_env(self, env_info):
        # ZZX: repeat
        self.repeat = env_info.get('repeat', 0)
        print('[====] SinglePool created with repeat action prob = {}'.format(
            self.repeat))
        self._env = envpool.make(
            env_info['name'].replace('NoFrameskip-v4', '-v5'),
            env_type='gym',
            num_envs=1,
            frame_skip=4,
            episodic_life=True,
            stack_num=4,
            noop_max=30,
            seed=random.randint(0, 1000),
            repeat_action_probability=self.repeat,
        )
        self.dim = env_info.get('dim', 84)
        self.last_state = np.zeros((self.dim, self.dim, 4))
        self.init_state = None

        gym_env = make_atari(env_info)
        self.action_type = infer_action_type(gym_env.action_space)
        return gym_env

    def init_stack_obs(self, num):
        pass

    def reset(self):
        self.init_state = self.last_state
        return self.last_state

    def step(self, action: int, agent_index=0):
        obs, reward, done, info = self._env.step(np.array([action]))
        obs = obs.transpose(0, 2, 3, 1)
        _info = {'real_done': info['lives'][0] == 0,
                 'eval_reward': info['reward'][0], 'ale.lives': info['lives'][0]}
        self.last_state = obs[0]
        return obs[0], reward[0], done[0], _info


@Registers.env
class EnvPool(Environment):
    def init_env(self, env_info):
        print('[GGLC] EnvPool created')
        self.size = env_info.get("size")
        self.name = env_info.get("name").replace('NoFrameskip-v4', '-v5')
        self.batch_size = env_info.get("wait_num", self.size)
        self.env_start_core = env_info.get("env_start_core", -1)
        if self.env_start_core != -1:
            print('[GGLC]: ENV PIPELINE BINDING START FROM {}'.format(
                self.env_start_core))
        assert self.size is not None and self.name is not None, "envpool must assign 'name' and 'size'."

        self.pool = envpool.make(
            task_id=self.name,
            env_type='gym',
            num_envs=self.size,
            batch_size=self.batch_size,
            frame_skip=4,
            episodic_life=True,
            stack_num=4,
            noop_max=30,
            seed=random.randint(0, 10000),
            repeat_action_probability=.0,
            num_threads=self.size,
            # start id of binding thread. -1 means not to use thread affinity
            thread_affinity_offset=self.env_start_core
        )

        self.spec = envpool.make_spec(self.name)
        self.action_type = infer_action_type(self.spec.action_space)

        self.dim = env_info.get('dim', 84)
        # using np.float64 will result in longer training and inference time.
        self.last_state = [np.zeros(
            (self.dim, self.dim, 4), dtype=np.uint8) for _ in range(self.batch_size)]
        self.init_state = self.last_state

        self.lives = np.zeros(self.size)
        self.finished_env = None

    def reset(self):
        self.init_state = self.last_state
        return self.last_state

    def step(self, action, agent_index=0) -> Tuple[List[np.ndarray], list, list, List[dict]]:
        # note that the first step needs to specify an action for each env.
        if self.finished_env is None:
            if len(action) != self.size:
                # print('[GGLC] len(action)!=len(finished_env)... padding... ***')
                action = list(action)
                needed = self.size - len(action)
                for i in range(needed):
                    action.append(action[0])

        obs, rew, done, info = self.pool.step(
            np.array(action), self.finished_env)
        obs = obs.transpose(0, 2, 3, 1)

        _info = []
        for i, lives, env_id in zip(range(self.batch_size), info['lives'], info['env_id']):
            _info.append({
                'env_id': env_id,
                'real_done': lives == 0,
                'ale.lives': lives
            })
            if self.lives[env_id] > lives > 0:
                done[i] = True
            self.lives[env_id] = lives

        self.last_state = obs
        self.finished_env = info['env_id']

        return list(obs), list(rew), list(done), _info

    def get_env_info(self):
        self.reset()
        env_info = {
            "n_agents": 1,
            "api_type": 'standalone',
            "agent_ids": [0],
            "action_type": self.action_type
        }
        return env_info

    def close(self):
        self.pool.close()


@Registers.env
class GymEnvPool(Environment):
    def init_env(self, env_info):
        print('[GGLC] EnvPool created')
        self.size = env_info.get("size")
        self.name = env_info.get("name")
        self.batch_size = env_info.get("wait_num", self.size)
        self.env_start_core = env_info.get("env_start_core", -1)
        if self.env_start_core != -1:
            print('[GGLC]: ENV PIPELINE BINDING START FROM {}'.format(
                self.env_start_core))
        assert self.size is not None and self.name is not None, "envpool must assign 'name' and 'size'."

        self.pool = envpool.make(
            task_id=self.name,
            env_type='gym',
            num_envs=self.size,
            batch_size=self.batch_size,
            seed=random.randint(0, 10000),
            num_threads=self.size,
            # start id of binding thread. -1 means not to use thread affinity
            thread_affinity_offset=self.env_start_core
        )

        self.spec = envpool.make_spec(self.name)
        self.action_type = infer_action_type(self.spec.action_space)

        self.dim = env_info.get('dim', [4])
        # using np.float64 will result in longer training and inference time.

        self.last_state = np.zeros(
            shape=(self.batch_size, *self.dim), dtype=np.uint8)
        self.init_state = self.last_state

        self.lives = np.zeros(self.size)
        self.finished_env = None

    def reset(self):
        self.init_state = self.last_state
        return self.last_state

    def step(self, action, agent_index=0) -> Tuple[List[np.ndarray], list, list, List[dict]]:
        # note that the first step needs to specify an action for each env.
        if self.finished_env is None:
            if len(action) != self.size:
                # print('[GGLC] len(action)!=len(finished_env)... padding... ***')
                action = list(action)
                needed = self.size - len(action)
                for i in range(needed):
                    action.append(action[0])

        obs, rew, done, info = self.pool.step(
            np.array(action), self.finished_env)

        _info = []
        for i, env_id in zip(range(self.batch_size), info['env_id']):
            _info.append({
                'env_id': env_id,
                'real_done': done[i],
            })

        self.last_state = obs
        self.finished_env = info['env_id']

        return list(obs), list(rew), list(done), _info

    def get_env_info(self):
        self.reset()
        env_info = {
            "n_agents": 1,
            "api_type": 'standalone',
            "agent_ids": [0],
            "action_type": self.action_type
        }
        return env_info

    def close(self):
        self.pool.close()
