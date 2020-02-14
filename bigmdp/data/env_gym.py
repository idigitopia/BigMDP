import gym
from gym_minigrid.wrappers import FullyObsWrapper
from enum import IntEnum
from gym import error, spaces, utils
from collections.abc import Iterable
from bigmdp.data.dataset import SimpleReplayBuffer
from matplotlib import pyplot as plt
import numpy as np
import logging

logger = logging.getLogger("mylogger")


class SimpleGymEnv:
    """
    Wrapper around Gym Environment
    Main Difference is that it returns a tuple of continious and discrete state rather than a single state.
    """

    def __init__(self, env_name: str, max_episode_length: int, action_repeat: int = 1,  seed = None, start_state=None, quantization_level = 1):
        self.env_name = env_name
        self._env = gym.make(env_name)
        self.seed = seed
        if self.seed is not None:
            self._env.seed(seed)
        self.max_episode_length = min(self._env._max_episode_steps, max_episode_length)
        self.action_repeat = action_repeat
        self.step_count = 0
        self.start_obs = start_state
        self.performance_as_steps = None
        self.quantization_level = quantization_level

        if (self._env._max_episode_steps < max_episode_length):
            logger.warn("Max episode length of {} over-rided by internal max_episode_steps of {}".format(max_episode_length,
                                                                                                         self._env._max_episode_steps))


    def set_state(self, s_c):
        self._env.env.state = s_c

    def reset(self) -> tuple:
        if self.seed is not None:
            self._env.seed(self.seed)
        self.step_count = 0  # Reset internal timer
        obs_c = self._env.reset()

        return obs_c

    def step(self, action):
        reward_k = 0
        for k in range(self.action_repeat):
            self.step_count += 1
            next_obs, reward, done, info = self._env.step(action)
            reward_k += reward
            done = done or self.step_count == self.max_episode_length
            if done:
                break

        info["max_episode_length_exceeded"] = bool(self.step_count >= self.max_episode_length)


        obs_c = next_obs

        return obs_c, reward_k, done, info

    def encode(self, obs_c: np.array) -> np.array:
        return np.array([round(i, self.quantization_level) for i in obs_c])

    def decode(self, obs_d: np.array) -> np.array:
        return obs_d

    def batch_encode(self, obs_cs: list) -> list:
        return [ self.encode(s) for s in np.array(obs_cs) ]

    def render(self, mode = 'rgb_array'):
        return self._env.render(mode = mode)

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return self._env.reset().shape[0]

    @property
    def action_size(self):
        return 1 #self._env.action_space.shape[0] # todo fix this hardcoded for cartpole

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return self._env.action_space.sample()

    def get_list_of_actions(self):
        return [i for i in range(self._env.action_space.n)]



class SimpleNormalizeEnv:
    """
        Wrapper around Gym Environment
        Main Difference is that it returns a tuple of continious and discrete state rather than a single state.
        """

    def __init__(self, env_name: str, max_episode_length: int, action_repeat: int = 1, seed=None, start_state=None,
                 quantization_level=1, multiply_state=False):
        import gym
        self.env_name = env_name
        self._env = gym.make(env_name)
        self.seed = seed
        if self.seed is not None:
            self._env.seed(self.seed)
        self.max_episode_length = min(self._env._max_episode_steps, max_episode_length)
        self.action_repeat = action_repeat
        self.step_count = 0
        self.start_obs = start_state
        self.performance_as_steps = None
        self.quantization_level = quantization_level
        self.normalizing_params = None
        self.reward_params = None
        self.set_normalizing_params()

        if (self._env._max_episode_steps < max_episode_length):
            logger.warn(
                "Max episode length of {} over-rided by internal max_episode_steps of {}".format(max_episode_length,
                                                                                                 self._env._max_episode_steps))

    def set_state(self, s_c, normalized=True):
        if normalized:
            self._env.env.state = self.state_denormalizing_func(s_c)
        else:
            self._env.env.state = s_c

    def reset(self) -> tuple:
        self.step_count = 0  # Reset internal timer
        return self._env.reset()

    def step(self, action):
        reward_k = 0
        for k in range(self.action_repeat):
            self.step_count += 1
            next_obs, reward, done, info = self._env.step(action)
            reward_k += reward
            done = done or self.step_count == self.max_episode_length
            if done:
                break

        info["max_episode_length_exceeded"] = bool(self.step_count >= self.max_episode_length)

        obs_c = self.state_normalizing_func(next_obs)

        return obs_c, reward_k, done, info

    def render(self, mode='rgb_array'):
        return self._env.render(mode=mode)

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return self._env.reset().shape[0]

    @property
    def action_size(self):
        return 1  # self._env.action_space.shape[0] # todo fix this hardcoded for cartpole

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return self._env.action_space.sample()

    def get_list_of_actions(self):
        return [i for i in range(self._env.action_space.n)]

    def state_normalizing_func(self, state):
        normalized_state = state
        for i in range(len(state)):
            max, min = self.normalizing_params[i]["max"], self.normalizing_params[i]["min"]
            normalized_state[i] = 2 * ((state[i] - min) / (max - min)) - 1

        return normalized_state

    def state_denormalizing_func(self, state):
        denormalized_state = state
        for i in range(len(state)):
            max, min = self.normalizing_params[i]["max"], self.normalizing_params[i]["min"]
            denormalized_state[i] = ((max - min) * (state[i] + 1) / 2) + min

        return denormalized_state

    def set_normalizing_params(self, policy=None):
        normalizing_params, reward_params = get_normalizing_params(SimpleGymEnv(env_name=self.env_name,
                                                                          max_episode_length=self.max_episode_length),
                                                                   policy=policy)
        self.normalizing_params = normalizing_params
        self.reward_params = reward_params
        print("normalizing_params_set")

class GymMiniGridEnv:

    def __init__(self, env='MiniGrid-Empty-5x5-v0', seed=4444, max_episode_length=1000, action_repeat=1):

        self.env_name = env
        self._env = FullyObsWrapper(gym.make(env))
        self._env.actions = GymMiniGridEnv.EmptyActions
        self._env.action_space = spaces.Discrete(len(self._env.actions))
        self._env.max_steps = max_episode_length

        self._env.seed(seed)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.step_count = 0

    class EmptyActions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

    def reset(self) -> tuple:
        self.step_count = 0  # Reset internal timer
        _ = self._env.reset()
        obs_c = self.get_observation()
        obs_d = self.encode(obs_c)
        return tuple((obs_c, obs_d))

    def is_starting_state(self, s):
        start_state = True
        for s_comp in s:
            if not (-0.05 < s_comp < 0.05):
                start_state = False
        return start_state

    def step(self, action):
        action = np.argmax(action)
        reward_k = 0
        for k in range(self.action_repeat):
            next_obs, reward, done, info = self._env.step(action)
            if self.is_starting_state(next_obs):
                reward = 10
            next_obs = self.get_observation()
            reward_k += reward if done else 0
            self.step_count += 1  # Increment internal timer
            done = done or self.step_count == self.max_episode_length
            if done:
                break

        obs_c = next_obs
        obs_d = self.encode(obs_c)

        info["max_episode_length_exceeded"] = bool(self.step_count == self.max_episode_length)

        return (obs_c, obs_d), reward_k, done, info

    def encode(self, obs_c: np.array) -> np.array:
        return np.array([round(i, 1) for i in obs_c])

    def decode(self, obs_d: np.array) -> np.array:
        return obs_d
        # pass

    def render(self, mode="render"):
        if mode == "plt":
            plt.imshow(self._env.render(mode="rgb_array", highlight=False))
        else:
            self._env.render()

    def get_observation(self):
        #         return np.array([*self._env.env.agent_pos, self._env.env.agent_dir])
        #         return np.array([*self._env.env.agent_pos, *self.get_oneHot(self._env.env.agent_dir,encoding_len = 4)])
        return np.array([*self.get_oneHot(self._env.env.agent_pos[0], encoding_len=5),
                         *self.get_oneHot(self._env.env.agent_pos[1], encoding_len=5),
                         *self.get_oneHot(self._env.env.agent_dir, encoding_len=4)])

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return len(self.reset()[0])

    @property
    def action_size(self):
        return self._env.action_space.n

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return self.get_oneHot(x=self._env.action_space.sample(),
                               encoding_len=self.action_size, )

    def get_list_of_actions(self):
        return [self.get_oneHot(x=i, encoding_len=self.action_size, )
                for i in range(self._env.action_space.n)]

    def get_oneHot(self, x, encoding_len, flatten=True):
        """
        Takes integer an converts it into one hot accordingly
        Or takes a list of integers and converts it into a list of one hot vectors
        if flatten is true , flattenst the list of vector to one vector
        """
        def get_single_one_hot(i, size):
            return np.eye(size)[i]

        if isinstance(x, Iterable):
            r = np.array([get_single_one_hot(i,encoding_len) for i in x])
            r = r.flatten() if flatten else r
            return r
        else:
            return get_single_one_hot(x,encoding_len)

def get_normalizing_params(env, buffer = None, policy = None):
    if policy is None:
        policy = lambda s:env.sample_random_action()

    if buffer is None:
        f_memory = SimpleReplayBuffer(200000)
        f_memory = collect_memory(env,
                                  memory=f_memory,
                                  policy_func=policy,
                                  episodes=1000,
                                  verbose=False)
    else:
        f_memory = buffer

    import math

    sample_ds, info = f_memory.simple_sample_all()
    s_, a_, r_, ns_, d_ = sample_ds
    all_state_components = []
    min_reward, max_reward = float('inf'), -float('inf')
    for i in range(len(s_)):
        s, a,r,ns,d = [d[i] for d in [ s_, a_, r_, ns_, d_]]
        all_state_components.append(s)
        all_state_components.append(ns)
        min_reward, max_reward = min(r, min_reward), max(r,max_reward)
    all_state_components = np.array(all_state_components)

    state_len = len(s)
    normalizing_params = [{} for _ in range(state_len)]

    for i in range(state_len):
        print(min(all_state_components[:, i]), max(all_state_components[:, i]))
        print(min(all_state_components[:, i]), (max(all_state_components[:, i])))
        normalizing_params[i]["min"] = min(all_state_components[:, i])
        normalizing_params[i]["max"] = max(all_state_components[:, i])

    print(normalizing_params)
    return normalizing_params, {"min":min_reward, "max":max_reward}


def collect_memory(env, memory, policy_func, episodes, verbose=False):
    """
    Collect training data from environment in the memory provided using the policy function
    :param self: Environment to be explored
    :param memory: Memory to append the expereinces in
    :param policy_func: policy for the exploration
    :param epoch_size: number of rollouts
    :return:
    """

    for i in range(episodes):
        s_c = env.reset()
        done = False
        step = 0
        while not done or step < env.max_episode_length:
            step += 1
            a = policy_func(s_c)
            s_prime_c, r, done, info = env.step(a)
            if (info["max_episode_length_exceeded"] == True):
                memory.add((s_c, a, r, s_prime_c, False))
            else:
                memory.add((s_c, a, r, s_prime_c, done))
            s_c = s_prime_c
            if (done):
                break
        if (verbose):
            print("====================================  Collectin Experience from True Environment: ", step)
    return memory
