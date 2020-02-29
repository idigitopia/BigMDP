import logging
import random
import time
from collections import deque
from copy import deepcopy as cpy
from timeit import timeit

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

# logger = logging.getLogger()
logger = logging.getLogger("mylogger")
OMIT_LIST = ["end_state", "unknown_state"]


# Define Replay Buffers
class BufferIterator:
    ''' Iterator class'''

    def __init__(self, buffer, batch_size):
        self._buffer = buffer
        self._batch_size = batch_size
        self._sample_index = 0
        self._sample_count = min(len(self._buffer), 500000)
        # self._sample_count = len(self._buffer)
        self._batch_count = int(self._sample_count / self._batch_size)
        self.sample_times = []
        # print("DEBUG MESSAGE: Iterator Batch Count:{}".format(self._batch_count))

    def __next__(self):
        ''' returns the next sample from the buffer sampling  '''
        ''' tensorize and send'''
        st = time.time()
        splitted_samples, info = self._buffer.sample(self._batch_size)
        ret_samples = splitted_samples #self._buffer.make_tensor(splitted_samples)
        self._sample_index += 1
        tt = time.time() - st
        self.sample_times.append(tt)

        if self._sample_index < self._batch_count:
            return ret_samples, info
        else:
            print("DEBUG MESSAGE: Iterator Stopped at Count:{}".format(self._sample_index))
            print("DEBUG MESSAGE: Average Sample Time:{}".format(sum(self.sample_times)/len(self.sample_times)))
            print("DEBUG MESSAGE: test sample time:{}".format(timeit(lambda: self._buffer.sample(32), number=250) / 250))

            raise StopIteration()


class SimpleReplayBuffer:
    def __init__(self, buffer_limit, batch_size=32):
        self.buffer = deque(maxlen=buffer_limit)
        self.priorities = deque(maxlen=buffer_limit) # Not used, just for consistency
        self.padded_info_buffer = deque(maxlen=buffer_limit)
        self._batch_size = batch_size
        self.more_efficient_sampling = None
        self.default_value = np.random.randint(10)
        self.buffer_limit = buffer_limit

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def add(self, transition, padded_info=None):
        # print(transition)
        self.buffer.append(transition)
        self.padded_info_buffer.append(padded_info or {})

    def sample(self, n):
        idxs = random.sample(range(len(self.buffer)), n)
        return self.sample_indices(idxs)

    def sample_indices(self, idxs):
        samples = np.array([self.buffer[i] for i in idxs])
        padded_infos = [self.padded_info_buffer[i] for i in idxs]
        info = {"sample_indices": idxs}
        info.update({k:[pi[k] for pi in padded_infos] for k in padded_infos[0].keys()})
        splitted_samples = [np.array(s) for s in zip(*samples)]
        assert all([i<len(self.buffer) for i in idxs])
        return splitted_samples, info

    def make_tensor(self, splitted_samples):
        split_tensors = (torch.FloatTensor(d) for d in splitted_samples)
        padded_tensors = (t.reshape(len(t), -1) for t in split_tensors)
        return padded_tensors

    def simple_sample_all(self):
        sample_indices = range(len(self.buffer))
        samples = np.array(self.buffer)[sample_indices]
        info = {"sample_indices": sample_indices}
        splitted_samples = [list(s) for s in zip(*samples)]
        return splitted_samples, info

    def size(self):
        return len(self.buffer)

    def reset_priorities(self):
        return

    def reset_padded_info_buffer(self):
        print("resetting padded_info_buffer")
        self.padded_info_buffer =  deque(maxlen=self.buffer_limit)
        for i in range(len(self.buffer)):
            self.padded_info_buffer.append({"qval":0})

    def __iter__(self):
        ''' Returns the iterator object '''
        return BufferIterator(self, self._batch_size)

    def __len__(self):
        return len(self.buffer)

def add_dataloaders(d1,d2):
    for t in d2.buffer:
        d1.add(t)
    return d1




class PrioritizedReplayBuffer:
    def __init__(self, buffer_limit, max_weight=100000, batch_size=32, offset = 1e-12, more_efficient_sampling = True):
        self.buffer = deque(maxlen=buffer_limit)
        self.priorities = deque(maxlen=buffer_limit)
        self.values = deque(maxlen=buffer_limit)
        self.padded_info_buffer = deque(maxlen=buffer_limit)
        self.max_weight = 1000000
        self._batch_size = batch_size
        self._offset = offset
        self.more_efficient_sampling = more_efficient_sampling
        self.buffer_limit = buffer_limit
        self.default_value = np.random.randint(10)



    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def size(self):
        return len(self.buffer)

    def add(self, experience, padded_info=None):
        # print(transition)
        self.buffer.append(experience)
        self.padded_info_buffer.append(padded_info or {})
        self.priorities.append(self.max_weight)
        self.values.append(self.default_value)
        assert len(self.buffer) == len(self.priorities)
        # print(len(self.buffer))

    def sample(self, batch_size):
        if self.more_efficient_sampling and len(self.priorities) > 5001:
            sample_start_index = np.random.randint(len(self.priorities) - 5000)
            base_sample_indices = list(
                WeightedRandomSampler(list(self.priorities)[sample_start_index:sample_start_index + 5000], batch_size,
                                      replacement=True))
            idxs = [i + sample_start_index for i in base_sample_indices]
        else:
            idxs = list(WeightedRandomSampler(self.priorities, batch_size, replacement=False))
        return self.sample_indices(idxs)

    def sample_indices(self, idxs):
        samples = np.array([self.buffer[i] for i in idxs])
        padded_infos = [self.padded_info_buffer[i] for i in idxs]
        info = {"sample_indices": idxs}
        info.update({k: [pi[k] for pi in padded_infos] for k in padded_infos[0].keys()})
        splitted_samples = [np.array(s) for s in zip(*samples)]
        assert all([i < len(self.buffer) for i in idxs])
        return splitted_samples, info

    def make_tensor(self, splitted_samples):
        split_tensors = (torch.FloatTensor(d) for d in splitted_samples)
        padded_tensors = (t.reshape(len(t), -1) for t in split_tensors)
        return padded_tensors

    def simple_sample_all(self):
        sample_indices = range(len(self.buffer))
        samples = np.array(self.buffer)[sample_indices]
        info = {"sample_indices": sample_indices}
        splitted_samples = [list(s) for s in zip(*samples)]
        return splitted_samples, info

    def set_priorities(self, indices, errors):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + self._offset

    def reset_priorities(self):
        print("resetting priorities, more efficient ssampling:{}".format(self.more_efficient_sampling))
        self.priorities =  deque(maxlen=self.buffer_limit)
        for i in range(len(self.buffer)):
            self.priorities.append(self.max_weight)

    def reset_padded_info_buffer(self):
        print("resetting padded_info_buffer")
        self.padded_info_buffer =  deque(maxlen=self.buffer_limit)
        for i in range(len(self.buffer)):
            self.padded_info_buffer.append({"qval":0})

    def set_custom_priorities(self, calc_prty_func, indices = None):
        """

        :param calc_prty_func: takes an input i.e the sample
        indxs: if none set priorities for all .
        :return:
        """
        indices = indices or range(len(self.buffer))
        samples, info = self.sample_indexes(indices)
        new_priorities = [calc_prty_func(s) for s in samples]
        for i, p in zip(indices, new_priorities):
            self.priorities[i] = p

    def __iter__(self):
        ''' Returns the iterator object '''
        return BufferIterator(self, self._batch_size)

    def __len__(self):
        return len(self.buffer)


from collections.abc import Iterable
from itertools import zip_longest

def gather_dataset_in_buffer_v2(exp_buffer, env, episodes, render, policies, policy_is_discrete=False):
    "can get multiple polices"
    "episodes can also be iterable"
    assert isinstance(policies, Iterable)
    assert isinstance(episodes, Iterable) and len(episodes)<=len(policies)

    _exp_buffer = exp_buffer
    eps_plcy_list = zip_longest(episodes, policies, fillvalue=episodes[-1] if isinstance(episodes, Iterable) else episodes)

    for episodes, policy in eps_plcy_list:
        _exp_buffer, info = gather_data_in_buffer(_exp_buffer, env, episodes, render, policy, policy_is_discrete)

    return _exp_buffer


def gather_data_in_buffer(exp_buffer, env, episodes, render, policy, frame_count = None, pad_attribute_fxn = None):

    # experience = obs, action, next_obs, reward, terminal_flag
    experiences = []

    rewards = 0
    frame_counter = 0
    eps_count = 0
    for _ in range(episodes):
        eps_count+=1
        done = False
        obs_c = env.reset()  # continious observation and discrete observation
        ep_reward = 0
        while not done:
            frame_counter+=1
            if render:
                env.render()

            action = policy(obs_c)
            next_obs_c, reward, done, info = env.step(action)

            # todo make an option for storing discrete expereinces as well.
            # todo make an option for one hot encoding of action
            # action_one_hot = [0 for _ in range(env.action_space.n)]
            # action_one_hot[action] = 1

            # Omit max_episode_length experience from terminal experience bucker.
            _done = False if done and "max_episode_length_exceeded" in info and info["max_episode_length_exceeded"] == True else done

            # add to buffer
            exp = [obs_c.tolist(), [action], next_obs_c.tolist(), [reward], [_done]]
            if pad_attribute_fxn is not None:
                exp_buffer.add(exp,padded_info={k:f(obs_c) for k , f in pad_attribute_fxn.items()})
            else:
                exp_buffer.add(exp)

            ep_reward += reward

            def populate_mdp_with_model(mdp, step_fxn, net, iterations=1):
                for i in range(iterations):
                    for hs in list(mdp.tD.keys()):
                        if hs in OMIT_LIST:
                            continue
                        for a in mdp.A:
                            hns, r, d, i = step_fxn(hs, a, net)
                            mdp.consume_transition((hs, a, hns, r, d))
                return mdp
            obs_c = next_obs_c
        rewards += ep_reward

        if  frame_count and frame_counter >  frame_count:
            break

    print('Average Reward of collected trajectories:{}'.format(round(rewards / eps_count, 3)))
    logger.info('Average Reward of collected trajectories:{}'.format(round(rewards / eps_count, 3)))

    info = {}
    return exp_buffer, info


def reset_and_return_copy(buffer):
    _buffer = cpy(buffer)
    _buffer.reset_priorities()
    return _buffer


