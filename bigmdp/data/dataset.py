from collections.abc import Iterable
from itertools import zip_longest
import pickle as pk
import gzip
import logging
import os
import random
import time
from collections import deque
from copy import deepcopy as cpy

import bigmdp.utils.utils_directory as dh
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
        self._sample_count = len(self._buffer)
        self._batch_count = int(self._sample_count / self._batch_size)
        self.sample_times = []
        # print("DEBUG MESSAGE: Iterator Batch Count:{}".format(self._batch_count))

    def __next__(self):
        ''' returns the next sample from the buffer sampling  '''
        ''' tensorize and send'''
        st = time.time()
        splitted_samples, info = self._buffer.sample(self._batch_size)
        ret_samples = splitted_samples  # self._buffer.make_tensor(splitted_samples)
        self._sample_index += 1
        tt = time.time() - st
        self.sample_times.append(tt)

        if self._sample_index < self._batch_count:
            return ret_samples, info
        else:
            # print("DEBUG MESSAGE: Iterator Stopped at Count:{}".format(self._sample_index))
            # print("DEBUG MESSAGE: Average Sample Time:{}".format(sum(self.sample_times)/len(self.sample_times)))
            # print("DEBUG MESSAGE: test sample time:{}".format(timeit(lambda: self._buffer.sample(32), number=250) / 250))
            raise StopIteration()


class SimpleReplayBuffer:
    def __init__(self, buffer_limit, batch_size=32, priority_flag=False, max_weight=100000, offset=1e-12, \
                 efficient_flag=True, lazy_frames=False):
        self.buffer_limit = buffer_limit
        self.buffer = deque(maxlen=buffer_limit)
        self.priorities = deque(maxlen=buffer_limit)
        self.padded_info_buffer = deque(maxlen=buffer_limit)
        self.suffix_ = 0

        # sampling params
        self._batch_size = batch_size
        self.priority_flag = priority_flag
        self.efficient_flag = efficient_flag
        self.max_weight = max_weight
        self._offset = offset
        self.lazy_frames = lazy_frames

    def __iter__(self):
        ''' Returns the iterator object '''
        return BufferIterator(self, self._batch_size)

    def __len__(self):
        return len(self.buffer)

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def size(self):
        return len(self.buffer)

    def add(self, experience, padded_info=None, priority_weight=None):
        # print(transition)
        self.buffer.append(experience)
        self.padded_info_buffer.append(padded_info or {})
        if priority_weight is not None:
            self.priorities.append(priority_weight)
        else:
            self.priorities.append(self.max_weight)
        assert len(self.buffer) == len(self.priorities)

    def sample(self, batch_size):
        if self.priority_flag:
            if self.efficient_flag and len(self.priorities) > 10001:
                sample_start_index = max(0, np.random.randint(len(self.priorities) - 1000) - 5000)
                base_sample_indices = list(
                    WeightedRandomSampler(list(self.priorities)[sample_start_index:sample_start_index + 10000],
                                          batch_size,
                                          replacement=True))
                idxs = [i + sample_start_index for i in base_sample_indices]
            else:
                idxs = list(WeightedRandomSampler(self.priorities, batch_siomze, replacement=False))
        else:
            idxs = random.sample(range(len(self.buffer)), batch_size)
        return self.sample_indices(idxs)

    def sample_indices(self, idxs):
        samples = [(np.array(item_) for item_ in self.buffer[i]) for i in idxs]
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
        print("resetting priorities, more efficient ssampling:{}".format(self.efficient_flag))
        self.priorities = deque(maxlen=self.buffer_limit)
        for i in range(len(self.buffer)):
            self.priorities.append(self.max_weight)

    def reset_padded_info_buffer(self, padded_info=None):
        print("resetting padded_info_buffer")
        self.padded_info_buffer = deque(maxlen=self.buffer_limit)
        for i in range(len(self.buffer)):
            self.padded_info_buffer.append(padded_info or {})

    def reset_buffer(self):
        self.buffer = deque(maxlen=self.buffer_limit)
        self.priorities = deque(maxlen=self.buffer_limit)  # Not used, just for consistency
        self.padded_info_buffer = deque(maxlen=self.buffer_limit)

    # def set_custom_priorities(self, calc_prty_func, indices=None):
    #     """
    #     :param calc_prty_func: takes an input i.e the sample
    #     indxs: if none set priorities for all .
    #     :return:
    #     """
    #     indices = indices or range(len(self.buffer))
    #     samples, info = self.sample_indexes(indices)
    #     new_priorities = [calc_prty_func(s) for s in samples]
    #     for i, p in zip(indices, new_priorities):
    #         self.priorities[i] = p

    def get_episodes_start_idx(self, max_episode_len=250):
        start_idxs = [0]
        c_ = 0

        for i, transition in self.buffer[:-1]:
            s, a, ns, r, d = transition
            c_ += 1
            if d[0] or c_ == max_episode_len:  # todo there will be a bug if end of episode is start of episode
                start_idxs.append(i + 1)
                c_ = 0
        return start_idxs

    def split_buffer(self, split_ratio=0.3, copy = False, split_on_episodes = False, max_episdoe_len= None):
        if split_on_episodes:
            assert max_episdoe_len is not None

        empty_buffer = SimpleReplayBuffer(buffer_limit=self.buffer_limit,
                                          batch_size=self._batch_size,
                                          priority_flag=self.priority_flag,
                                          max_weight=self.max_weight,
                                          offset=self._offset,
                                          efficient_flag=self.efficient_flag)

        train_buffer, test_buffer = cpy(empty_buffer), cpy(empty_buffer)

        assert len(self.buffer) == len(self.padded_info_buffer)
        tmp_buffer = cpy(self) if copy else self

        put_in_train , c_ = split_ratio == 0, 0 # put in train if test_size is 0 else put in test
        while tmp_buffer.buffer:
            c_ += 1
            data_point = tmp_buffer.buffer.popleft()
            data_info = tmp_buffer.padded_info_buffer.popleft()
            d = data_point[-1][0]
            if put_in_train:
                train_buffer.add(data_point,padded_info=data_info)
            else:
                test_buffer.add(data_point,padded_info=data_info)

            if split_on_episodes:
               if d or c_ == max_episdoe_len:
                   put_in_train = random.random() > split_ratio
            else:
                put_in_train = random.random() > split_ratio

        return train_buffer, test_buffer


    def _save_to_cache(self, save_dir, data_suffix=None, format_str="$store$_{}_ckpt_{}.pth",
                       overwrite=True):
        if data_suffix is None:
            self.suffix_ += 1
            suffix_ = self.suffix_
        else:
            suffix_ = data_suffix
        print("Data Dump Started For suffix: {}".format(suffix_))


        dh.create_hierarchy(save_dir)
        slices = ["observation", "action", "next_observation", "reward", "done"]

        if self.lazy_frames:
            main_buffer_file_name = save_dir + "/" + format_str.format("mainBuffer", suffix_)
            pk.dump(self.buffer, open(main_buffer_file_name, "wb"))
        else:
            # Save the Data now
            for i, array_name in enumerate(slices):
                file_name = save_dir + "/" + format_str.format(array_name, suffix_)
                to_save_array = np.array([tran[i] for tran in self.buffer])
                if not overwrite and os.path.exists(file_name):
                    print("skipping {}. File already exists".format(file_name))
                    continue
                with open(file_name, 'wb') as f:
                    with gzip.GzipFile(fileobj=f) as outfile:
                        np.save(outfile, to_save_array, allow_pickle=False)

        padded_info_file_name = save_dir + "/" + format_str.format("paddedInfo", suffix_)
        pk.dump(self.padded_info_buffer, open(padded_info_file_name, "wb"))
        print("Data Dump Complete for suffix: {}".format(suffix_))

    def _fetch_from_cache(self, load_dir, data_suffix, format_str="$store$_{}_ckpt_{}.pth"):
        slices = ["observation", "action", "next_observation", "reward", "done"]
        loaded_arrays = {}

        if self.lazy_frames:
            main_buffer_file_name = save_dir + "/" + format_str.format("mainBuffer", suffix_)
            main_buffer = pk.load(open(main_buffer_file_name, "rb"))

            padded_info_file_name = load_dir + "/" + format_str.format("paddedInfo", data_suffix)
            if os.path.exists(padded_info_file_name):
                padded_info_buffer = pk.load(open(padded_info_file_name, "rb"))
            else:
                padded_info_buffer = [{} for _ in range(len(main_buffer))]

            print("Parse Complete for suffix: {}".format(data_suffix))
            for transition, padded_info in zip(main_buffer, padded_info_buffer):
                self.add(transition, padded_info=padded_info)
            print("Data Load Complete for suffix: {}".format(data_suffix))


        else:
            # load the data
            for array_name in slices:
                file_name = load_dir + "/" + format_str.format(array_name, data_suffix)
                if not os.path.exists(file_name):
                    print(file_name, "Does not exist. Skipping . . . . ")
                    return None
                with open(file_name, 'rb') as f:
                    with gzip.GzipFile(fileobj=f) as infile:
                        loaded_arrays[array_name] = np.load(infile, allow_pickle=False)

            padded_info_file_name = load_dir + "/" + format_str.format("paddedInfo", data_suffix)
            if os.path.exists(padded_info_file_name):
                padded_info_buffer = pk.load(open(padded_info_file_name, "rb"))
            else:
                padded_info_buffer = [{} for _ in range(len(loaded_arrays["observation"]))]

            print("Parse Complete for suffix: {}".format(data_suffix))
            all_data =[loaded_arrays[arr_name] for arr_name in slices] + [padded_info_buffer]
            for obs, a, obs_prime, r, d, padded_info in zip(*all_data):
                self.add([obs, a, obs_prime, r, d], padded_info=padded_info)
            print("Data Load Complete for suffix: {}".format(data_suffix))




def gather_dataset_in_buffer_v2(exp_buffer, env, episodes, render, policies, policy_is_discrete=False):
    "can get multiple polices"
    "episodes can also be iterable"
    assert isinstance(policies, Iterable)
    assert isinstance(episodes, Iterable) and len(episodes) <= len(policies)

    _exp_buffer = exp_buffer
    eps_plcy_list = zip_longest(episodes, policies,
                                fillvalue=episodes[-1] if isinstance(episodes, Iterable) else episodes)

    for episodes, policy in eps_plcy_list:
        _exp_buffer, info = gather_data_in_buffer(_exp_buffer, env, episodes, render, policy, policy_is_discrete)

    return _exp_buffer


def gather_data_in_buffer(exp_buffer, env, episodes, render, policy, frame_count=None, pad_attribute_fxn=None,
                          verbose=False, policy_on_states= False, pad_with_state = False, show_elapsed_time = True,):
    """

    :param exp_buffer:
    :param env:
    :param episodes:
    :param render:
    :param policy:
    :param frame_count:
    :param pad_attribute_fxn:
    :param verbose:
    :param policy_on_states:  if set to true , the policy provided is assumed to be on the state variable of unwrapped env
    :return:
    """
    # experience = obs, action, next_obs, reward, terminal_flag
    start_time = time.time()
    experiences = []

    rewards = 0
    frame_counter = 0
    eps_count = 0
    all_rewards = []
    for _ in range(episodes):
        eps_count += 1
        done = False
        obs_c = env.reset()  # continious observation and discrete observation
        prev_internal_state = env.unwrapped.state
        ep_reward = 0
        while not done:
            frame_counter += 1
            if render:
                env.render()

            action =policy(env.unwrapped.state) if policy_on_states else policy(obs_c)
            next_obs_c, reward, done, info = env.step(action)
            internal_state = env.unwrapped.state
            # todo make an option for storing discrete expereinces as well.
            # todo make an option for one hot encoding of action
            # action_one_hot = [0 for _ in range(env.action_space.n)]
            # action_one_hot[action] = 1

            # Omit max_episode_length experience from terminal experience bucker.
            if "max_episode_length_exceeded" in info and info[ "max_episode_length_exceeded"]:
                continue

            _done = False if done and "max_episode_length_exceeded" in info and info[
                "max_episode_length_exceeded"] == True else done

            # add to buffer
            if exp_buffer.lazy_frames:
                exp = [obs_c, [action], next_obs_c, [reward], [_done]]
            else:
                exp = [obs_c.tolist(), [action], next_obs_c.tolist(), [reward], [_done]]

            if pad_attribute_fxn is not None:
                exp_buffer.add(exp, padded_info={k: f(obs_c) for k, f in pad_attribute_fxn.items()})
            elif pad_with_state:
                exp_buffer.add(exp, padded_info = {"state": prev_internal_state, "next_state": internal_state})
            else:
                exp_buffer.add(exp)
            prev_internal_state = internal_state
            ep_reward += reward

            obs_c = next_obs_c
        all_rewards.append(ep_reward)
        rewards += ep_reward
        if verbose:
            print(ep_reward, frame_count)

        if frame_count and frame_counter > frame_count:
            break

    print('Average Reward of collected trajectories:{}'.format(round(rewards / eps_count, 3)))
    logger.info('Average Reward of collected trajectories:{}'.format(round(rewards / eps_count, 3)))

    info = {"all_rewards":all_rewards}
    if show_elapsed_time:
        print("Data Collection Complete in {} Seconds".format(time.time()-start_time))
    return exp_buffer, info


def reset_and_return_copy(buffer):
    _buffer = cpy(buffer)
    _buffer.reset_priorities()
    return _buffer
