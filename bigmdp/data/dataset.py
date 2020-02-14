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
        ret_samples = self._buffer.make_tensor(splitted_samples)
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
        self.values = deque(maxlen=buffer_limit)
        self._batch_size = batch_size
        self.more_efficient_sampling = None
        self.default_value = np.random.randint(10)

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def add(self, transition):
        # print(transition)
        self.buffer.append(transition)
        self.values.append(self.default_value)


    def sample(self, n):
        sample_indices = random.sample(range(len(self.buffer)), n)
        samples = np.array([self.buffer[i] for i in sample_indices])
        info = {"sample_indices": sample_indices}
        splitted_samples = [list(s) for s in zip(*samples)]
        assert all([i<len(self.buffer) for i in sample_indices])
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

    def __iter__(self):
        ''' Returns the iterator object '''
        return BufferIterator(self, self._batch_size)

    def __len__(self):
        return len(self.buffer)



def add_dataloaders(d1,d2):
    for t in d2.buffer:
        d1.add(t)
    return d1
