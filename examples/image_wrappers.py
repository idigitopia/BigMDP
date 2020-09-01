"""
basic wrappers, useful for reinforcement learning on gym envs
Most of the function are taken from the PyTorch Agent Net (PTAN) by Shmuma.
PTAN: https://github.com/Shmuma/ptan"""
# Mostly copy-pasted from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2


class MaxLenNotDoneWrap(gym.Wrapper):
    def __init__(self, env, max_episode_length):
        """
        add a max_len_reached flag in the info of the return, so that it can be handled differently tha
        n the real terminal signal.
        """
        gym.Wrapper.__init__(self, env)
        self.step_count = 0
        self.max_episode_length = max_episode_length

    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.step_count += 1
        ns,r,d,i = self.env.step(action)
        i.update({"max_episode_length_exceeded":1 if self.step_count>=self.max_episode_length else 0})
        return ns, r, d, i

class SingStartWrap(gym.Wrapper):
    def __init__(self, env, seed):
        """
        Sets the starting state to a single state using the seed provided. seeds every reset.
        """
        gym.Wrapper.__init__(self,env)
        self.seed = seed

    def reset(self, **kwargs):
        self.env.seed(self.seed)
        return self.env.reset(**kwargs)

class RepeatActionWrap(gym.Wrapper):
    def __init__(self, env, action_repeat):
        gym.Wrapper.__init__(self,env)
        self.action_repeat = action_repeat

    def step(self, action):
        reward_sum = 0
        for _ in range(self.action_repeat):
            ob, reward, done, info = self.env.step(action)
            reward_sum += reward
            if done:
                break
        return ob, reward_sum, done, info

class squeezeFrameWrap(gym.ObservationWrapper):
    def __init__(self, env):
        """squeezes the obersvations"""
        gym.Wrapper.__init__(self,env)
        self.observation_space = spaces.Box(low=0, high=255, shape=np.zeros(self.observation_space.shape).squeeze().shape, dtype=np.float32)
    
    def observation(self, observation):
        return observation.squeeze()


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not belive how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out
        
class StackFramesWrap(gym.Wrapper):
    def __init__(self, env, stack_count, skip_count = 1, zero_start=False, lazy_frames = False):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.skip_frame_count = skip_count
        self.out_k = stack_count
        self.frame_hist_count = self.skip_frame_count * self.out_k
        self.frames = deque([], maxlen= self.frame_hist_count)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.out_k, *shp), dtype=np.float32)
        self.zero_start = zero_start
        self.lazy_frames = lazy_frames

    def reset(self, **kwargs):
        ob = self.env.reset()
        ob = ob.squeeze()
        for _ in range(self.frame_hist_count):
            if self.zero_start:
                self.frames.append(np.zeros(ob.shape))
            else:
                self.frames.append(ob)
        self.frames.append(ob)
        return self._get_ob()


    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob = ob.squeeze()
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.frame_hist_count
        return_frames = list(self.frames)[-1::-self.skip_frame_count][::-1]
        return LazyFrames(return_frames) if self.lazy_frames else np.stack(return_frames)

class Img2TorchWrap(gym.ObservationWrapper):
    """
    Change image shape to CWH
    """
    def __init__(self, env):
        # super(Img2TorchWrap, self).__init__(env)
        gym.Wrapper.__init__(self, env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.swapaxes(np.swapaxes(observation, 2, 0), 1,2) 
    
    @staticmethod
    def torch2Img(obs):
        return np.swapaxes(np.swapaxes(obs, 1, 2), 0,2)


class RenderWrap(gym.ObservationWrapper):
    def __init__(self, env=None):
        # super(RenderWrap, self).__init__(env)
        gym.Wrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.env.observation_space.shape[:-1], dtype=np.uint8)

    def observation(self, obs):
        img = self.unwrapped.render("rgb_array")
        return self.to_GrayScale(img)

    def to_GrayScale(self,frame):
        frame = frame[:, :, 0] * 0.299 + frame[:, :, 1] * 0.587 + frame[:, :, 2] * 0.114 # GrayScale Image
        return frame.astype(np.uint8)

class ResizeWrap(gym.ObservationWrapper):
    def __init__(self, env, new_shape):
        # super(ResizeWrap, self).__init__(env)
        gym.Wrapper.__init__(self, env)
        self.new_shape = new_shape[:2]
        self.observation_space =  gym.spaces.Box(low=0.0, high=1.0, shape=new_shape,
                                                dtype=np.float32)

    def observation(self, frame):
        return cv2.resize(frame, self.new_shape, interpolation=cv2.INTER_LINEAR)


class CropWrap(gym.ObservationWrapper):
    """
    Change image shape to CWH
    """
    def __init__(self, env, h1=0, h2=84, w1=0, w2=84):
        # super(CropWrap, self).__init__(env)
        gym.Wrapper.__init__(self, env)
        self.h1, self.h2 = h1, h2
        self.w1, self.w2 = w1, w2

    def observation(self, frame_stack):
        return crop_frame(frame_stack, self.h1, self.h2, self.w1, self.w2)

def crop_frame(frame, h1=None, h2=None, w1=None, w2=None):
    """
    Crops the frame but maintains the size of the image
    """
    assert h1 is None or h2 is None or (h1<h2 or h2<0)  , "h1 much be smaller than h2, h1:{}, h2:{}".format(h1,h2)
    assert h2 is None or h2<=frame.shape[0], "Crop height must be smaller than frame height"
    assert w1 is None or w2 is None or (w1<w2 or w2<0), "w1 much be smaller than w2, w1:{}, w2:{}".format(w1,w2)
    assert w2 is None or w2<=frame.shape[1] ,"Crop width must be smaller than frame Width"
    original_shape = frame.shape
    return cv2.resize(frame[h1:h2, w1:w2], original_shape, interpolation=cv2.INTER_LINEAR)

def crop_stack(frame_stack, h1, h2, w1, w2):
    return np.stack([crop_frame(f, h1, h2, w1, w2) for f in frame_stack])

# def get_image_env(env, stack_count=4, skip_count = 1):
#     return c_FrameStack(c_ImageToPyTorch(RenderWrap(gym.make(env))), stack_count, skip_count)