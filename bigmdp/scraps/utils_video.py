import os

import cv2
# from plotly.graph_objs import Scatter
from torchvision.utils import make_grid
import torch
from bigmdp.scraps.tmp_vi_helper import *
import math


def write_video(frames, title, path=''):
    #   frames = np.multiply(np.stack(frames, axis=0).transpose(0, 2, 3, 1), 255).clip(0, 255).astype(np.uint8)[:, :, :, ::-1]  # VideoWrite expects H x W x C in BGR
    _, H, W, _ = frames.shape
    writer = cv2.VideoWriter(os.path.join(path, '%s.mp4' % title), cv2.VideoWriter_fourcc(*'mp4v'), 30., (W, H), True)
    for frame in frames:
        writer.write(frame)
    writer.release()


def pad_all_frames(all_r_frames):
    max_len = max([a.shape[0] for a in all_r_frames])
    padded_frames = [np.zeros((max_len, all_r_frames[0].shape[1], all_r_frames[0].shape[2], all_r_frames[0].shape[3]),
                              dtype=np.uint8) for _ in range(len(all_r_frames))]

    for i in range(len(padded_frames)):
        f = all_r_frames[i]
        padded_frames[i][:f.shape[0], :, :, :] = f
        if (max_len > f.shape[0]):
            padded_frames[i][f.shape[0]:, :, :, :] = np.array([f[-1, :, :, :] for _ in range(max_len - f.shape[0])])

    p = np.array(padded_frames, dtype=np.uint8)
    return p

def stack_and_write_video(all_rollout_frames, base_path="./", title="default_name"):
    all_rollout_frames_ = [np.array(f) for f in all_rollout_frames]
    p = pad_all_frames(all_rollout_frames_)
    videos = torch.tensor(p).permute(0, 1, 4, 3, 2)
    no_of_frames = videos.shape[1]
    new_video = []
    row_n = int(math.sqrt(len(all_rollout_frames)))
    for i in range(no_of_frames):
        new_video.append(make_grid(videos[:, i, :, :, :], nrow=row_n).permute(2, 1, 0))
    new_video_ = np.array([t.numpy() for t in new_video])
    write_video(new_video_, title, path=base_path)
    return new_video_

