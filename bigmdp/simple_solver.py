import time

import argparse
from deeprmax.vi.VI_helper_funcs import *
from deeprmax.data.dataset import SimpleReplayBuffer, PrioritizedReplayBuffer
from deeprmax.data.env_gym import SimpleNormalizeEnv
from deeprmax.utils.utils_log import *
from deeprmax.utils.utils_video import *
from async_vi.utils.tmp_vi_helper import *
from async_vi.simple_worker import *
from async_vi.utils.image_wrappers import *
from async_vi.utils.hyper_params import HYPERPARAMS
from deeprmax.utils.utils_directory import *
# from async_vi.MDP import *
import numpy as np

from async_vi.MDP_GPU import FullMDP

# Get all Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--bottle_neck_size", help="size", type=int, default=32)
parser.add_argument("--discrete_bn", help="size", type=int, default=0)
parser.add_argument("--multiplyer", help="multiplyer of feature space", type=int, default=10)
parser.add_argument("--env", help="environment name", type=str, default="CartPole")
parser.add_argument("--name", help="Experiment name", type=str, default="CartPoleR1")
parser.add_argument("--load", help="Load the previous MDP ?", type=int, default=0)
parser.add_argument("--symbolic", help="Use Symbolic env if 1 else use image based env", type=int, default=1)
parser.add_argument("--steps_to_train", help="Number of steps to train the whole pipeline", type=int, default=0)
parser.add_argument("--rmax", help="Use rmax exploration?", type=int, default=0)
parser.add_argument("--video_every", help="get a rollout video every", type=int, default= 999999999)


# args = parser.parse_args()
args = parser.parse_args("--video_every 10 --env MountainCar --rmax 1 --multiplyer 10".split(" "))
base_file_path = "./result_dump/{}/{}".format(args.env, args.name +
                                              "_bn-" + str(args.bottle_neck_size) +
                                              "_sym-" + str(args.symbolic))

create_hierarchy(base_file_path)

def latent_to_hs_disc_fxn(s):
    if len(s)==1:
        return hAsh(s[0] * args.multiplyer)
    else:
        assert False

# log_dirs_dict, loggers_dict = get_advanced_log_dir_and_logger(ROOT_FOLDER = "Symbolic" if args.symbolic else "Image",
#                                                              EXP_ID = args.env,
#                                                              EXP_PARAMS="_bn-" + str(args.bottle_neck_size) +
#                                                                         "_sym-" + str(args.symbolic) +
#                                                                         "_rmax-" + str(bool(args.rmax)),
#                                                              tb_log_keys=["tb_train_logger", "tb_vi_logger"])


if args.symbolic:
    params = HYPERPARAMS[args.env + "-sym"]
#     env = SimpleNormalizeEnv(params["env_name"], max_episode_length=params["max_episode_length"])
    env = torch.load("mountain_car_env.pk")
else:
    print("Not Implemented Yet")  # Todo
    params = HYPERPARAMS[args.env + "-img"]
    assert False

if args.load:
    mdp = torch.load(base_file_path + "mdp_class.pth")
else:
    mdp = FullMDP(A=env.get_list_of_actions(),
                  ur=params["unknown_transition_reward"],
                  vi_params={"gamma": params["gamma"],
                             "slip_prob": params["slip_probability"],
                             "rmax_reward": params["rmax_reward"],
                             "rmax_thres": params["rmax_thres"],
                             "balanced_explr": True},
                  MAX_S_COUNT=int(5e6))

all_rewards = []
eval_rewards, safe_eval_rewards = [], []
policy_fetch_time = []
bellman_backup_time = [9999]
tran_buffer = SimpleReplayBuffer(int(params["replay_size"]))
eps_tracker = EpsilonTracker(params)

frame_count = 0
warmup_eps = 10
eval_reward = 0

safe_policy = lambda s : mdp.get_safe_action(latent_to_hs_disc_fxn([s]))
opt_policy =  lambda s : mdp.get_opt_action(latent_to_hs_disc_fxn([s]))
explr_policy =  lambda s : mdp.get_explr_action(latent_to_hs_disc_fxn([s]))
random_policy = lambda s: env.sample_random_action()

for eps in range(100000):
    s = env.reset()
    running_reward = 0

    while True:
        frame_count += 1

        if frame_count > params["replay_initial"] and frame_count % 20 == 0:
            st = time.time()
            for i in range(10):
                mdp.do_optimal_backup(mode="GPU", n_backups=1)
                mdp.do_explr_backup(mode="GPU", n_backups=1)
                mdp.do_safe_backup(mode="GPU", n_backups=1)
            bellman_backup_time.append(time.time() - st)

        st = time.time()
        if frame_count < (params["replay_initial"] + 100) or (
                np.random.random() < eps_tracker.get_eps(frame_count) and not args.rmax):
            a = random_policy(s)
        else:
            a = explr_policy(s) if args.rmax else opt_policy(s)
        policy_fetch_time.append(time.time() - st)


        ns, r, d, i = env.step(a)
        _d = False if d and i["max_episode_length_exceeded"] == True else d

        hs_d, hns_d = latent_to_hs_disc_fxn([s]), latent_to_hs_disc_fxn([ns])
        #         shared_store_1.add_to_transition_queue.remote([hs_d, a, hns_d, r, _d])
        mdp.consume_transition([hs_d, a, hns_d, r, _d])
        tran_buffer.add([s, a, ns, r, _d])

        running_reward += r
        s = ns

        if frame_count > params["replay_initial"] and eps % 10 == 0:
            env.render()

        if frame_count % params['checkpoint_every'] == 0:
            cache_mdp = mdp
            torch.save(cache_mdp, base_file_path + "mdp_class.pth")

        if d:
            time.sleep(3)  # Just cutting the VI Solver some Slack
            print("-====-")
            break

    all_rewards.append(running_reward)

    if (eps > 10):
        break

    #### Evaluation Code
    if frame_count > params["replay_initial"] and frame_count > params['checkpoint_every']:
        cache_mdp = mdp
        rmax_count = sum([1 for s in cache_mdp.tC for a in cache_mdp.tC[s] if sum(cache_mdp.tC[s][a].values()) < 10])
        #         vi_error, e_vi_error, s_vi_error = ray.get(shared_store_1.get_curr_vi_errors.remote())
        vi_error, e_vi_error, s_vi_error = [mdp.curr_vi_error, mdp.e_curr_vi_error, mdp.s_curr_vi_error]

        eval_reward = evaluate_on_env(env, opt_policy, eps_count=2, render=False)[0]
        eval_rewards.append(eval_reward)

        safe_eval_reward = evaluate_on_env(env, safe_policy, eps_count=2, render=False)[0]
        safe_eval_rewards.append(safe_eval_reward)

        #         loggers_dict["tb_train_logger"].add_scalar('Safe policy performance', float(safe_eval_reward), eps)
        #         loggers_dict["tb_train_logger"].add_scalar('Optimal policy performance', float(eval_reward), eps)
        #         loggers_dict["tb_train_logger"].add_scalar('Expl/Expt policy performance', float(running_reward), eps)

        #         loggers_dict["tb_train_logger"].add_scalar('MDP State Count', float(len(cache_mdp.vD)), eps)
        #         loggers_dict["tb_train_logger"].add_scalar('Rmax Count', float(rmax_count), eps)
        #         loggers_dict["tb_train_logger"].add_histogram('Optimal Value Distr', torch.tensor(list(cache_mdp.vD.values())), eps)
        #         loggers_dict["tb_train_logger"].add_histogram('Optimal Policy Distr', torch.tensor(list(cache_mdp.pD.values())), eps)

        #         loggers_dict["tb_train_logger"].add_scalar('Expl VI_error', float(e_vi_error), eps)
        #         loggers_dict["tb_train_logger"].add_scalar('Opt VI_error', float(vi_error), eps)
        #         loggers_dict["tb_train_logger"].add_scalar('Safe VI_error', float(s_vi_error), eps)

        #         loggers_dict["tb_train_logger"].add_scalar('Explr Epsilon', float(eps_tracker.get_eps(frame_count)), eps)
        #         loggers_dict["tb_train_logger"].add_scalar('Policy Fetch rate', float(mean(policy_fetch_time)), eps)

        print("episode:", eps,
              "reward:", running_reward,
              #               "Bkp error:", [round(d,6) for d in ray.get(shared_store_1.get_curr_vi_errors.remote())],
              "Bkp error:", [round(d, 6) for d in [mdp.curr_vi_error, mdp.e_curr_vi_error, mdp.s_curr_vi_error]],
              "s8 visited:", len(mdp.tC),
              "Rmax S8 Count", rmax_count,
              "epsilon", round(eps_tracker.get_eps(frame_count), 2),
              "Eval Reward", eval_reward,
              "policy_fetch_rate", round(mean(policy_fetch_time), 6),
              "bellman_backup_time", round(mean(bellman_backup_time), 6)
              )

        policy_fetch_time = []
        bellman_backup_time = []

#     if eps%args.video_every == 0 and eps != 0 and frame_count>params['checkpoint_every']:
#         nn_performance, info_ , video = rollout_with_nn_behavior(env = env,
#                                                                  policy = explr_policy if args.rmax else opt_policy,
# #                                                                  hs_nn_fxn = lambda hs: ray.get(shared_store_1._get_nn_hs.remote(hs)),250
#                                                                  hs_nn_fxn = lambda hs: shared_store_1._get_nn_hs(hs),
#                                                                  pi_dict = cache_mdp.pD,
#                                                                  v_dict = cache_mdp.e_vD if args.rmax else cache_mdp.vD ,
#                                                                  hs_st_disc_fxn = latent_to_hs_disc_fxn,
#                                                                  A = env.get_list_of_actions(),
#                                                                  tranDict = cache_mdp.tD,
#                                                                  rewardDict = cache_mdp.e_rD,
#                                                                  eps=2, render=False)
#         save_video(video, title = "rollout_t_DECODE_NN_" + str(eps), base_path=log_dirs_dict["py_log_dir"])