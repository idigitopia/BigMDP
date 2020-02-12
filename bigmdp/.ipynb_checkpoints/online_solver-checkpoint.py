import time

import argparse
from deeprmax.vi.VI_helper_funcs import *
from deeprmax.data.dataset import SimpleReplayBuffer, PrioritizedReplayBuffer
from deeprmax.data.env_gym import SimpleNormalizeEnv
from deeprmax.utils.utils_log import *
from async_vi.utils.tmp_vi_helper import *
from async_vi.continious_worker import *
from async_vi.utils.image_wrappers import *
from async_vi.utils.hyper_params import HYPERPARAMS
from deeprmax.utils.utils_directory import *
from async_vi.MDP import *


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


args = parser.parse_args()
# args = parser.parse_args("--discrete_bn 0 --bottle_neck_size 8 --env CartPole --rmax 1".split(" "))
base_file_path = "./result_dump/{}/{}".format(args.env, args.name +
                                              "_bn-" + str(args.bottle_neck_size) +
                                              "_sym-" + str(args.symbolic))

create_hierarchy(base_file_path)

def latent_to_disc_fxn(s):
    return hAsh(s * args.multiplyer)


log_dirs_dict, loggers_dict = get_advanced_log_dir_and_logger(ROOT_FOLDER = "Symbolic" if args.symbolic else "Image",
                                                             EXP_ID = args.env,
                                                             EXP_PARAMS="_bn-" + str(args.bottle_neck_size) +
                                                                        "_sym-" + str(args.symbolic) +
                                                                        "_rmax-" + str(bool(args.rmax)),
                                                             tb_log_keys=["tb_train_logger", "tb_vi_logger"])


if args.symbolic:
    params = HYPERPARAMS[args.env + "-sym"]
    env = SimpleNormalizeEnv(params["env_name"], max_episode_length=params["max_episode_length"])
else:
    print("Not Implemented Yet")  # Todo
    params = HYPERPARAMS[args.env + "-img"]
    assert False

if args.load:
    mdp = torch.load(base_file_path + "mdp_class.pth")
    unhash_dict_ = {hs: unhAsh(hs) for hs in mdp.tD if hs not in ["end_state", "unknown_state"]}

else:
    mdp = FullMDP(A=env.get_list_of_actions(),
                  ur=params["unknown_transition_reward"],
                  vi_params={"gamma": params["gamma"],
                             "slip_prob": params["slip_probability"],
                             "rmax_reward": params["rmax_reward"],
                             "rmax_thres": params["rmax_thres"], })
    unhash_dict_ = {}

shared_store_1 = SharedStorage.remote()
continious_solver = ContiniousSolver.remote(shared_store_1, mdp, backups_per_sync=5)
continious_solver.start.remote()
shared_store_1.start_value_iteration.remote()

all_rewards = []
eval_rewards, safe_eval_rewards = [], []
policy_fetch_time = []
tran_buffer = SimpleReplayBuffer(int(params["replay_size"]))
eps_tracker = EpsilonTracker(params)

frame_count = 0
warmup_eps = 10
eval_reward = 0


for eps in range(100000):
    s = env.reset()
    running_reward = 0

    while True:
        frame_count += 1

        st = time.time()
        if frame_count % params["vi_policy_sync"] == 0:
            if frame_count == params["vi_policy_sync"]:
                time.sleep(5)  # justa gate keeper for the VI to settle down in the beginning

            optimal_vectors, exploration_vectors, safe_vectors = ray.get(shared_store_1.get_value_vectors.remote())
            vD, qD, pD = optimal_vectors
            e_vD, e_qD, e_pD = exploration_vectors
            s_vD, s_qD, s_pD = safe_vectors
            assert "end_state" in pD
            assert "end_state" in e_pD
            assert "end_state" in s_pD


            opt_nn_policy = get_pi_policy(pD, unhash_dict_)
            expl_nn_policy = get_pi_policy(e_pD, unhash_dict_)
            safe_nn_policy = get_pi_policy(s_pD, unhash_dict_)

            nn_policy = expl_nn_policy if args.rmax else opt_nn_policy
        policy_fetch_time.append(time.time() - st)



        if frame_count < params["replay_initial"] or np.random.random() < eps_tracker.get_eps(frame_count):
            a = env.sample_random_action()
        else:
            a = nn_policy(latent_to_disc_fxn(s))

        if frame_count > params["replay_initial"] and args.rmax and eps%2==0:
            a = nn_policy(latent_to_disc_fxn(s))

        ns, r, d, i = env.step(a)
        _d = False if d and i["max_episode_length_exceeded"] == True else d
        s_d, ns_d = latent_to_disc_fxn(s), latent_to_disc_fxn(ns)
        unhash_dict_.update({s_d: s, ns_d: ns})
        shared_store_1.add_to_transition_queue.remote([s_d, a, ns_d, r, _d])
        tran_buffer.add([s, a, ns, r, _d])

        running_reward += r
        s = ns

        if frame_count > params["replay_initial"] and eps % 5 == 0:
            env.render()

        if frame_count % params['checkpoint_every'] == 0:
            shared_store_1.stop_value_iteration.remote()
            cache_mdp = ray.get(continious_solver.get_mdp.remote())
            shared_store_1.start_value_iteration.remote()
            continious_solver.start.remote()
            torch.save(cache_mdp, base_file_path + "mdp_class.pth")

        if d:
            time.sleep(1)  # Just cutting the VI Solver some Slack
            break

    all_rewards.append(running_reward)
    shared_store_1.stop_value_iteration.remote()
    cache_mdp = ray.get(continious_solver.get_mdp.remote())
    continious_solver.start.remote()
    shared_store_1.start_value_iteration.remote()



    #### Evaluation Code
    rmax_count = sum([1 for s in cache_mdp.tC for a in cache_mdp.tC[s] if sum(cache_mdp.tC[s][a].values()) < 10])
    vi_error, e_vi_error, s_vi_error = ray.get(shared_store_1.get_curr_vi_errors.remote())
    if frame_count > params["replay_initial"]:
        eval_reward = evaluate_on_env(env, lambda s: opt_nn_policy(latent_to_disc_fxn(s)), eps_count=2)[0]
        eval_rewards.append(eval_reward)

        safe_eval_reward = evaluate_on_env(env, lambda s: opt_nn_policy(latent_to_disc_fxn(s)), eps_count=2)[0]
        safe_eval_rewards.append(safe_eval_reward)

        loggers_dict["tb_train_logger"].add_scalar('Safe policy performance', float(safe_eval_reward), eps)
        loggers_dict["tb_train_logger"].add_scalar('Optimal policy performance', float(eval_reward), eps)
        loggers_dict["tb_train_logger"].add_scalar('Expl/Expt policy performance', float(running_reward), eps)

        loggers_dict["tb_train_logger"].add_scalar('MDP State Count', float(len(cache_mdp.vD)), eps)
        loggers_dict["tb_train_logger"].add_scalar('Rmax Count', float(rmax_count), eps)
        loggers_dict["tb_train_logger"].add_histogram('Optimal Value Distr', torch.tensor(list(cache_mdp.vD.values())), eps)
        loggers_dict["tb_train_logger"].add_histogram('Optimal Policy Distr', torch.tensor(list(cache_mdp.pD.values())), eps)

        loggers_dict["tb_train_logger"].add_scalar('Expl VI_error', float(e_vi_error), eps)
        loggers_dict["tb_train_logger"].add_scalar('Opt VI_error', float(vi_error), eps)
        loggers_dict["tb_train_logger"].add_scalar('Safe VI_error', float(s_vi_error), eps)

        loggers_dict["tb_train_logger"].add_scalar('Explr Epsilon', float(eps_tracker.get_eps(frame_count)), eps)
        loggers_dict["tb_train_logger"].add_scalar('Policy Fetch rate', float(mean(policy_fetch_time)), eps)


    print("episode:", eps,
          "reward:", running_reward,
          "Bkp error:", ray.get(shared_store_1.get_curr_vi_errors.remote()),
          "s8 visited:", len(unhash_dict_),
          "Rmax S8 Count", rmax_count,
          "epsilon", round(eps_tracker.get_eps(frame_count), 2),
          "Eval Reward", eval_reward,
          "policy_fetch_rate", mean(policy_fetch_time))

    policy_fetch_time = []



