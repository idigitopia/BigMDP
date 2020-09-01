from ast import literal_eval
import numpy as np 
from collections import Iterable
import time
from statistics import mean, median
import math
from PIL import ImageFont, Image, ImageDraw
from statistics import mean , median
from tqdm import tqdm
from collections import defaultdict

class EpsilonTracker:
    def __init__(self, params):
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_frames = params['epsilon_frames']

    def get_eps(self, frame):
        return max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)

def get_eps_policy(greedy_policy, random_policy, epsilon=0.1):
    """
    returns a exploration exploitation policy based on epsilon , greedy and random policy
    """
    return lambda s: random_policy(s) if (np.random.rand() < epsilon) else greedy_policy(s)

# def hAsh(state):
#     return repr([int(round(i)) for i in state])
# 
# def unhAsh(encoded_state):
#     try:
#         return literal_eval(encoded_state)
#     except:
#         print(encoded_state)

def hAsh(state):
    return tuple([i for i in state])

def unhAsh(encoded_state):
    try:
        return list(encoded_state)
    except:
        print(encoded_state)


def hm_dist(l1: list, l2: list):
    return sum(abs(le1 - le2) for le1, le2 in zip(l1, l2))



def evaluate_on_env(env, policy_func, eps_count=30, verbose=False, render = False, lag = 0, progress_bar=True, every_step_hook=None):
    """
    takes input environment and a policy and returns average rewards
    latent policy flag = True if the policy is a discrete policy
    :param env:
    :param policy_func:
    :param eps_count:
    :param return_info:
    :param verbose:
    :param policy_is_discrete:
    :return:
    """

    eps_rewards, eps_step_counts = [],[]
    run_info = {}
    iter__ = tqdm(range(eps_count)) if progress_bar else range(eps_count)
    action_counts = defaultdict(lambda :0)

    for e in iter__:
        sum_rewards, sum_steps = 0,0
        state_c = env.reset()

        done = False
        steps = 0

        while sum_steps < env.max_episode_length and not done:
            sum_steps += 1
            policyAction = policy_func(state_c)
            action_counts[policyAction] += 1
            state_c, reward, done, info = env.step(policyAction)
            sum_rewards += reward
            if every_step_hook is not None:
                every_step_hook(env, state_c)
            if(render):
                env.render(mode = "human")
                time.sleep(lag)

        eps_rewards.append(sum_rewards)
        eps_step_counts.append(sum_steps)
        run_info["Run"+str(e)]= {"perf":sum_rewards,
                                 "steps":sum_steps}


        if verbose:
            print("Steps:{}, Reward: {}".format(steps, sum_rewards))


    info = {"avg_reward":  mean(eps_rewards),
            "std_reward": np.std(eps_rewards),
            "avg_steps": mean(eps_step_counts),
            "std_steps" : np.std(eps_step_counts),
            "max_reward":  max(eps_rewards),
            "min_reward":  min(eps_rewards),
            "max_steps": max(eps_step_counts),
            "min_steps": min(eps_step_counts) ,
            "action_counts": action_counts,
            "run_info": run_info}

    return info["avg_reward"], info

def init2zero():
    return 0

def evaluate_on_env_with_nn_count(env, policy_func, check_if_exists_in_mdp, A = None,  eps_count=30, verbose=False, render = False, lag = 0):
    """
        takes input environment and a policy and returns average rewards
        latent policy flag = True if the policy is a discrete policy
        :param env:
        :param policy_func:
        :param eps_count:
        :param return_info:
        :param verbose:
        :param policy_is_discrete:
        :return:
        """

    eps_rewards, eps_step_counts = [], []
    run_info = {}

    for e in tqdm(range(eps_count)):
        sum_rewards, sum_steps = 0, 0
        state_c = env.reset()
        ########
        action_call_counts = {k:0 for k in A} if A is not None else defaultdict(init2zero)
        run_reward, nn_search_count, policy_search_count = 0,0,0


        done = False
        steps = 0

        while steps < env.max_episode_length and not done:
            sum_steps += 1
            policyAction = policy_func(state_c)
            state_c, reward, done, info = env.step(policyAction)

            action_call_counts[policyAction[0] if isinstance(policyAction, Iterable) else policyAction ] +=1
            nn_search_count += 0 if check_if_exists_in_mdp(state_c) else 1
            policy_search_count += 1


            sum_rewards += reward
            if (render):
                env.render()
                time.sleep(lag)

        eps_rewards.append(sum_rewards)
        eps_step_counts.append(sum_steps)

        run_info["Run"+str(e)]= {"perf":sum_rewards,
                                 "steps":policy_search_count,
                                 "nn_search_count":nn_search_count,
                                 "policy_search_count":policy_search_count,
                                 "action_call_counts":action_call_counts,
                                 "nn_search_perc":(nn_search_count/policy_search_count)* 100}

        if verbose:
            print("Steps:{}, Reward: {}".format(steps, sum_rewards))

    info = {"avg_reward": mean(eps_rewards),
            "avg_steps": mean(eps_step_counts),
            "max_reward": max(eps_rewards),
            "min_reward": min(eps_rewards),
            "max_steps": max(eps_step_counts),
            "min_steps": min(eps_step_counts),
            "run_info": run_info}

    return info["avg_reward"], info




def populate_model_from_buffer(mdp, tran_buffer, latent2discfxn):
    # all_transitions = tran_buffer.buffer
    _batch_size = 256

    start_end_indexes = get_iter_indexes(len(tran_buffer.buffer), _batch_size)
    for start_i, end_i in tqdm(start_end_indexes):
        batch, info = tran_buffer.sample_indices(list(range(start_i, end_i)))
        batch_s, batch_a, batch_ns, batch_r, batch_d = batch

        batch_hs = latent2discfxn(batch_s)
        batch_hns = latent2discfxn(batch_ns)
        for i in range(len(batch_hs)):
            mdp.consume_transition((hAsh(batch_hs[i]),
                                    int(batch_a[i][0]),
                                    hAsh(batch_hns[i]),
                                    float(batch_r[i][0]),
                                    int(batch_d[i][0])))
    return mdp


def get_iter_indexes(last_index, batch_size):
    if last_index % batch_size == 0:
        start_batch_list = list(
            zip(list(range(0, last_index, batch_size)), [batch_size] * int(last_index / batch_size)))
    else:
        start_batch_list = list(
            zip(list(range(0, last_index - batch_size, batch_size)) + [last_index - last_index % batch_size],
                [batch_size] * int(last_index / batch_size) + [last_index % batch_size]))

    start_end_list = [(i, i + b) for i, b in start_batch_list]

    return start_end_list
