from tqdm import  tqdm
import time
from collections import defaultdict
import numpy as np

def evaluate_on_env(env, policy_func, eps_count=30, verbose=False, render = False, lag = 0, progress_bar=True, every_step_hook=None, action_repeat=1, eval_eps = 0.001):
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
    last_action , same_action_count = 0,0
    for e in iter__:
        sum_rewards, sum_steps = 0,0
        state_c = env.reset()

        done = False
        steps = 0

        while not done:
            sum_steps += 1
            policyAction = policy_func(state_c) if np.random.uniform(0,1) > eval_eps else env.action_space.sample()
            # logic to not get stuck

            for _ in range(action_repeat):
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


    info = {"avg_reward":  np.mean(eps_rewards),
            "std_reward": np.std(eps_rewards),
            "avg_steps": np.mean(eps_step_counts),
            "std_steps" : np.std(eps_step_counts),
            "max_reward":  max(eps_rewards),
            "min_reward":  min(eps_rewards),
            "max_steps": max(eps_step_counts),
            "min_steps": min(eps_step_counts) ,
            "action_counts": action_counts,
            "run_info": run_info}

    return info["avg_reward"], info
