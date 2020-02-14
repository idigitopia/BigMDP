from ast import literal_eval
import numpy as np 
from collections import Iterable
import time
from statistics import mean, median
import math
from PIL import ImageFont, Image, ImageDraw
import tqdm
from statistics import mean , median

class EpsilonTracker:
    def __init__(self, params):
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_frames = params['epsilon_frames']

    def get_eps(self, frame):
        return max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)

def hAsh(state):
    return repr([int(round(i)) for i in state])

def unhAsh(encoded_state):
    try:
        return literal_eval(encoded_state)
    except:
        print(encoded_state)
        
    
def hm_dist(l1: list, l2: list):
    return sum(abs(le1 - le2) for le1, le2 in zip(l1, l2))



def rollout_with_nn_behavior(env, policy, hs_nn_fxn, pi_dict, v_dict, hs_st_disc_fxn, A,tranDict, rewardDict, epsilon=0, eps=1, lag=0, render=True):

    font_size = 10
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", font_size)
    video_list= []
    
    total_reward = 0
    all_rewards = []
    run_info = {}
    
    
    
    # Check if the state was found ! 
    # Check for the probability of the transition if the next state matches the MDP
    # else if the staet matches the MDP check for neighbors. one step or two step , may be other line up totally actions ? (optional)
    # if it matches some other state let it be flagged for now
    # Number of choices that the TranDict has for the current state. 
    # If it is a NN how often does it track back to one of the probable states ? 
    # make a summary of all this. 
    
    prev_in_MDP = True
    
    for i in range(eps):
        video = []
        run_reward, nn_search_count, policy_search_count = 0,0,0
        tran_perfect_match_count, tran_nn_match_count, tran_no_match_count = 0, 0, 0
        action_call_counts = {k:0 for k in A}
        state = env.reset()
        while True:
            to_print_dict={}
            
            action = policy(state)
            if epsilon > np.random.random():
                action = [env.action_space.sample()]
                
            

            # take a step
            action = action[0] if isinstance(action, Iterable) else action
            next_state, reward, done, _ = env.step(action)
            # update silly variables
            total_reward += reward
            run_reward += reward
            action_call_counts[action] +=1
            
            # check if the state exists inthe MDP
            hs_d = hs_st_disc_fxn([state])
            hns_d = hs_st_disc_fxn([next_state])
            nn_hs_d, nn_hns_d = hs_nn_fxn(hs_d), hs_nn_fxn(hns_d)
            a = action
            in_mdp = hns_d in pi_dict
            
            s_match, a_match, ns_match = prev_in_MDP, a in tranDict[nn_hs_d] , hns_d in pi_dict
            tran_perfect_match = s_match  and a_match and ns_match and nn_hs_d in tranDict[hs_d][a]
            tran_nn_match = not tran_perfect_match and nn_hns_d in tranDict[nn_hs_d][a] 
            tran_perfect_match_count += 1 if tran_perfect_match else 0 
            tran_nn_match_count += 1 if tran_nn_match else 0 
            tran_no_match_count += 1 if (not tran_perfect_match) and (not tran_nn_match) else 0 

            to_print_dict["Error"] = "None"
            to_print_dict["S_In_MDP"] = str(s_match) + " ->(" + str(a_match) + ")->" +  str(ns_match)
            if a not in tranDict[nn_hs_d]:
                to_print_dict["Error"] = "Action Not Found"
                to_print_dict["Tran_prob"] = "----"
            else:
                to_print_dict["Tran_prob"] = round(tranDict[nn_hs_d][a][nn_hns_d], 4)  if nn_hns_d in tranDict[nn_hs_d][a] else "----"
                to_print_dict["pred_reward"] = round(rewardDict[nn_hs_d][a], 4)


            to_print_dict["#  Actions"] = str(len(tranDict[nn_hns_d]))

            # print he expected Value and Next State Value:
            to_print_dict["VI State Value"] = str(round(v_dict[nn_hns_d], 4))
            expected_val = rewardDict[nn_hs_d][a] + \
                            0.999*sum([tranDict[nn_hs_d][a][hns_d] * (v_dict[hns_d] if hns_d in v_dict else 0) for hns_d in  tranDict[nn_hs_d][a]])
            to_print_dict["Exp VI (s,a) value"] = round(expected_val, 4)
            
            to_print_dict["-------------"] = "-----------------"
            to_print_dict["tran_perfect_match_count"]=tran_perfect_match_count
            to_print_dict["tran_nn_match_count"]=tran_nn_match_count
            to_print_dict["tran_no_match_count"]=tran_no_match_count

            policy_search_count  += 1
            nn_search_count += 0 if in_mdp else 1
                

            
            prev_in_MDP = in_mdp

                
            
            # check if transition match and spit the probability 

            state = next_state
            if render:
                env.render()
                time.sleep(lag)
                

            img_arr = env.render("rgb_array")
            img = Image.fromarray(np.concatenate([np.uint8(img_arr),np.full_like(img_arr, 1, dtype=np.uint8)], axis = 1))
            val = "\n".join([str(k)+": "+str(v) for k, v in to_print_dict.items()])
            draw = ImageDraw.Draw(img).text((img_arr.shape[1]+5,0), str(val)  , (255,255,0), font=font)
            
            video.append(np.array(img))    
            
            if done:
                break

        video_list.append(video)

        all_rewards.append(run_reward)

        run_info["Run"+str(i)]= {"perf":run_reward,
                                 "nn_search_count":nn_search_count,
                                 "policy_search_count":policy_search_count,
                                 "action_call_counts":action_call_counts,
                                 "nn_search_perc":(nn_search_count/policy_search_count)* 100,
                                 "tran_perfect_match_count":tran_perfect_match_count,
                                 "tran_nn_match_count":tran_nn_match_count,
                                 "tran_no_match_count":tran_no_match_count
                                 }

        print(run_reward)
    print("evaluate reward total avg", total_reward / eps)
    # info = {"min_reward": min(all_rewards), "max_reward": max(all_rewards)}
    info = {"mean_perf": mean(all_rewards),
            "median_perf": median(all_rewards),
            "min_reward": min(all_rewards),
            "max_reward": max(all_rewards),
            "run_info":run_info}
    return total_reward / eps, info, video_list



def evaluate_on_env(env, policy_func, eps_count=30, verbose=False, render = False, lag = 0):
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

    for _ in range(eps_count):
        sum_rewards, sum_steps = 0,0
        state_c = env.reset()

        done = False
        steps = 0

        while steps < env.max_episode_length and not done:
            sum_steps += 1
            policyAction = policy_func(state_c)
            state_c, reward, done, info = env.step(policyAction)
            sum_rewards += reward
            if(render):
                env.render()
                time.sleep(lag)

        eps_rewards.append(sum_rewards)
        eps_step_counts.append(sum_steps)

        if verbose:
            print("Steps:{}, Reward: {}".format(steps, sum_rewards))


    info = {"avg_reward":  mean(eps_rewards),
            "avg_steps": mean(eps_step_counts),
            "max_reward":  max(eps_rewards),
            "min_reward":  min(eps_rewards),
            "max_steps": max(eps_step_counts),
            "min_steps": min(eps_step_counts) ,}

    return info["avg_reward"], info






def evaluate_on_env(env, policy_func, eps_count=30, verbose=False, render = False, lag = 0):
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

    for _ in range(eps_count):
        sum_rewards, sum_steps = 0,0
        state_c = env.reset()

        done = False
        steps = 0

        while steps < env.max_episode_length and not done:
            sum_steps += 1
            policyAction = policy_func(state_c)
            state_c, reward, done, info = env.step(policyAction)
            sum_rewards += reward
            if(render):
                env.render()
                time.sleep(lag)

        eps_rewards.append(sum_rewards)
        eps_step_counts.append(sum_steps)

        if verbose:
            print("Steps:{}, Reward: {}".format(steps, sum_rewards))


    info = {"avg_reward":  mean(eps_rewards),
            "avg_steps": mean(eps_step_counts),
            "max_reward":  max(eps_rewards),
            "min_reward":  min(eps_rewards),
            "max_steps": max(eps_step_counts),
            "min_steps": min(eps_step_counts) ,}

    return info["avg_reward"], info


def evaluate_on_env_with_nn_count(env, policy_func, pi_dict,img_to_disc_fxn,  A, eps_count=30, verbose=False, policy_is_discrete=True, render = False, lag = 0):
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
        state_c, state_d = env.reset()
        ########
        action_call_counts = {k:0 for k in A}
        run_reward, nn_search_count, policy_search_count = 0,0,0


        done = False
        steps = 0

        while steps < env.max_episode_length and not done:
            sum_steps += 1
            policyAction = policy_func(state_d) if policy_is_discrete else policy_func(state_c)
            state_set, reward, done, info = env.step(policyAction)
            state_c, state_d = state_set

            action_call_counts[policyAction[0] if isinstance(policyAction, Iterable) else policyAction ] +=1
            nn_search_count += 1 if hAsh(img_to_disc_fxn([state_c])) not in pi_dict else 0
            policy_search_count += 1


            sum_rewards += env.reward_denormalizing_func(reward) if hasattr(env, 'normalizing_params') else reward
            if (render):
                env.render()
                time.sleep(lag)

        eps_rewards.append(sum_rewards)
        eps_step_counts.append(sum_steps)

        run_info["Run"+str(e)]= {"perf":run_reward,
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