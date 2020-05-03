from bigmdp.xai_module.video_helper import *
from copy import deepcopy as cpy

def simple_rollout_visualizer(env, policy, mdp, abstraction_fxn,
                              epsilon=0, eps_count=1, lag=0, render=False, action_meanings=None):

    get_nn_fxn = mdp._get_nn_hs_kdtree
    q_dict = cpy(mdp.qvalDict)
    v_dict = cpy(mdp.valueDict)
    A = mdp.A
    mdp.tD = mdp.tD
    rewardDict = mdp.rD
    action_meanings = action_meanings or [str(a) for a in A]


    video_list = []
    total_reward = 0
    all_rewards = []
    run_info = {}

    for i in range(eps_count):
        video = []
        run_reward, nn_search_count, policy_search_count = 0, 0, 0
        tran_perfect_match_count, tran_nn_match_count, tran_no_match_count = 0, 0, 0
        value_history = []
        action_call_counts = {k: 0 for k in A}
        state = env.reset()
        steps = 0

        frame_count = 0

        hs_d = abstraction_fxn([state])
        nn_hs_d = get_nn_fxn(hs_d)
        prev_state_in_mdp = hs_d in mdp.s2idx


        while True:

            # Get an action
            a = policy(state)
            # take the action
            next_state, reward, done, _ = env.step(a)

            # Render logic
            if render:
                env.render()
                time.sleep(lag)

            # Start the track of metrics
            total_reward += reward
            run_reward += reward
            action_call_counts[a] += 1
            policy_search_count += 1
            nn_search_count += 0 if prev_state_in_mdp else 1

            to_print_dict = {}

            # Get State match for observations and next ovservations
            hns_d = abstraction_fxn([next_state])
            nn_hns_d = get_nn_fxn(hns_d)

            # check if current state is in MDP
            this_state_in_mdp = hns_d in mdp.s2idx

            # print("#" * 40)
            # print(hs_d, hns_d)
            # print(nn_hs_d, a, nn_hns_d)


            # Get All possible transition Metrics
            s_match, a_match, ns_match = prev_state_in_mdp, a in mdp.tD[nn_hs_d], this_state_in_mdp
            t_exact_match = hns_d in mdp.tD[hs_d][a]
            t_nn_match = not t_exact_match and nn_hns_d in mdp.tD[nn_hs_d][a]

            tran_perfect_match_count += 1 if t_exact_match else 0
            tran_nn_match_count += 1 if t_nn_match else 0
            tran_no_match_count += 1 if (not t_exact_match) and (not t_nn_match) else 0

            to_print_dict["Error"] = "None"
            to_print_dict["State Id"] = str(mdp.s2idx[nn_hs_d])
            to_print_dict["S_In_MDP"] = str(s_match) + " ->(" + str(a_match) + ")->" + str(ns_match)
            to_print_dict["Tran_prob"] = round(mdp.tD[nn_hs_d][a][nn_hns_d], 4) if t_exact_match or  t_nn_match else "----"
            to_print_dict["pred_reward"] = round(rewardDict[nn_hs_d][a], 4)
#             to_print_dict["# Actions"] = str(mdp.get_seen_action_count(nn_hs_d))
            to_print_dict["Best Axn"]= str(action_meanings[a])
            to_print_dict["VI State Value"] = str(round(v_dict[nn_hns_d], 4))
            to_print_dict["Exp VI (s,a) value"] = round(q_dict[nn_hs_d][a], 4)
            to_print_dict["Actual Reward"] = round(reward, 4)
            to_print_dict["-------------"] = "-----------------"
            to_print_dict["tran_perfect_match_count"] = tran_perfect_match_count
            to_print_dict["tran_nn_match_count"] = tran_nn_match_count
            to_print_dict["tran_no_match_count"] = tran_no_match_count

            # create observation transition row
            img = Image.fromarray(env.render("rgb_array").astype("uint8")).resize((300,200))
            next_obs_arr = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
            next_obs_state_metrics_arr = get_printed_array(np.ones(next_obs_arr.shape, dtype=np.uint8),
                                                           to_print_dict=to_print_dict, font_size=9, fill_color="Aqua")
            obs_transition_row = np.concatenate([next_obs_state_metrics_arr, next_obs_arr],
                                                axis=1)
            obs_transition_row = add_title_on_top(obs_transition_row, title_height=20,
                                                  title_text="Metris" + " "*32 +  "State",
                                                  font_size=12)
            # Add all rows in to one image and append it in the video
            video.append(np.array(obs_transition_row).tolist())

            # other tracking metrics
            value_history.append(v_dict[nn_hns_d])

            # Update counts for next iteration
            prev_state_in_mdp = this_state_in_mdp
            hs_d = hns_d
            nn_hs_d = nn_hns_d

            state = next_state

            if done:
                break

        video_list.append(video)
        all_rewards.append(run_reward)

        run_info["Run" + str(i)] = {"perf": run_reward,
                                    "steps": steps,
                                    "nn_search_count": nn_search_count,
                                    "policy_search_count": policy_search_count,
                                    "action_call_counts": action_call_counts,
                                    "nn_search_perc": (nn_search_count / policy_search_count) * 100,
                                    "tran_perfect_match_count": tran_perfect_match_count,
                                    "tran_nn_match_count": tran_nn_match_count,
                                    "tran_no_match_count": tran_no_match_count,
                                    "value_history": value_history
                                    }
        print(run_reward)

    print("evaluate reward total avg", total_reward / eps_count)
    info = {"mean_perf": mean(all_rewards),
            "median_perf": median(all_rewards),
            "min_reward": min(all_rewards),
            "max_reward": max(all_rewards),
            "run_info": run_info}
    return total_reward / eps_count, info, video_list




def simple_rollout_visualizer_2(env, policy, mdp, abstraction_fxn,
                              epsilon=0, eps_count=1, lag=0, render=False, action_meanings=None):

    get_nn_fxn = mdp._get_nn_hs_kdtree
    q_dict = cpy(mdp.qvalDict)
    v_dict = cpy(mdp.valueDict)
    A = mdp.A
    mdp.tD = mdp.tD
    rewardDict = mdp.rD
    action_meanings = action_meanings or [str(a) for a in A]


    video_list = []
    total_reward = 0
    all_rewards = []
    run_info = {}

    for i in range(eps_count):
        video = []
        run_reward, nn_search_count, policy_search_count = 0, 0, 0
        tran_perfect_match_count, tran_nn_match_count, tran_no_match_count = 0, 0, 0
        value_history = []
        action_call_counts = {k: 0 for k in A}
        state = env.reset()
        steps = 0

        frame_count = 0

        hs_d = abstraction_fxn(state)
        nn_hs_d = get_nn_fxn(hs_d)
        prev_state_in_mdp = hs_d in mdp.s2idx


        while True:

            # Get an action
            a = policy(state)
            # take the action
            next_state, reward, done, _ = env.step(a)

            # Render logic
            if render:
                env.render()
                time.sleep(lag)

            # Start the track of metrics
            total_reward += reward
            run_reward += reward
            action_call_counts[a] += 1
            policy_search_count += 1
            nn_search_count += 0 if prev_state_in_mdp else 1

            to_print_dict = {}

            # Get State match for observations and next ovservations
            hns_d = abstraction_fxn(next_state)
            nn_hns_d = get_nn_fxn(hns_d)

            # check if current state is in MDP
            this_state_in_mdp = hns_d in mdp.s2idx

            # print("#" * 40)
            # print(hs_d, hns_d)
            # print(nn_hs_d, a, nn_hns_d)


            # Get All possible transition Metrics
            s_match, a_match, ns_match = prev_state_in_mdp, a in mdp.tD[nn_hs_d], this_state_in_mdp
            t_exact_match = hns_d in mdp.tD[hs_d][a]
            t_nn_match = not t_exact_match and nn_hns_d in mdp.tD[nn_hs_d][a]

            tran_perfect_match_count += 1 if t_exact_match else 0
            tran_nn_match_count += 1 if t_nn_match else 0
            tran_no_match_count += 1 if (not t_exact_match) and (not t_nn_match) else 0

            to_print_dict["Error"] = "None"
            to_print_dict["State Id"] = str(mdp.s2idx[nn_hs_d])
            to_print_dict["S_In_MDP"] = str(s_match) + " ->(" + str(a_match) + ")->" + str(ns_match)
            to_print_dict["Tran_prob"] = round(mdp.tD[nn_hs_d][a][nn_hns_d], 4) if t_exact_match or  t_nn_match else "----"
            to_print_dict["pred_reward"] = round(rewardDict[nn_hs_d][a], 4)
            to_print_dict["# Actions"] = str(mdp.get_seen_action_count(nn_hs_d))
            to_print_dict["Best Axn"]= str(action_meanings[a])
            to_print_dict["VI State Value"] = str(round(v_dict[nn_hns_d], 4))
            to_print_dict["Exp VI (s,a) value"] = round(q_dict[nn_hs_d][a], 4)
            to_print_dict["Actual Reward"] = round(reward, 4)
            to_print_dict["-------------"] = "-----------------"
            to_print_dict["tran_perfect_match_count"] = tran_perfect_match_count
            to_print_dict["tran_nn_match_count"] = tran_nn_match_count
            to_print_dict["tran_no_match_count"] = tran_no_match_count

            # create observation transition row
            img = Image.fromarray(env.render("rgb_array").astype("uint8")).resize((300,200))
            next_obs_arr = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
            next_obs_state_metrics_arr = get_printed_array(np.ones(next_obs_arr.shape, dtype=np.uint8),
                                                           to_print_dict=to_print_dict, font_size=9, fill_color="Aqua")
            obs_transition_row = np.concatenate([next_obs_state_metrics_arr, next_obs_arr],
                                                axis=1)
            obs_transition_row = add_title_on_top(obs_transition_row, title_height=20,
                                                  title_text="Metris" + " "*32 +  "State",
                                                  font_size=12)
            # Add all rows in to one image and append it in the video
            video.append(np.array(obs_transition_row).tolist())

            # other tracking metrics
            value_history.append(v_dict[nn_hns_d])

            # Update counts for next iteration
            prev_state_in_mdp = this_state_in_mdp
            hs_d = hns_d
            nn_hs_d = nn_hns_d

            state = next_state

            if done:
                break

        video_list.append(video)
        all_rewards.append(run_reward)

        run_info["Run" + str(i)] = {"perf": run_reward,
                                    "steps": steps,
                                    "nn_search_count": nn_search_count,
                                    "policy_search_count": policy_search_count,
                                    "action_call_counts": action_call_counts,
                                    "nn_search_perc": (nn_search_count / policy_search_count) * 100,
                                    "tran_perfect_match_count": tran_perfect_match_count,
                                    "tran_nn_match_count": tran_nn_match_count,
                                    "tran_no_match_count": tran_no_match_count,
                                    "value_history": value_history
                                    }
        print(run_reward)

    print("evaluate reward total avg", total_reward / eps_count)
    info = {"mean_perf": mean(all_rewards),
            "median_perf": median(all_rewards),
            "min_reward": min(all_rewards),
            "max_reward": max(all_rewards),
            "run_info": run_info}
    return total_reward / eps_count, info, video_list