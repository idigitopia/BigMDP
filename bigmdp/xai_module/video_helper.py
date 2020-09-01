import torch
from collections import defaultdict,Iterable
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from bigmdp.utils.tmp_vi_helper import hAsh, unhAsh
from statistics import mean, median
import cv2
from torchvision.utils import make_grid
import time


def pad_image_with_one(img_arr, new_width):
    assert new_width >= img_arr.shape[1]
    new_shape = (img_arr.shape[0], new_width, img_arr.shape[2])
    padded_img_arr = np.zeros(new_shape, dtype=np.uint8)
    padded_img_arr[:img_arr.shape[0], :img_arr.shape[1], :] = img_arr
    return padded_img_arr


def write_video(frames, title, path=''):
    #   frames = np.multiply(np.stack(frames, axis=0).transpose(0, 2, 3, 1), 255).clip(0, 255).astype(np.uint8)[:, :, :, ::-1]  # VideoWrite expects H x W x C in BGR
    _, H, W, _ = frames.shape
    writer = cv2.VideoWriter(os.path.join(path, '%s.mp4' % title), cv2.VideoWriter_fourcc(*'mp4v'), 30., (W, H), True)
    for frame in frames:
        writer.write(frame)
    writer.release()

# def get_match_dict(img_buffer, latent_buffer, latent_to_disc_fxn):
#     disc_buffer = get_disc_buffer_from_latent_buffer(latent_buffer, latent_to_disc_fxn)
#     state_to_image_dict = defaultdict(init_with_list)
#     for i in tqdm(range(len(img_latent_buffer))):
#         state_to_image_dict[disc_buffer[i]].append(img_buffer[i])
#
#     return state_to_image_dict
#
#
# def get_match_dict_with_idx(img_buffer, latent_buffer, latent_to_disc_fxn):
#     disc_buffer = get_disc_buffer_from_latent_buffer(latent_buffer, latent_to_disc_fxn)
#     state_to_image_dict = defaultdict(init_with_list)
#     for i in tqdm(range(len(img_latent_buffer))):
#         img_with_idx = img_buffer[i]
#         state_to_image_dict[disc_buffer[i]].append(img_buffer[i])
#
#         state_to_image_dict[disc_buffer[i]].append(i)
#
#     return state_to_image_dict


def get_printed_array(img_arr, print_offset=(4, 4), to_print_dict={"Metrics": "Not Found"}, font_size=8,
                      fill_color="black", bold=False):
    metric_img = Image.fromarray(img_arr)
    if bold:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/OpenSans-ExtraBold.ttf", size=font_size)
    else:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/OpenSans-Regular.ttf", size=font_size)

    print_formatted_str = "\n".join([str(k) + ": " + str(v) for k, v in to_print_dict.items()])
    draw = ImageDraw.Draw(metric_img).text((4, 4), str(print_formatted_str), fill=fill_color, font=font)
    return np.array(metric_img)


def add_title_on_top(img_arr, title_height, title_text, font_size=12):
    title_shape = (title_height, img_arr.shape[1], img_arr.shape[2])
    img_arr = np.concatenate([np.ones(title_shape, dtype=np.uint8) * 200, np.uint8(img_arr)], axis=0)
    img_arr = get_printed_array(img_arr,
                                print_offset=(4, 4),
                                to_print_dict={"": title_text + " :"},
                                fill_color="DarkSlateGray",
                                font_size=font_size,
                                bold=True)

    return img_arr


from math import floor


def shuffle_image_for_grid(img_list, width=3):
    new_list = []
    for j in range(width):
        for i in range(int(floor(len(img_list) / width))):
            new_list.append(img_list[i * width + j])
    return new_list


def make_simple_grid(lazy_frames, max_list_length=6, nrow=2, rotate=True, row_meta={"Metrics": "Not Found"},
                     add_title=False, row_title="Default Title for Grid", title_height=20):
    fill_intensity = 40
    overflow = len(lazy_frames) - max_list_length + 1

    lazy_frames = lazy_frames[:max_list_length - 1]
    list_of_img = [cv2.cvtColor(np.array(gray)[-1, :, :], cv2.COLOR_GRAY2RGB) for gray in lazy_frames]
    pad_img = np.ones(list_of_img[0].shape, dtype=np.uint8) * fill_intensity
    padded_list_of_img = list_of_img + [pad_img.tolist()] * (max_list_length - 1 - len(list_of_img))

    metric_img_arr = get_printed_array(pad_img,
                                       print_offset=(4, 4),
                                       to_print_dict={**row_meta, **{"overflow": overflow if overflow > 0 else 0}},
                                       fill_color="Aqua",
                                       font_size=8)
    if rotate:
        metric_img_arr = torch.tensor(metric_img_arr).permute(1, 0, 2).numpy()
    padded_img = np.array(
        shuffle_image_for_grid([metric_img_arr.tolist()] + padded_list_of_img, width=int(max_list_length / nrow)))

    if rotate:
        grid_image = make_grid(torch.tensor(padded_img).permute(0, 3, 1, 2), nrow=nrow).permute(2, 1, 0).numpy()
    else:
        grid_image = make_grid(torch.tensor(padded_img).permute(0, 3, 2, 1), nrow=nrow).permute(2, 1, 0).numpy()

        # Add title on the top
    if add_title:
        grid_image = add_title_on_top(grid_image, title_height, title_text=row_title)

    return grid_image


def make_one_row(list_of_lazy_frames, row_title="Default Title", title_height=20, add_title=True,
                 list_of_meta_data=None, row_meta={"Default": "Row Meta"}, max_list_length=4, pad_width=160):
    fill_intensity = 80

    # get a grid for all lazy frames
    list_of_meta_data = list_of_meta_data or [{"id": i} for i in range(len(list_of_lazy_frames))]
    overflow_flag = len(list_of_lazy_frames) > max_list_length
    filtered_list_of_lazy_frames = list_of_lazy_frames[:max_list_length]
    grid_list = [make_simple_grid(lazy_frames, row_meta=list_of_meta_data[i]) for i, lazy_frames in
                 enumerate(filtered_list_of_lazy_frames)]
    pad_img = np.ones(grid_list[0].shape, dtype=np.uint8) * 40
    padded_grid_list = np.stack(grid_list + [pad_img] * (max_list_length - len(grid_list)))

    # add metrics on the rightmost side
    row_img = make_grid(torch.tensor(padded_grid_list).permute(0, 3, 2, 1), nrow=1).permute(2, 1, 0).numpy()
    pad_img = np.ones((row_img.shape[0], pad_width, row_img.shape[2]), dtype=np.uint8) * fill_intensity
    pad_img_with_metrics = get_printed_array(pad_img,
                                             print_offset=(4, 4),
                                             to_print_dict=row_meta,
                                             fill_color="Aqua",
                                             font_size=10,
                                             bold=False)

    row_img = np.concatenate([pad_img_with_metrics, row_img], axis=1)

    # Add title on the top
    if add_title:
        row_img = add_title_on_top(row_img, title_height, title_text=row_title, font_size=12)

    return row_img


def do_a_tree_search(s, tD, rD, vD, pD, nn_fxn, mode="most_probable"):
    trajectory = []
    #     trajectory.append(s)

    for i in range(14 * 3):
        s = hAsh(nn_fxn(unhAsh(s)))
        a = pD[s]
        ns = max(tD[s][a], key=vD.get)
        r = rD[s][a] if a in rD[s] else 0
        trajectory.append([s, a, ns, r])
        s = ns
        if s == "end_state":
            break

    return trajectory


# def rollout_with_nn_behavior_v2(env, policy, get_nn_fxn, pi_dict, v_dict, img_to_disc_fxn, A, tranDict, rewardDict,
#                                 state_to_image_dict, epsilon=0, eps=1, lag=0, render=True, action_meanings=None):
#     idx_to_state_dict = {i: s for i, s in enumerate(state_to_image_dict.keys())}
#     state_to_idx_dict = {s: i for i, s in enumerate(state_to_image_dict.keys())}
#
#     video_list = []
#     total_reward = 0
#     all_rewards = []
#     run_info = {}
#
#     prev_in_MDP = False
#
#     for i in range(eps):
#         video = []
#         run_reward, nn_search_count, policy_search_count = 0, 0, 0
#         tran_perfect_match_count, tran_nn_match_count, tran_no_match_count = 0, 0, 0
#         action_call_counts = {k: 0 for k in A}
#         state = env.reset()
#         obs_arr = env.render("rgb_array")
#
#         frame_count = 0
#
#         while True:
#
#             # Housekeeping
#             #             frame_count += 1
#             #             if frame_count >100:
#             #                 break
#
#             # Get an action
#             action = env.action_space.sample() if epsilon > np.random.random() else policy([state])
#             a = action[0] if isinstance(action, Iterable) else action
#             # take the action
#             next_state, reward, done, _ = env.step(a)
#
#             # Render logic
#             if render:
#                 env.render()
#                 time.sleep(lag)
#
#             # Start the track of metrics
#             total_reward += reward
#             run_reward += reward
#             action_call_counts[action] += 1
#
#             to_print_dict = {}
#
#             # Get State match for observations and next ovservations
#             s_d, ns_d = img_to_disc_fxn([state]), img_to_disc_fxn([next_state])
#             hs_d, hns_d = hAsh(s_d), hAsh(ns_d)
#             nn_hs_d, nn_hns_d = hAsh(get_nn_fxn(s_d)), hAsh(get_nn_fxn(ns_d))
#             # check if current state is in MDP
#             in_mdp = hns_d in pi_dict and hns_d in tranDict
#
#             print("#" * 40)
#             print(nn_hs_d, a, nn_hns_d)
#
#             # Get predicted next states for the last state and the action taken by the policy
#             # state_row, pot_next_states_row, next_state_row, (optional: Expected rollout Row, optimistic rollout row, pecimistic rollout row)
#             optimistic_trajectory = do_a_tree_search(hs_d, tranDict, rewardDict, v_dict, pi_dict, get_nn_fxn)
#             opt_states = [t[0] for t in optimistic_trajectory]
#             opt_images = [state_to_image_dict[s][0] for s in opt_states]
#             optimistic_row = make_simple_grid(opt_images,
#                                               max_list_length=14 * 3,
#                                               nrow=3,
#                                               add_title=True,
#                                               row_title="Optimistic Trajectory",
#                                               row_meta={"State Id": state_to_idx_dict[nn_hs_d],
#                                                         "Value": round(v_dict[nn_hs_d], 4),
#                                                         "Best Action": pi_dict[nn_hs_d]}
#                                               )
#
#             pot_next_states = [state_to_image_dict[hns_d_] for hns_d_ in tranDict[nn_hs_d][a]]
#             pot_next_states_row = make_one_row(pot_next_states,
#                                                max_list_length=4,
#                                                row_title="Predicted Potential Current States (Observation set)",
#                                                row_meta={"fan_out": len(pot_next_states),
#                                                          "pred R": round(rewardDict[nn_hs_d][a], 4)},
#                                                list_of_meta_data=[{"State Id": state_to_idx_dict[hns_d_],
#                                                                    "Value": round(v_dict[hns_d_], 4),
#                                                                    "Best Action": pi_dict[hns_d_]}
#                                                                   for hns_d_ in tranDict[nn_hs_d][a] if
#                                                                   hns_d_ not in ["end_state", "unknown_state"]]
#                                                )
#
#             next_state_row = make_simple_grid(state_to_image_dict[nn_hns_d],
#                                               max_list_length=14,
#                                               nrow=1,
#                                               add_title=True,
#                                               row_title="Current State (Observation set)",
#                                               row_meta={"State Id": state_to_idx_dict[nn_hns_d],
#                                                         "Value": round(v_dict[nn_hns_d], 4),
#                                                         "Best Action": pi_dict[nn_hns_d]}
#                                               )
#
#             state_row = make_simple_grid(state_to_image_dict[nn_hs_d],
#                                          max_list_length=14,
#                                          nrow=1,
#                                          add_title=True,
#                                          row_title="Previous State (Observation set)",
#                                          row_meta={"State Id": state_to_idx_dict[nn_hs_d],
#                                                    "Value": round(v_dict[nn_hs_d], 4),
#                                                    "Best Action": pi_dict[nn_hs_d]}
#                                          )
#
#             # Get All possible transition Metrics
#             s_match, a_match, ns_match = prev_in_MDP, a in tranDict[nn_hs_d], hns_d in pi_dict
#             tran_perfect_match = s_match and a_match and ns_match and nn_hs_d in tranDict[hs_d][a]
#             tran_nn_match = not tran_perfect_match and nn_hns_d in tranDict[nn_hs_d][a]
#             tran_perfect_match_count += 1 if tran_perfect_match else 0
#             tran_nn_match_count += 1 if tran_nn_match else 0
#             tran_no_match_count += 1 if (not tran_perfect_match) and (not tran_nn_match) else 0
#
#             to_print_dict["Error"] = "None"
#             to_print_dict["S_In_MDP"] = str(s_match) + " ->(" + str(a_match) + ")->" + str(ns_match)
#             if a not in tranDict[nn_hs_d]:
#                 to_print_dict["Error"] = "Action Not Found"
#                 to_print_dict["Tran_prob"] = "----"
#             else:
#                 to_print_dict["Tran_prob"] = round(tranDict[nn_hs_d][a][nn_hns_d], 4) if nn_hns_d in tranDict[nn_hs_d][
#                     a] else "----"
#                 to_print_dict["pred_reward"] = round(rewardDict[nn_hs_d][a], 4)
#
#             to_print_dict["#  Actions"] = str(len(tranDict[nn_hns_d]))
#
#             # print he expected Value and Next State Value:
#             to_print_dict["VI State Value"] = str(round(v_dict[nn_hns_d], 4))
#             expected_val = rewardDict[nn_hs_d][a] + \
#                            0.999 * sum(
#                 [tranDict[nn_hs_d][a][hns_d] * (v_dict[hns_d] if hns_d in v_dict else 0) for hns_d in
#                  tranDict[nn_hs_d][a]])
#             to_print_dict["Exp VI (s,a) value"] = round(expected_val, 4)
#
#             to_print_dict["-------------"] = "-----------------"
#             to_print_dict["tran_perfect_match_count"] = tran_perfect_match_count
#             to_print_dict["tran_nn_match_count"] = tran_nn_match_count
#             to_print_dict["tran_no_match_count"] = tran_no_match_count
#
#             # create observation transition row
#             next_obs_arr = env.render("rgb_array")
#             next_obs_state_metrics_arr = get_printed_array(np.ones(next_obs_arr.shape, dtype=np.uint8),
#                                                            to_print_dict=to_print_dict, font_size=9, fill_color="Aqua")
#             obs_action_arr = get_printed_array(np.ones(next_obs_arr.shape, dtype=np.uint8),
#                                                to_print_dict={"Action": a, "meta": action_meanings[a]}, font_size=12,
#                                                fill_color="Aqua", bold=True)
#             obs_transition_row = np.concatenate([next_obs_state_metrics_arr, next_obs_arr, obs_action_arr, obs_arr],
#                                                 axis=1)
#             obs_transition_row = add_title_on_top(obs_transition_row, title_height=20,
#                                                   title_text="                                                  Current State          <==          Action          <==          Prev State",
#                                                   font_size=12)
#             # Add all rows in to one image and append it in the video
#             max_width = max(d.shape[1] for d in
#                             [obs_transition_row, pot_next_states_row, next_state_row, state_row, optimistic_row])
#             #             print(obs_transition_row.shape[1],pot_next_states_row.shape[1],next_state_row.shape[1],state_row.shape[1])
#             obs_transition_row, pot_next_states_row, next_state_row, state_row, optimistic_row = [
#                 pad_image_with_one(img, max_width)
#                 for img in [obs_transition_row, pot_next_states_row, next_state_row, state_row, optimistic_row]]
#             full_img = np.concatenate(
#                 (state_row, obs_transition_row, next_state_row, pot_next_states_row, optimistic_row), axis=0)
#             video.append(np.array(full_img).tolist())
#
#             # Update counts for next iteration
#             policy_search_count += 1
#             nn_search_count += 0 if in_mdp else 1
#             prev_in_MDP = in_mdp
#             state = next_state
#             obs_arr = next_obs_arr
#
#             if done:
#                 break
#
#         video_list.append(video)
#
#         all_rewards.append(run_reward)
#
#         run_info["Run" + str(i)] = {"perf": run_reward,
#                                     "nn_search_count": nn_search_count,
#                                     "policy_search_count": policy_search_count,
#                                     "action_call_counts": action_call_counts,
#                                     "nn_search_perc": (nn_search_count / policy_search_count) * 100,
#                                     "tran_perfect_match_count": tran_perfect_match_count,
#                                     "tran_nn_match_count": tran_nn_match_count,
#                                     "tran_no_match_count": tran_no_match_count
#                                     }
#
#         print(run_reward)
#     print("evaluate reward total avg", total_reward / eps)
#     # info = {"min_reward": min(all_rewards), "max_reward": max(all_rewards)}
#     info = {"mean_perf": mean(all_rewards),
#             "median_perf": median(all_rewards),
#             "min_reward": min(all_rewards),
#             "max_reward": max(all_rewards),
#             "run_info": run_info}
#     return total_reward / eps, info, video_list




# def simple_rollout_with_nn_behavior_v2(env, policy, mdp, img_to_disc_fxn,
#                                       epsilon=0, eps=1, lag=0, render=True, action_meanings=None):
#     get_nn_fxn = mdp._get_nn_hs
#     pi_dict = mdp.pD
#     v_dict = mdp.vD
#     A = mdp.A
#     tranDict = mdp.tD
#     rewardDict = mdp.rD
#     action_meanings = action_meanings or [str(i) for i in A]
#
#
#
#     video_list = []
#     total_reward = 0
#     all_rewards = []
#     run_info = {}
#
#     prev_in_MDP = False
#
#     for i in range(eps):
#         video = []
#         run_reward, nn_search_count, policy_search_count = 0, 0, 0
#         tran_perfect_match_count, tran_nn_match_count, tran_no_match_count = 0, 0, 0
#         action_call_counts = {k: 0 for k in A}
#         state = env.reset()
#         obs_arr = env.render("rgb_array")
#
#         frame_count = 0
#
#         while True:
#
#             # Get an action
#             action = env.action_space.sample() if epsilon > np.random.random() else policy(state)
#             a = action[0] if isinstance(action, Iterable) else action
#             # take the action
#             next_state, reward, done, _ = env.step(a)
#
#             # Render logic
#             if render:
#                 env.render()
#                 time.sleep(lag)
#
#             # Start the track of metrics
#             total_reward += reward
#             run_reward += reward
#             action_call_counts[action] += 1
#
#             to_print_dict = {}
#
#             # Get State match for observations and next ovservations
#             s_d, ns_d = img_to_disc_fxn([state]), img_to_disc_fxn([next_state])
#             hs_d, hns_d = hAsh(s_d), hAsh(ns_d)
#             nn_hs_d, nn_hns_d = get_nn_fxn(hs_d), get_nn_fxn(hns_d)
#             # check if current state is in MDP
#             in_mdp = hns_d in pi_dict and hns_d in tranDict
#
#             # Get All possible transition Metrics
#             s_match, a_match, ns_match = prev_in_MDP, a in tranDict[nn_hs_d], hns_d in pi_dict
#             tran_perfect_match = s_match and a_match and ns_match and nn_hs_d in tranDict[hs_d][a]
#             tran_nn_match = not tran_perfect_match and nn_hns_d in tranDict[nn_hs_d][a]
#             tran_perfect_match_count += 1 if tran_perfect_match else 0
#             tran_nn_match_count += 1 if tran_nn_match else 0
#             tran_no_match_count += 1 if (not tran_perfect_match) and (not tran_nn_match) else 0
#             if s_match:
#                 assert nn_hs_d == hs_d
#
#             to_print_dict["Error"] = "None"
#             to_print_dict["State Id"] = str(mdp.s2idx[nn_hs_d])
#             to_print_dict["S_In_MDP"] = str(s_match) + " ->(" + str(a_match) + ")->" + str(ns_match)
#             if a not in tranDict[nn_hs_d]:
#                 to_print_dict["Error"] = "Action Not Found"
#                 to_print_dict["Tran_prob"] = "----"
#             else:
#                 to_print_dict["Tran_prob"] = round(tranDict[nn_hs_d][a][nn_hns_d], 4) if nn_hns_d in tranDict[nn_hs_d][a] else "----"
#                 to_print_dict["pred_reward"] = round(rewardDict[nn_hs_d][a], 4)
#
#             to_print_dict["#  Actions"] = str(len(tranDict[nn_hns_d]))
#             to_print_dict["Best Axn"]= str(action_meanings[pi_dict[nn_hs_d]])
#             # print he expected Value and Next State Value:
#             to_print_dict["VI State Value"] = str(round(v_dict[nn_hns_d], 4))
#             expected_val = rewardDict[nn_hs_d][a] + \
#                            0.999 * sum(
#                 [tranDict[nn_hs_d][a][hns_d] * (v_dict[hns_d] if hns_d in v_dict else 0) for hns_d in
#                  tranDict[nn_hs_d][a]])
#             to_print_dict["Exp VI (s,a) value"] = round(expected_val, 4)
#             to_print_dict["Actual Reward"] = round(reward, 4)
#
#
#
#             to_print_dict["-------------"] = "-----------------"
#             to_print_dict["tran_perfect_match_count"] = tran_perfect_match_count
#             to_print_dict["tran_nn_match_count"] = tran_nn_match_count
#             to_print_dict["tran_no_match_count"] = tran_no_match_count
#
#             # create observation transition row
#             next_obs_arr = env.render("rgb_array")
#             next_obs_state_metrics_arr = get_printed_array(np.ones(next_obs_arr.shape, dtype=np.uint8),
#                                                            to_print_dict=to_print_dict, font_size=9, fill_color="Aqua")
#             obs_transition_row = np.concatenate([next_obs_state_metrics_arr, next_obs_arr],
#                                                 axis=1)
#             obs_transition_row = add_title_on_top(obs_transition_row, title_height=20,
#                                                   title_text="                                                  Current State          <==          Action          <==          Prev State",
#                                                   font_size=12)
#             # Add all rows in to one image and append it in the video
#             video.append(np.array(obs_transition_row).tolist())
#
#             # Update counts for next iteration
#             policy_search_count += 1
#             nn_search_count += 0 if in_mdp else 1
#             prev_in_MDP = in_mdp
#             state = next_state
#
#             if done:
#                 break
#
#         video_list.append(video)
#
#         all_rewards.append(run_reward)
#
#         run_info["Run" + str(i)] = {"perf": run_reward,
#                                     "nn_search_count": nn_search_count,
#                                     "policy_search_count": policy_search_count,
#                                     "action_call_counts": action_call_counts,
#                                     "nn_search_perc": (nn_search_count / policy_search_count) * 100,
#                                     "tran_perfect_match_count": tran_perfect_match_count,
#                                     "tran_nn_match_count": tran_nn_match_count,
#                                     "tran_no_match_count": tran_no_match_count
#                                     }
#
#         print(run_reward)
#     print("evaluate reward total avg", total_reward / eps)
#     info = {"mean_perf": mean(all_rewards),
#             "median_perf": median(all_rewards),
#             "min_reward": min(all_rewards),
#             "max_reward": max(all_rewards),
#             "run_info": run_info}
#     return total_reward / eps, info, video_list





