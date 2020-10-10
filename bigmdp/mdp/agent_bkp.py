""" Quick script for an "Episodic Controller" Agent, i.e. nearest neighbor """
from bigmdp.data.buffer import get_iter_indexes
from collections import defaultdict
from sklearn.neighbors import KDTree
from collections import namedtuple
from tqdm import tqdm
from copy import deepcopy as cpy
import math
import random
from bigmdp.mdp.MDP_GPU_bkp import init2zero, init2list, init2dict, init2zero_def_dict, init2zero_def_def_dict
import time
import numpy as np

# from wrappers import *
MDPUnit = namedtuple('MDPUnit', 'tranProb origReward dist')

import pickle as pk
from os import path


    
class SimpleAgent(object):
    """
    Episodic agent is a simple nearest-neighbor based agent:
    - At training time it remembers all tuples of (state, action, reward).
    - After each episode it computes the empirical value function based
        on the recorded rewards in the episode.
    - At test time it looks up k-nearest neighbors in the state space
        and takes the action that most often leads to highest average value.
    """

    def __init__(self, mdp_T, net,  gamma=0.99, pred_error_threshold=1, epsilon_min=0.1,
                 fill_with = "model" , sample_neighbors = True, penalty_type = False, penalty_beta=1,
                 abstraction_flag = True, abstraction_threshold=0.05, filter_for_nn_region= True, use_prediction=True,
                 normalize_by_distance = False):

        # Main Components
        self.mdp_T = mdp_T
        self.net = net
        self.pred_error_threshold = pred_error_threshold
        self.fill_with = fill_with
        self.penalty_type = penalty_type
        self.penalty_beta = penalty_beta
        self.normalize_by_distance = normalize_by_distance
        self.gamma = gamma  # discount factor
        assert all([hasattr(net, attr) for attr in
                    ["encode_batch", "encode_single", "predict_single_transition", "predict_batch_transition"]])

        # Abstraction Flags
        self.abstraction_flag = abstraction_flag
        self.abstraction_threshold = abstraction_threshold

        # internal vars
        self.unseen_sa_pred_cache = {}  # predicted for unseen sa pairs
        self.in_mdp_sa_pairs = {}
        self.to_commit_sa_pairs = defaultdict(init2zero)
        self.to_commit_transitions = []
        self.dist_to_nn_cache = []
        self.mdp_cache= defaultdict(init2zero_def_def_dict)
        self.orig_reward_cache = defaultdict(init2zero_def_def_dict)
        self.iter = 0
        self.KDTree = None
        self.last_commit_iter = 0
        self.commit_seen_time, self.commit_predicted_time , self.solve_time = [], [] ,[]

        self.seed_policies()


    def random_policy(self, obs):
        return random.choice(self.mdp_T.A)

    def opt_policy(self, obs):
        return self.mdp_T.get_opt_action(self.net.encode_single(obs), smoothing=self.smoothing,
                                         soft_q=self.soft_q, weight_nn=self.normalize_by_distance)

    def safe_policy(self, obs):
        return self.mdp_T.get_safe_action(self.net.encode_single(obs), smoothing=self.smoothing,
                                          soft_q=self.soft_q, weight_nn=self.normalize_by_distance)

    def eps_optimal_policy(self, obs):
        eps_opt_pol = self.get_eps_policy(self.opt_policy, self.random_policy, epsilon=0.1)
        return eps_opt_pol(obs)

    def get_eps_policy(self,greedy_policy, random_policy, epsilon=0.1):
        """
        returns a exploration exploitation policy based on epsilon , greedy and random policy
        """
        return lambda s: random_policy(s) if (np.random.rand() < epsilon) else greedy_policy(s)

    def seed_policies(self, smoothing = False, soft_q = False ):
        self.smoothing = smoothing
        self.soft_q = soft_q
        self.policies = {"optimal":  self.opt_policy,
                                "random": self.random_policy,
                                "eps_optimal": self.eps_optimal_policy,
                                "safe": self.safe_policy}


    def parse(self, obs, a, obs_prime, r, d):
        """
        Parses a observation transition to state transition and stores it in a to_commit list
        :param obs:
        :param a:
        :param obs_prime:
        :param r:
        :param d:
        :return:
        """
        # workhorse of MDP Agent
        # get corresponding states for the transition and add it to "to_commit" list
        s, s_prime = self.net.encode_single(obs), self.net.encode_single(obs_prime)
        self.to_commit_transitions.append((s, a, s_prime, r, d))

    def batch_parse(self, obs_batch, a_batch, obs_prime_batch, r_batch, d_batch):
        s_batch, s_prime_batch = self.net.encode_batch(obs_batch), self.net.encode_batch(obs_prime_batch)
        for s,a,s_prime, r, d in zip(s_batch, a_batch,s_prime_batch, r_batch,d_batch):
            self.to_commit_transitions.append((s, a, s_prime, r, d))

    def commit_seen_transitions(self):
        # Add all to commit transitions to the MDP
        # track all to predict state action pairs
        print("Len of to commit transitions", len(self.to_commit_transitions))
        print("ABstraction Faldg", self.abstraction_flag)
        for s, a, s_prime, r, d in tqdm(self.to_commit_transitions):
            self.mdp_T.consume_transition((s, a, s_prime, r, d))

            for a_ in self.mdp_T.A:
                sa_pair = (s, a_)
                # 1 for seen sa_pair, 0 for unseen
                self.to_commit_sa_pairs[sa_pair] = 1 if a_ == a or self.to_commit_sa_pairs[sa_pair] == 1 else 0

                if (s_prime,a_) not in self.to_commit_sa_pairs and not d:
                    self.to_commit_sa_pairs[(s_prime,a_)] = 0


        self.mdp_T._update_nn_kd_tree()
        self.mdp_T._update_nn_kd_with_action_tree()
        print("Len of to commit sa pairs", len(self.to_commit_sa_pairs))

    def commit_predicted_transitions(self, verbose = False):
        if self.fill_with == "0Q_src-KNN":
            iterator_ = self.to_commit_sa_pairs.items()
            iterator_ = tqdm(iterator_) if verbose else iterator_

            for sa_pair, seen_flag in iterator_:
                # Calculate nearest neighbors of the state action in question and the ns they are pointing towards.
                s_, a_ = sa_pair
                knn_sa = self.mdp_T._get_knn_hs_kd_with_action_tree((s_,a_), k=self.mdp_T.mdp_build_k)
                knn_sa_normalized = self.mdp_T.get_kernel_probs(knn_sa, delta=self.mdp_T.knn_delta)
                self.dist_to_nn_cache.extend(list(knn_sa.values()))
                # assert all([len(self.mdp_T.known_tC[nn_s][a])==1 for nn_s, a in knn_sa]), \
                #     "Non stotchastic dynamics is not handled , please update the codebase"
                knn_sa_tran = { nn_s:(nn_s, a,
                               [rsum/tsum for rsum, tsum in zip(self.mdp_T.known_rC[nn_s][a].values() ,self.mdp_T.known_tC[nn_s][a].values())],  # reward
                               list(self.mdp_T.known_tC[nn_s][a].keys()), # ns
                               knn_sa[(nn_s,a)],knn_sa_normalized[(nn_s,a)]) for nn_s, a in knn_sa}

                if self.mdp_T.mdp_build_k > len(knn_sa_tran):
                    print(knn_sa_tran)
                    import pdb; pdb.set_trace()

                # Update the MDP transition probabilities with respect the found nearest neighbors
                # Reset Counts
                self.mdp_T.reset_counts_for_sa(s_, a_)

                for nn_s, a,r_list, nn_ns_list, dist_to_nn_s, prob_ns in knn_sa_tran.values():
                    for r,nn_ns in zip(r_list,nn_ns_list):
                        tran_count = int(prob_ns*100) if self.normalize_by_distance else 1
                        disc_reward = self.get_reward_logic(r, dist_to_nn_s, self.penalty_type, self.penalty_beta)
                        reward_count = int(prob_ns * 100) * disc_reward if self.normalize_by_distance else disc_reward
                        assert a==a_
                        if self.normalize_by_distance and prob_ns < 0.001:
                            continue # getting rid of highly improbable transitions
                        self.mdp_T.tC[s_][a_][nn_ns] += tran_count
                        self.mdp_T.rC[s_][a_][nn_ns] += reward_count
                        self.mdp_cache[s_][a_][nn_ns] = MDPUnit(prob_ns, r, dist_to_nn_s)
                        self.orig_reward_cache[s_][a_][nn_ns] = r

                try:
                    self.mdp_T.filter_sa_count_for_max_ns_count(s_, a_)
                    self.mdp_T.update_mdp_for(s_, a_)
                except:
                    import pdb; pdb.set_trace()

            self.to_commit_sa_pairs = defaultdict(init2zero)
            self.to_commit_transitions = []

        elif self.fill_with == "none":
            print("Leaving the unknown  state actions ot the same state")
            pass
        else:
            assert False , "Fill with can only be with the model or knn"

    def get_reward_logic(self, reward, dist_to_nn_ns, penalty_type, penalty_beta):
        if penalty_type == "none":
            disc_reward = reward
        elif penalty_type == "linear":
            disc_reward = reward - penalty_beta * dist_to_nn_ns
        else:
            assert False, "Unspecified Penalty type , please check parameters"

        return disc_reward


    def solve_mdp(self):
        self.mdp_T.curr_vi_error = 10
        self.mdp_T.solve(eps=0.001, mode="GPU", safe_bkp = True)
        self.qvalDict_cache = cpy(self.mdp_T.qvalDict)
        self.valueDict_cache = cpy(self.mdp_T.valueDict)

    def get_value(self,s):
        return self.valueDict_cache[self.mdp_T._get_nn_hs_kdtree(self.net.encode_single(s))]

    def get_q_value(self,s,a):
        return self.qvalDict_cache[self.mdp_T._get_nn_hs_kdtree(self.net.encode_single(s))][a]


    def build_mdp(self, train_buffer):

        print("Step 1 (Parse Transitions):  Running")
        st = time.time()

        _batch_size = 256
        start_end_indexes = get_iter_indexes(len(train_buffer), _batch_size)
        for start_i, end_i in tqdm(start_end_indexes):
            batch = train_buffer.sample_indices(list(range(start_i, end_i)))
            batch_ob, batch_a, batch_ob_prime, batch_r, batch_nd = batch
            batch_d = 1 - batch_nd
            self.batch_parse(batch_ob.numpy(), batch_a.numpy().squeeze(), batch_ob_prime.numpy(),
                               batch_r.numpy().squeeze(), batch_d.numpy().squeeze())

        print("Step 1 [Parse Transitions]:  Complete,  Time Elapsed: {}\n\n".format(time.time()-st))


        print("Step 2 [Seed Seen Transitions + Unknown (s,a) pairs]:  Running")
        st = time.time()
        self.commit_seen_transitions()
        print("Step 2 (Commit Seen Transitions):  Complete,  Time Elapsed: {} \n\n".format(time.time()-st))


        print("Step 3 [Commit all Transitions]:  Running")
        st = time.time()
        self.commit_predicted_transitions(verbose=True)
        print("Step 3 (Commit UnSeen Transitions):  Complete,  Time Elapsed: {}".format(time.time()-st))


        print("Step 4 [Solve MDP]:  Running")
        st = time.time()
        self.solve_mdp()
        print("% of missing transitions", self.mdp_T.unknown_state_action_count / (len(self.mdp_T.tD) * len(self.mdp_T.A)))
        self.seed_policies(smoothing=False, soft_q=False)
        print("Step 4 [Solve MDP]:  Complete,  Time Elapsed: {}\n\n".format(time.time()-st))


    def cache_mdp(self, file_path):
        st = time.time()
        print("Saving MDP and learnt net, gentle reminder that the parameters might have changed, plese resolve the MDP after loading")

        mdpCache_and_learnt_net = (self.mdp_cache, self.net.state_dict(),
                              self.net.pca if hasattr(self.net, "pca") else None)
        pk.dump(mdpCache_and_learnt_net, open(file_path, "wb"))

        print("Save Complete, Elapsed Time:{}s".format(time.time() - st))

    def load_mdp_from_cache(self, file_path):
        if not path.exists(file_path):
            print("File Does not Exist")
        else:
            st = time.time()
            print("loading MDP, and learnt net")
            mdp_and_learnt_net = pk.load(open(file_path, "rb"))
            self.mdp_cache, net_state_dict, pca = mdp_and_learnt_net
            self.net.load_state_dict(net_state_dict)
            self.net.pca = pca
            self.net.pca_flag = True if pca is not None else False

            print("Load Complete, Elapsed Time: {}s".format(time.time() - st))

            print("Building and solving Cache MDP")
            self.build_mdp_from_cache(self.mdp_cache)
            self.solve_mdp()

    def build_mdp_from_cache(self, mdp_cache):
        for s in tqdm(mdp_cache):
            for a in mdp_cache[s]:
                for ns in mdp_cache[s][a]:
                    r = mdp_cache[s][a][ns].origReward
                    dist_to_nn_s = mdp_cache[s][a][ns].dist
                    if dist_to_nn_s == 0:
                        self.mdp_T.consume_transition(cpy((s, a, ns, r, False)))

        for s in tqdm(mdp_cache):
            for a in mdp_cache[s]:
                self.mdp_T.reset_counts_for_sa(s, a) # get rid of transistions to unknown states
                for ns in mdp_cache[s][a]:
                    r = mdp_cache[s][a][ns].origReward
                    dist_to_nn_s = mdp_cache[s][a][ns].dist
                    prob_ns = mdp_cache[s][a][ns].tranProb

                    tran_count = max(1,int(prob_ns*100)) if self.normalize_by_distance else 1
                    disc_reward = self.get_reward_logic(r, dist_to_nn_s, self.penalty_type, self.penalty_beta)
                    reward_count = max(1,int(prob_ns*100)) * disc_reward if self.normalize_by_distance else disc_reward

                    self.mdp_T.tC[s][a][ns] = tran_count
                    self.mdp_T.rC[s][a][ns] = reward_count
                try:
                    self.mdp_T.filter_sa_count_for_max_ns_count(s, a)
                    self.mdp_T.update_mdp_for(s, a)
                except:
                    print("Some Exception occred here")
        self.mdp_T._update_nn_kd_tree()
        self.mdp_T._update_nn_kd_with_action_tree()
        print("Build Complete")

    def save_mdp(self, file_path):
        st = time.time()
        print("Saving MDP and learnt net")

        mdp_and_learnt_net = (self.mdp_T, self.net.state_dict(), self.net.pca if hasattr(self.net, "pca") else None)
        pk.dump(mdp_and_learnt_net, open(file_path, "wb"))

        sec_file_path = "".join(["".join(file_path.split(".")[:-1]), "_other_vars", ".", file_path.split(".")[-1]])
        other_variables = {"dist_to_nn_cache": self.dist_to_nn_cache,
                           "qvalDict_cache": self.qvalDict_cache,
                           "valueDict_cache": self.valueDict_cache}
        pk.dump(other_variables, open(sec_file_path, "wb"))

        print("Save Complete, Elapsed Time:{}s".format(time.time() - st))

    def load_mdp(self, file_path):
        if not path.exists(file_path):
            print("File Does not Exist")
        else:
            st = time.time()
            print("loading MDP, and learnt net")
            mdp_and_learnt_net = pk.load(open(file_path, "rb"))
            self.mdp_T, net_state_dict, pca = mdp_and_learnt_net
            self.net.load_state_dict(net_state_dict)
            self.net.pca = pca
            self.net.pca_flag = True if pca is not None else False

            sec_file_path = "".join(
                ["".join(file_path.split(".")[:-1]), "_other_vars", ".", file_path.split(".")[-1]])
            if path.exists(sec_file_path):
                other_variables = pk.load(open(sec_file_path, "rb"))
                self.dist_to_nn_cache = other_variables["dist_to_nn_cache"]
                self.qvalDict_cache = other_variables["qvalDict_cache"]
                self.valueDict_cache = other_variables["valueDict_cache"]

            print("Load Complete, Elapsed Time: {}s".format(time.time() - st))

    def log_all_mdp_metrics(self, mdp_frame_count,wandb_logger =None, tag_parent="MDP stats"):
        mdp_T = self.mdp_T
        all_distr = {"Transition Probabilty Distribution": mdp_T.tran_prob_distribution,
                     "Reward Distribution": mdp_T.reward_distribution,
                     "Value Distribution": list(mdp_T.valueDict.values()),
                     "Safe Value Distribution": list(mdp_T.s_valueDict.values()),
                     "State Action Fan In Distribution": mdp_T.state_action_fan_in_distribution,
                     "State Action Fan Out Distribution": mdp_T.state_action_fan_out_distribution,
                     "State Action Count Distribution": mdp_T.state_action_count_distribution,
                     "NN distance Distribtuion": [d for d in self.dist_to_nn_cache if d != 0 ],
                     "Self Loop Probability Distribution": mdp_T.self_loop_prob_distribution,
                     }

        all_scalars = {"State Count": len(mdp_T.tD)}

        if wandb_logger is not None:
            for name, metric in all_scalars.items():
                wandb_logger.log({tag_parent + "/_" + name: metric, 'mdp_frame_count': mdp_frame_count})

            for name, distr in all_distr.items():
                wandb_logger.log({tag_parent + "/Plotly_" + name: go.Figure(data=[go.Histogram(x=distr)]),
                                  'mdp_frame_count': mdp_frame_count})
                wandb_logger.log(
                    {tag_parent + "/_" + name: wandb_logger.Histogram(np.array(distr)), 'mdp_frame_count': mdp_frame_count})

        return all_distr


def get_eucledian_dist(s1, s2):
    return math.sqrt( sum( [ (s1[i]-s2[i])**2 for i, _ in enumerate(s1)] ))

