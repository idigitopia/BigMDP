""" Quick script for an "Episodic Controller" Agent, i.e. nearest neighbor """
from bigmdp.data.dataset import *
from bigmdp.utils.tmp_vi_helper import *
from bigmdp.xai_module.rollout_visualizer import *
from sklearn.neighbors import KDTree
import plotly.graph_objects as go

# from wrappers import *

import pickle as pk
from os import path
def init2list():
    return []

def get_error_metrics_parametric(myAgent,tran_buffer):
    # all_transitions = tran_buffer.buffer
    _batch_size = 256

    start_end_indexes = get_iter_indexes(len(tran_buffer.buffer), _batch_size)
    all_distances = []
    for start_i, end_i in tqdm(start_end_indexes):
        batch, info = tran_buffer.sample_indices(list(range(start_i, end_i)))
        batch = [torch.FloatTensor(b).cuda() if myAgent.net.use_cuda else torch.FloatTensor(b) for b in batch]
        batch_s, batch_a, batch_ns, batch_r, batch_d = batch
        
        for i, s in enumerate(batch_s):
            z, z_prime = myAgent.net.encode_single(batch_s[i]), myAgent.net.encode_single(batch_ns[i])
            z_prime_hat, r_hat = myAgent.net.predict_single_transition(z, int(batch_a[i].squeeze().cpu().numpy()) )
            all_distances.append(torch.dist(torch.tensor(list(z_prime)),torch.tensor(list(z_prime_hat) )).item())
        
    return np.mean(all_distances), np.median(all_distances), np.quantile(all_distances, 0.1) , all_distances

    

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
        assert all([hasattr(net, attr) for attr in
                    ["encode_batch", "encode_single", "predict_single_transition", "predict_batch_transition"]])

        self.pred_error_threshold = pred_error_threshold
        self.fill_with = fill_with
        # self.sample_neighbors = sample_neighbors
        self.penalty_type = penalty_type
        self.penalty_beta = penalty_beta
        # self.filter_for_nn_region = filter_for_nn_region
        # self.use_prediction = use_prediction
        self.normalize_by_distance = normalize_by_distance

        self.abstraction_flag = abstraction_flag
        self.abstraction_threshold = abstraction_threshold

        # Consumption Parameters
        self.no_match_predictions = {}
        self.no_match_predictions_KDTree = None

        # options
        self.nnfind = 50  # how many nearest neighbors to consider in the policy?
        self.mem_needed = 10000  # amount of data to have before we can start exploiting
        self.mem_size = 100000  # maximum size of memory
        self.gamma = gamma  # discount factor

        # internal vars
        # self.seen_sa_cache = {}  # state action pairs
        self.unseen_sa_pred_cache = {}  # predicted for unseen sa pairs
        self.in_mdp_sa_pairs = {}
        self.to_commit_sa_pairs = defaultdict(init2zero)
        self.to_commit_transitions = []
        self.dist_to_nn_cache = []

        self.iter = 0
        self.KDTree = None

        self.seed_policies()
        self.last_commit_iter = 0
        self.commit_seen_time, self.commit_predicted_time , self.solve_time = [], [] ,[]


    def opt_policy(self, obs):
        return self.mdp_T.get_opt_action(self.net.encode_single(obs), smoothing=self.smoothing, soft_q=self.soft_q)

    def safe_policy(self, obs):
        return self.mdp_T.get_safe_action(self.net.encode_single(obs), smoothing=self.smoothing, soft_q=self.soft_q)

    def eps_optimal_policy(self, obs):
        eps_opt_pol = get_eps_policy(self.opt_policy, self.random_policy, epsilon=0.1)
        return eps_opt_pol(obs)

    def random_policy(self, obs):
        return random.choice(self.mdp_T.A)

    def seed_policies(self, smoothing = False, soft_q = False ):
        self.smoothing = smoothing
        self.soft_q = soft_q
        self.policies = {"optimal":  self.opt_policy,
                                "random": self.random_policy,
                                "eps_optimal": self.eps_optimal_policy,
                                "safe": self.safe_policy}


    def act(self, observation, policy_name = "optimal"):
        # we have enough data, we want to explore, and we have seen at least one episode already (so values were computed)
        policy_name = policy_name if self.iter > self.mem_needed else "random"
        return self.policies[policy_name](observation)

    def get_match_in_mdp(self, s_hat):
        nn_idxs, dist = self.mdp_T.KDTree.query_radius(np.array([s_hat]), r = self.pred_error_threshold,
                                               return_distance = True, sort_results = True)
        nn_idxs = nn_idxs[0][:20]
        nn_states = [self.mdp_T.state_list[i] for i in nn_idxs]
        return nn_states

    def update_no_match_KDTree(self):
        print("Updated no match KD Tree")
        self.pred_cache_transitions = defaultdict(init2list)
        for sa_pair, transition in self.no_match_predictions.items():
            s_, a_, pred_, r, d = transition
            self.pred_cache_transitions[pred_].append(sa_pair)

        self.pred_cache_state_list = list(self.pred_cache_transitions.keys())
        print("pred_cache_list", len(self.pred_cache_state_list))
        self.no_match_predictions_KDTree = KDTree(np.array(self.pred_cache_state_list), leaf_size=4)

    def lookup_no_match_cache(self, s):
        # returns a list of matched s,a pairs if pred match
        nn_idx = self.no_match_predictions_KDTree.query_radius(np.array([s]), r = self.pred_error_threshold)[0]
        nn_predictions = [self.pred_cache_state_list[i] for i in nn_idx]
        nn_sa_pairs = [self.pred_cache_transitions[pred_] for pred_ in nn_predictions]
        return nn_sa_pairs

    def consume_batch(self, obs_batch, a_batch, obs_prime_batch, r_batch, d_batch, max_len_reached,
                commit_seen_flag= True, commit_pred_flag= True, update_funknown_flag= True, solve_mdp_flag = True, lag_to_commit = 1000):

        s_batch, s_prime_batch = self.net.encode_batch(obs_batch), self.net.encode_batch(obs_prime_batch)
        for s,a,s_prime, r, d in zip(s_batch, a_batch,s_prime_batch, r_batch,d_batch):
            self.to_commit_transitions.append((s, a, s_prime, r, d))
            # self.seen_sa_cache[(s, a)] = 1
            self.iter += 1

            if (d or max_len_reached) and self.iter > self.last_commit_iter + lag_to_commit:  # episode Ended;
                self.commit_function(commit_seen_flag, commit_pred_flag, solve_mdp_flag)

    def consume(self, obs, a, obs_prime, r, d, max_len_reached,
                commit_seen_flag= True, commit_pred_flag= True, update_funknown_flag= True, solve_mdp_flag = True, lag_to_commit = 1000):
        """

        :param obs: observation
        :param a: action
        :param obs_prime: next observation
        :param r: reward
        :param d: done flag
        :param max_len_reached:
        :param commit_seen_flag:
        :param commit_pred_flag:
        :param update_funknown_flag:
        :param solve_mdp_flag:
        :param lag_to_commit:
        :return:
        """
        # assert isinstance(observation, np.ndarray) and observation.ndim == 1, 'unsupported observation type for now.'

        # Set all sa pair sas uncommited
        s, s_prime = self.net.encode_single(obs), self.net.encode_single(obs_prime)
        self.to_commit_transitions.append((s, a, s_prime, r, d))
        # self.seen_sa_cache[(s, a)] = 1

        self.iter += 1
        if (d or max_len_reached) and self.iter > self.last_commit_iter + lag_to_commit:  # episode Ended;
            self.commit_function(commit_seen_flag, commit_pred_flag, solve_mdp_flag)

    def commit_function(self, commit_seen_flag, commit_pred_flag,solve_mdp_flag):
            self.last_commit_iter = self.iter

            st = time.time()
            if commit_seen_flag:
                self.commit_seen_transitions()
                self.commit_seen_time.append(time.time()-st)

            st = time.time()
            if commit_pred_flag:
                self.commit_predicted_transitions(verbose=True)
                self.commit_predicted_time.append(time.time()-st)

            st = time.time()
            if solve_mdp_flag:
                self.solve_mdp()
                self.solve_time.append(time.time()-st)


    def commit_seen_transitions(self):
        # Add all to commit transitions to the MDP
        # track all to predict state action pairs
        print("Len of to commit transitions", len(self.to_commit_transitions))
        print("ABstraction Faldg", self.abstraction_flag)
        for s, a, s_prime, r, d in tqdm(self.to_commit_transitions):
            to_commit = False

            if self.abstraction_flag:
                assert False, "Abstraction Logic is not defined yet"
                # nn_s, dist_s  = list(self.mdp_T._get_knn_hs_kdtree(s, k=1).items())[0]
                # nn_sprime, dist_sprime = list(self.mdp_T._get_knn_hs_kdtree(s_prime, k=1).items())[0]
                # if self.abstraction_threshold > dist_s:
                #     if self.penalize_uncertainity:
                #         self.mdp_T.consume_transition(
                #             (nn_s, a, nn_sprime, r - self.penalty_beta * dist_sprime, d or dist_sprime > self.pred_error_threshold))
                #     else:
                #         self.mdp_T.consume_transition(
                #             (nn_s, a, nn_sprime, r, d or dist_sprime > self.pred_error_threshold))
                # else:
                #     self.mdp_T.consume_transition((s, a, s_prime, r, d))
                #     to_commit = True
            else:
                self.mdp_T.consume_transition((s, a, s_prime, r, d))
                to_commit = True

            if to_commit:
                for a_ in self.mdp_T.A:
                    sa_pair = (s, a_)
                    # 1 for seen sa_pair, 0 for unseen
                    self.to_commit_sa_pairs[sa_pair] = 1 if a_ == a or self.to_commit_sa_pairs[sa_pair] == 1 else 0

                    if (s_prime,a_) not in self.to_commit_sa_pairs and not d:
                        self.to_commit_sa_pairs[(s_prime,a_)] = 0


        self.mdp_T._update_nn_kd_tree()
        self.mdp_T._update_nn_kd_with_action_tree()
        print("Len of to commit sa pairs", len(self.to_commit_sa_pairs))

    # if self.use_prediction:
    #     pred_ns, pred_r, pred_d = self.net.predict_single_transition(s_, a_)
    #     knn_ns = self.mdp_T._get_knn_hs_kdtree(pred_ns, k=self.mdp_T.def_k)
    # else:

    def commit_predicted_transitions(self, verbose = False):
        #             print("checking to commit sa pairs")
        if self.fill_with == "0Q_src-KNN":
            #### House Keeping ####
            print("Implementation of Averagers Framework with costs called with Hyperparmeters, CodeName:", self.fill_with)
            print("K={} - Nearest Neighbors, Penalty Type:{}, penalty_beta:{}"
                  .format(self.mdp_T.mdp_build_k, self.penalty_type, self.penalty_beta))

            # assert not self.normalize_by_distance , "This logic is yet to be verified, may play bad in this config for seen transitions"
            if self.normalize_by_distance:
                print("This logic is yet to be verified, may play bad in this config for seen transitions, proceed at your own risk")

            ###############################################################################

            iterator_ = self.to_commit_sa_pairs.items()
            iterator_ = tqdm(iterator_) if verbose else iterator_

            for sa_pair, seen_flag in iterator_:
                # if seen_flag and not self.sample_neighbors:
                #     pass # Do not treat seen and unseen state actions differently

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
                try:
                    self.mdp_T.filter_sa_count_for_max_ns_count(s_, a_)
                    # update the probability ditribution
                    self.mdp_T.update_mdp_for(s_, a_)
                except:
                    import pdb; pdb.set_trace()

                # assert self.mdp_T.def_k > len(self.mdp_T.tD[s_][a_]) , "The number of next states is not supposet to be greater than the value for k"
            # clear to commit caches m set all as comited
            self.to_commit_sa_pairs = defaultdict(init2zero)
            self.to_commit_transitions = []

        elif self.fill_with == "1Q_dst-KNN":
            #### House Keeping ####
            print("Implementation of Plannable repr Framework with costs called with Hyperparmeters, CodeName:",
                  self.fill_with)
            print("K={} - Nearest Neighbors, Penalty Type:{}, penalty_beta:{}"
                  .format(self.mdp_T.mdp_build_k, self.penalty_type, self.penalty_beta))
            #
            # if self.sample_neighbors:
            #     print("#" * 40, "Warning", "#" * 40, "\n",
            #           "Sampling neighbors is invalide for K-NN case proceeding without sampling nearest neighbors")
            ###############################################################################

            iterator_ = self.to_commit_sa_pairs.items()
            iterator_ = tqdm(iterator_) if verbose else iterator_

            for sa_pair, seen_flag in iterator_:
                if seen_flag: # and not self.sample_neighbors:
                    continue # if it is a seen state action pair and we dont need to sample neighbors then nothing to do here.

                # Calculate nearest neighbors of the state action in question and the ns they are pointing towards.
                s_, a_ = sa_pair
                pred_ns, pred_r, pred_d = self.net.predict_single_transition(s_, a_)
                knn_ns = {"end_state": 1e-20} if pred_d else self.mdp_T._get_knn_hs_kdtree(pred_ns, k=self.mdp_T.mdp_build_k)
                knn_ns_normalized = self.mdp_T.get_kernel_probs(knn_ns, delta=self.mdp_T.knn_delta)
                # assert all([len(self.mdp_T.tD[sa[0]][sa[1]])==1 for sa in knn_sa]), "Non stotchastic dynamics is not handled , please update the codebase"
                knn_sa_tran = {nn_ns:(s_, a_, pred_r, nn_ns, dist_to_nn_ns,knn_ns_normalized[nn_ns])
                                for nn_ns,dist_to_nn_ns in knn_ns.items()}

                # Update the MDP transition probabilities with respect the found nearest neighbors
                # Reset Counts
                self.mdp_T.reset_counts_for_sa(s_, a_)

                for nn_s, a,r, nn_ns, dist_to_nn_s, prob_ns in knn_sa_tran.values():
                    tran_count = int(prob_ns*100) if self.normalize_by_distance else 1
                    disc_reward = self.get_reward_logic(r, dist_to_nn_s, self.penalty_type, self.penalty_beta)
                    reward_count = int(prob_ns * 100) * disc_reward if self.normalize_by_distance else disc_reward
                    assert a==a_
                    if self.normalize_by_distance and prob_ns < 0.001:
                        continue  # getting rid of highly improbable transitions
                    self.mdp_T.tC[s_][a_][nn_ns] += tran_count
                    self.mdp_T.rC[s_][a_][nn_ns] += reward_count

                self.mdp_T.filter_sa_count_for_max_ns_count(s_, a_)
                # update the probability ditribution
                self.mdp_T.update_mdp_for(s_, a_)

                # assert self.mdp_T.def_k > len(self.mdp_T.tD[s_][a_]) , "The number of next states is not supposet to be greater than the value for k"
            # clear to commit caches m set all as comited
            self.to_commit_sa_pairs = defaultdict(init2zero)
            self.to_commit_transitions = []

        elif self.fill_with == "kkQ_dst-1NN":
            #### House Keeping ####
            print("Implementation of Voronoi sampling  Framework with costs called with Hyperparmeters, CodeName:",
                  self.fill_with)
            print("K_q={} - Nearest Neighbors, Penalty Type:{}, penalty_beta:{}"
                  .format(self.mdp_T.mdp_build_k, self.penalty_type, self.penalty_beta))

            assert not self.normalize_by_distance , "This logic is yet to be verified, may play bad in this config for seen transitions"

            # if not self.sample_neighbors:
            #     print("#" * 40, "Warning", "#" * 40, "\n",
            #           "Sampling neighbors must be on for this , automatically switching it on")
            #     self.sample_neighbors = True
            ###############################################################################

            iterator_ = self.to_commit_sa_pairs.items()
            iterator_ = tqdm(iterator_) if verbose else iterator_

            for sa_pair, seen_flag in iterator_:
                if seen_flag: # and not self.sample_neighbors:
                    pass  # We dont care if we see or dont see a transition , we continue

                # Calculate nearest neighbors of the state action in question and the ns they are pointing towards.
                s_, a_ = sa_pair
                voronoi_samples = self.get_voronoi_samples(s_, self.mdp_T.mdp_build_k)
                assert len(voronoi_samples)
                k_predictions = {sample_s:self.net.predict_single_transition(sample_s, a_) for sample_s in voronoi_samples}
                knn_ns_normalized = self.mdp_T.get_kernel_probs(voronoi_samples, delta=self.mdp_T.knn_delta)
                # k_predictions[s] = pred_ns, pred_r, pred_ d
                knn_sa_tran = {sample_s: (s_, a_, k_predictions[sample_s][1],
                               "end_state" if k_predictions[sample_s][2]
                                          else self.mdp_T._get_nn_hs_kdtree(k_predictions[sample_s][0]),
                                dist_to_sample, knn_ns_normalized[sample_s])
                               for sample_s, dist_to_sample in voronoi_samples.items()}

                # Update the MDP transition probabilities with respect the found nearest neighbors
                # Reset Counts
                self.mdp_T.reset_counts_for_sa(s_, a_)

                for nn_s, a,r, nn_ns, dist_to_nn_s, prob_ns in knn_sa_tran.values():
                    tran_count = int(prob_ns*100) if self.normalize_by_distance else 1
                    disc_reward = self.get_reward_logic(r, dist_to_nn_s, self.penalty_type, self.penalty_beta)
                    reward_count = int(prob_ns * 100) * disc_reward if self.normalize_by_distance else disc_reward
                    assert a==a_
                    if self.normalize_by_distance and prob_ns < 0.001:
                        continue  # getting rid of highly improbable transitions
                    self.mdp_T.tC[s_][a_][nn_ns] += tran_count
                    self.mdp_T.rC[s_][a_][nn_ns] += reward_count

                self.mdp_T.filter_sa_count_for_max_ns_count(s_, a_)
                # update the probability ditribution
                self.mdp_T.update_mdp_for(s_, a_)

                # assert self.mdp_T.def_k > len(self.mdp_T.tD[s_][a_]) , "The number of next states is not supposet to be greater than the value for k"
            # clear to commit caches m set all as comited
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

    def get_voronoi_samples(self, s, k):
        assert self.mdp_T.is_kd_tree_initialized

        knn_s = self.mdp_T._get_knn_hs_kdtree(s, k=k)
        sample_search_list = list(knn_s.keys())
        all_sampled_neighbors = []
        for nn_s, dist in knn_s.items():
            if dist == 0:
                all_sampled_neighbors.extend([nn_s])
            else:
                all_sampled_neighbors.extend([
                    get_inner_section(s, nn_s, 4, 5), get_outer_section(s, nn_s, 4, 4 + 5 + 4),
                    get_inner_section(s, nn_s, 3, 7), get_outer_section(s, nn_s, 3, 3 + 7 + 3),
                    get_inner_section(s, nn_s, 2, 8), get_outer_section(s, nn_s, 2, 2 + 8 + 2),
                ])

        #filter samples from line
        all_sampled_neighbors = [nn_s for nn_s in all_sampled_neighbors
                                 if bruteForce_nn(sample_search_list, nn_s)[0] == s]
        all_sampled_neighbors = random.choices(all_sampled_neighbors, k = k )
        sampled_neighbor_dict = {sample_s: get_eucledian_dist(sample_s, s) for sample_s in all_sampled_neighbors}
        return sampled_neighbor_dict

    def update_fully_unknown_sa_pairs(self):
        for s in self.mdp_T.fully_unknown_states:
            for a in self.mdp_T.A:
                self.mdp_T.fill_knn_for_s_a(s, a)
        self.mdp_T.fully_unknown_states= {}

    def solve_mdp(self):
        self.mdp_T.curr_vi_error = 10
        self.mdp_T.solve(eps=0.001, mode="GPU", safe_bkp = True)
        self.qvalDict_cache = cpy(self.mdp_T.qvalDict)
        self.valueDict_cache = cpy(self.mdp_T.valueDict)

    def get_value(self,s):
        return self.valueDict_cache[self.mdp_T._get_nn_hs_kdtree(self.net.encode_single(s))]

    def get_q_value(self,s,a):
        return self.qvalDict_cache[self.mdp_T._get_nn_hs_kdtree(self.net.encode_single(s))][a]

        
    def save_mdp(self, file_path):
        st = time.time()
        print("Saving MDP and learnt net")
        
        mdp_and_learnt_net = (self.mdp_T, self.net.learnt_net.state_dict(), self.net.learnt_net.pca if hasattr(self.net.learnt_net, "pca") else None)
        other_variables = {"nn"}
        pk.dump(mdp_and_learnt_net, open(file_path, "wb"))
        
        sec_file_path = "".join(["".join(file_path.split(".")[:-1]) ,"_other_vars",".",file_path.split(".")[-1]])
        other_variables = {"dist_to_nn_cache":self.dist_to_nn_cache,
                          "qvalDict_cache":self.qvalDict_cache,
                            "valueDict_cache":self.valueDict_cache}
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
            self.net.learnt_net.load_state_dict(net_state_dict)
            self.net.learnt_net.pca = pca
            self.net.learnt_net.pca_flag = True if pca is not None else False
            
            sec_file_path = "".join(["".join(file_path.split(".")[:-1]) ,"_other_vars",".",file_path.split(".")[-1]])
            if path.exists(sec_file_path):
                other_variables =  pk.load(open(sec_file_path, "rb"))
                self.dist_to_nn_cache = other_variables["dist_to_nn_cache"]
                self.qvalDict_cache =  other_variables["qvalDict_cache"]
                self.valueDict_cache =  other_variables["valueDict_cache"]
            
            print("Load Complete, Elapsed Time: {}s".format(time.time() -st))

    def build_mdp(self, train_buffer, batch_parse=False):
        # print("Network being Used for pred:{}, reward:{},  terminal:{} over_Rid_threshold:{} [0 = False]" \
        #       .format(self.net.pred, self.net.reward, self.net.terminal, self.net.over_ride_threshold))

        print("Parsing Dataset Batch Parse:", batch_parse)
        if batch_parse:
            _batch_size = 256
            start_end_indexes = get_iter_indexes(len(train_buffer.buffer), _batch_size)
            for start_i, end_i in tqdm(start_end_indexes):
                batch, info = train_buffer.sample_indices(list(range(start_i, end_i)))
                batch_ob, batch_a, batch_ob_prime, batch_r, batch_d = batch
                self.consume_batch(np.array(batch_ob), np.array(batch_a).squeeze(), np.array(batch_ob_prime),
                                      np.array(batch_r).squeeze(), np.array(batch_d).squeeze(), False,
                                      commit_seen_flag=False, commit_pred_flag=False, update_funknown_flag=False,
                                      solve_mdp_flag=False)

        else:
            for ob, action, ob_prime, reward, _d, in tqdm(train_buffer.buffer):
                self.consume(np.array(ob), action[0], np.array(ob_prime), reward[0], _d[0], False,
                                commit_seen_flag=False, commit_pred_flag=False, update_funknown_flag=False,
                                solve_mdp_flag=False)

        print("Parsing Complete")
        print("Commiting Seen Transitions")
        self.commit_seen_transitions()
        print("Commit Complete")

        self.commit_predicted_transitions(verbose=True)
        if self.fill_with == "none":
            print("filling fully unknown states here, total unknown state count:{}",self.mdp_T.unknown_state_action_count)
            self.mdp_T.fill_fully_unknown_states()
        else:
            assert self.mdp_T.unknown_state_action_count == 0

        self.solve_mdp()
        print("% of missing transitions", self.mdp_T.unknown_state_action_count / (
                len(self.mdp_T.tD) * len(self.mdp_T.A)))
        self.seed_policies(smoothing=False, soft_q=False)


    def log_all_mdp_metrics(self, mdp_frame_count,wandb_logger =None, tag_parent="MDP stats"):
        mdp_T = self.mdp_T
        all_distr = {"Transition Probabilty Distribution": mdp_T.tran_prob_distribution,
                     "Reward Distribution": mdp_T.reward_distribution,
                     "Value Distribution": list(mdp_T.valueDict.values()),
                     "Safe Value Distribution": list(mdp_T.s_valueDict.values()),
                     "State Action Fan In Distribution": mdp_T.state_action_fan_in_distribution,
                     "State Action Fan Out Distribution": mdp_T.state_action_fan_out_distribution,
                     "State Action Count Distribution": mdp_T.state_action_count_distribution,
                     #                  "State Count Distr": mdp_T.state_count_distribution,
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

    def model_rollout(s, tD, rD, vD, pD, nn_fxn, mode="most_probable"):
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


    def evaluate_policy_in_MDP(self,start_state, max_episode_length, reward_fxn = None):
        tD, rD, vD, qD = self.mdp_T.tD, self.mdp_T.rD, cpy(self.mdp_T.valueDict), cpy(self.mdp_T.qvalDict)

        z = self.net.encode_single(start_state)
        nn_z = self.mdp_T._get_knn_hs_kdtree(z)
        s = nn_z
        cumulative_reward = 0
        cumulative_penalty = 0

        for i in range(max_episode_length):
            a = self.mdp_T.get_opt_action(nn_z,smoothing = self.smoothing, soft_q = self.soft_q)
            ns = random.choices(list(tD[s][a].keys()), weights= list(tD[s][a].values()), k = 1 )
            r = rD[s][a]
            cumulative_reward += 2
            cumulative_penalty += 2-r
            s = ns
            if s == "end_state":
                break

        return cumulative_reward, cumulative_penalty



def get_inner_section( start_point, end_point, m,n):
    return tuple([(m * end_point[i] + n * start_point[i]) / (m+n) for i in range(len(start_point))])

def get_outer_section( start_point, end_point, m,n):
    return tuple([(m * end_point[i] - n * start_point[i]) / (m-n) for i in range(len(start_point))])

def bruteForce_nn(search_list, query, return_nn =True):
    dist_dict = {s_hat: get_eucledian_dist(s_hat, query) for s_hat in search_list}
    # print(len(dist_dict))
    if return_nn:
        nn = min(dist_dict, key = dist_dict.get)
        return nn, dist_dict[nn]
    else:
        return dist_dict

import math

def get_eucledian_dist(s1, s2):
    return math.sqrt( sum( [ (s1[i]-s2[i])**2 for i, _ in enumerate(s1)] ))
    # s1,s2 = torch.FloatTensor(list(s1)), torch.FloatTensor(list(s2))
    # return torch.dist(s1,s2).item()



def ensemble_agent(list_of_agent):
    pass


