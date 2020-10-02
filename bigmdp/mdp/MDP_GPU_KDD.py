import torch
from collections import defaultdict
from collections import deque
import math
import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from collections import defaultdict
from pycuda.reduction import ReductionKernel
import numpy
import time
import math as mth
from bigmdp.utils.kernel_templates import complex_vi_kernel_code_template as vi_kernel_template
from bigmdp.utils.kernel_templates import NN_kernel_code_template
import pycuda
# -- initialize the device
from statistics import mean
from bigmdp.utils.tmp_vi_helper import *
import numpy as np
from tqdm import tqdm
import pycuda.autoinit
from collections import namedtuple
from copy import deepcopy as cpy
from collections import Counter, deque
import random
from bigmdp.utils.tmp_vi_helper import *
from bisect import bisect
from sklearn.neighbors import KDTree
from collections import Counter
from heapq import nsmallest, nlargest


def init2dict():
    return {}


def init2list():
    return []


def init2zero():
    return 0


def init2zero_def_dict():
    return defaultdict(init2zero)


def init2zero_def_def_dict():
    return defaultdict(init2zero_def_dict)


class FullMDP(object):
    def __init__(self, A, ur=-1000, MAX_S_COUNT=1000000, MAX_NS_COUNT=20,
                 vi_params={"gamma": 0.99,
                            "slip_prob": 0.1,
                            "rmax_reward": 1000,
                            "rmax_thres": 10,
                            "balanced_explr": False,
                             "rmin":-1000},
                 mdp_params={"weight_transitions":True,
                            "mdp_build_k":10,
                             "plcy_lift_k":10,
                            "knn_delta" : 0.01,
                            "calc_action_vector":True,
                            "penalty_beta":1},
                 default_mode="GPU",
                 smooth_with_seen = True):

        """
        :param A: Action Space of the MDP
        :param ur: reward for undefined state action pair
        """
        self.omit_list = ["end_state", "unknown_state"]
        self.penalty_beta = mdp_params["penalty_beta"]
        self.smooth_with_seen = smooth_with_seen

        self.fill_engine_time = deque(maxlen = 100)
        self.unknown_sa_fill_time_list = deque(maxlen = 100)

        self.to_commit_sa_dict = {}
        self.known_sa_dict = {}
        self.unknown_sa_dict = {}

        self.KDTree = None
        self.KDActionTrees = {a: None for a in A}
        self.known_sa_list = []
        self.factored_known_sa_list = []


        self.penalize_uncertainity = mdp_params["penalize_unknown_transitions"]
        self.calc_action_vector = mdp_params["calc_action_vector"]


        self.consumption_count = 0
        self.mdp_build_k = mdp_params["mdp_build_k"]
        self.plcy_lift_k = mdp_params["plcy_lift_k"]
        self.knn_delta = mdp_params["knn_delta"]
        self._nn_cache = {}
        self.def_radius = 0

        # KNN Params
        self.known_tC = defaultdict(init2zero_def_def_dict)
        self.known_rC = defaultdict(init2zero_def_def_dict)


        # VI CPU/GPU parameters
        self.vi_params = vi_params
        self.MAX_S_COUNT = MAX_S_COUNT  # Maximum number of state that can be allocated in GPU
        self.MAX_NS_COUNT = MAX_NS_COUNT  # MAX number of next states for a single state action pair
        self.curr_vi_error = float("inf")  # Backup error
        self.e_curr_vi_error = float("inf")  # exploration Backup error
        self.s_curr_vi_error = float("inf")  # safe  Backup error

        # MDP Parameters
        self.tC = defaultdict(init2zero_def_def_dict)  # Transition Counts
        self.rC = defaultdict(init2zero_def_def_dict)  # Reward Counts
        self.tD = defaultdict(init2zero_def_def_dict)  # Transition Probabilities
        self.rD = init2zero_def_def_dict()  # init2def_def_dict() # Reward Expected
        self.ur = ur
        self.A = A
        self.weight_transitions = mdp_params["weight_transitions"]
        self.default_mode = default_mode

        # MDP GPU Parameters
        self.s2idx = {s: i for i, s in enumerate(self.tD)}
        self.idx2s = {i: s for i, s in enumerate(self.tD)}
        self.a2idx = {s: i for i, s in enumerate(self.A)}
        self.idx2a = {i: s for i, s in enumerate(self.A)}

        # self.tranCountMatrix_cpu = np.zeros((len(self.A), self.MAX_S_COUNT, self.MAX_NS_COUNT)).astype('float32')
        self.tranProbMatrix_cpu = np.zeros((len(self.A), self.MAX_S_COUNT, self.MAX_NS_COUNT)).astype('float32')
        self.tranidxMatrix_cpu = np.zeros((len(self.A), self.MAX_S_COUNT, self.MAX_NS_COUNT)).astype('float32')
        self.rewardMatrix_cpu = np.zeros((len(self.A), self.MAX_S_COUNT, self.MAX_NS_COUNT)).astype('float32')

        # self.tranCountMatrix_gpu = gpuarray.to_gpu(np.zeros((len(self.A), self.MAX_S_COUNT, self.MAX_NS_COUNT)).astype('float32'))
        self.tranProbMatrix_gpu = gpuarray.to_gpu(np.zeros((len(self.A), self.MAX_S_COUNT, self.MAX_NS_COUNT)).astype('float32'))
        self.tranidxMatrix_gpu = gpuarray.to_gpu(np.zeros((len(self.A), self.MAX_S_COUNT, self.MAX_NS_COUNT)).astype('float32'))
        self.rewardMatrix_gpu = gpuarray.to_gpu(np.zeros((len(self.A), self.MAX_S_COUNT, self.MAX_NS_COUNT)).astype('float32'))

        # Optimal Policy GPU parameters
        self.vD_gpu = gpuarray.to_gpu(np.zeros((self.MAX_S_COUNT, 1)).astype('float32')) # value vector in gpu
        self.qD_gpu = gpuarray.to_gpu(np.zeros((self.MAX_S_COUNT, len(self.A))).astype('float32')) # q matrix in gpu
        self.vD_cpu = np.zeros((self.MAX_S_COUNT, 1)).astype('float32')
        self.qD_cpu = np.zeros((self.MAX_S_COUNT, len(self.A))).astype('float32')
        self.gpu_backup_counter = 0


        # Exploration Policy GPU parameters
        self.e_vD_gpu = gpuarray.to_gpu(np.zeros((self.MAX_S_COUNT, 1)).astype('float32'))
        self.e_qD_gpu = gpuarray.to_gpu(np.zeros((self.MAX_S_COUNT, len(self.A))).astype('float32'))
        self.e_rewardMatrix_gpu = gpuarray.to_gpu(np.zeros((len(self.A), self.MAX_S_COUNT, self.MAX_NS_COUNT)).astype('float32'))
        self.e_vD_cpu = np.zeros((self.MAX_S_COUNT, 1)).astype('float32')
        self.e_qD_cpu = np.zeros((self.MAX_S_COUNT, len(self.A))).astype('float32')
        self.e_rewardMatrix_cpu = np.zeros((len(self.A), self.MAX_S_COUNT, self.MAX_NS_COUNT)).astype('float32')
        self.e_gpu_backup_counter = 0

        # Safe Policy GPU parameters
        self.s_vD_gpu = gpuarray.to_gpu(np.zeros((self.MAX_S_COUNT, 1)).astype('float32'))
        self.s_qD_gpu = gpuarray.to_gpu(np.zeros((self.MAX_S_COUNT, len(self.A))).astype('float32'))
        self.s_vD_cpu = np.zeros((self.MAX_S_COUNT, 1)).astype('float32')
        self.s_qD_cpu = np.zeros((self.MAX_S_COUNT, len(self.A))).astype('float32')
        self.s_gpu_backup_counter = 0

        # Optimal Policy CPU Parameters
        self.vD = init2zero_def_dict()  # Optimal Value Vector
        self.qD = init2zero_def_def_dict()  # Optimal Q Value Matrix
        self.pD = {}  # Optimal Policy Vector

        # Exploration Policy CPU Parameters
        self.e_vD = init2zero_def_dict()  # Exploration Value Vector
        self.e_qD = init2zero_def_def_dict()  # Exploration Q Value Matrix\
        self.e_pD = {}  # Exploration Policy Vector
        self.e_rD = init2zero_def_def_dict()  # exploration reward to check which state is visited.

        # Safe Policy CPU Parameters
        self.s_vD = init2zero_def_dict()  # Exploration Value Vector
        self.s_qD = init2zero_def_def_dict()  # Exploration Q Value Matrix\
        self.s_pD = {}  # Exploration Policy Vectorup

        # Policy Search , nn parameters
        self._total_calls = 0
        self._nn_calls = 0

        # Contraction Operator Data structures
        self.reversetD = defaultdict(init2zero_def_def_dict)

        # Track fully unknown_states
        self._fully_unknown_states = {}
        self._initialize_end_and_unknown_state()

        # self._update_nn_kd_tree()
        # self._update_nn_kd_with_action_tree()

    @property
    def missing_state_action_count(self):
        return sum([1 for s in self.rD for a in self.rD[s] if self.ur == self.rD[s][a]])

    @property
    def total_state_action_count(self):
        return len(self.tD)*len(self.A)

    @property
    def missing_state_action_percentage(self):
        return round(self.missing_state_action_count/self.total_state_action_count, 4)

    @property
    def rmax_state_action_count(self):
        return sum([1 for s in self.rD for a in self.rD[s] if self.vi_params["rmax_reward"] >= self.rD[s][a]])

    @property
    def valueDict(self):
        return {s:float(self.vD_cpu[i]) for s,i in self.s2idx.items()}

    @property
    def s_valueDict(self):
        return {s: float(self.s_vD_cpu[i]) for s, i in self.s2idx.items()}

    @property
    def qvalDict(self):
        return {s: {a:self.qD_cpu[i][j] for a,j in self.a2idx.items()} for s, i in self.s2idx.items()}

    @property
    def s_qvalDict(self):
        return {s: {a: self.s_qD_cpu[i][j] for a, j in self.a2idx.items()} for s, i in self.s2idx.items()}

    @property
    def polDict(self):
        qval = cpy(self.qvalDict)
        return {s: max(qval[s], key=qval[s].get) for s in qval}

    @property
    def s_polDict(self):
        qval = cpy(self.s_qvalDict)
        return {s: max(qval[s], key=qval[s].get) for s in qval}

    @property
    def is_kd_tree_initialized(self):
        return self.KDTree is not None and None not in self.KDActionTrees.values()
    def _initialize_end_and_unknown_state(self):
        """
        Initializes the MDP With "end_state" and a "unknown_state"
        "end_state": Abstract state for all terminal State
        "unknown_state": Abstract target state for all undefined State action pairs
        """
        for a in self.A:
            self.consume_transition(["unknown_state", a, "unknown_state", 0, 0])  # [s, a, ns, r, d]
            self.consume_transition(["end_state", a, "end_state", 0, 1])  # [s, a, ns, r, d]

    def update_mdp_for(self, s, a):
        """
        updates transition probabilities as well as reward as per the transition counts for the passed state action pair
        """

        self.tD[s][a] = init2zero_def_dict()
        #         self.update_state_indexes()

        for ns_ in self.tC[s][a]:
            self.tD[s][a][ns_] = self.tC[s][a][ns_] / sum(self.tC[s][a].values()) if self.weight_transitions else 1 / len(self.tC[s][a])
            self.rD[s][a] = sum(self.rC[s][a].values()) / sum(self.tC[s][a].values())
            # self.rd[s][a][ns_] = self.rc[s][a][ns_] / sum(self.tc[s][a][ns_])  # for state action nextstate

        self.update_mdp_matrix_at_cpu_for(s,a)


    def contraction_operation(self):
        # get top 20 nearset for all states.
        # Makes sure no unknown_state_action is present
        # make a list of top 20 state
        return

    def hash_state_action(self,s,a):
        return tuple([s,a])

    def unhash_state_action(self,sa):
        s,a = sa
        return s, a

    def get_kernel_probs(self, knn_dist_dict, delta =None):
        #todo Add a choice to do exponential averaging here.
        delta = delta or self.knn_delta
        all_knn_kernels = {nn: 1 / (dist + delta) for nn, dist in knn_dist_dict.items()}
        all_knn_probs = {nn: knn_kernel / sum(all_knn_kernels.values()) for nn, knn_kernel in all_knn_kernels.items()}
        return all_knn_probs


    def fill_knn_for_all_sa_pairs(self, within_radius = None):
        """
        :param use_cache:  Generaly put false because we will cache things using this
        :param just_known:
        :return:
        """
        to_update_sa_pairs = [(s,a) for s in self.tC for a in self.A if s not in self.omit_list]

        for s, a in tqdm(to_update_sa_pairs):
            self.fill_knn_for_s_a(s, a, within_radius=within_radius)

    # def fill_knn_within_radius(self,s,a,k=None, radius = None):
    #     # Switch to default params if not provided
    #     k = k or self.def_k
    #     radius = radius or self.def_radius
    #
    #     # Do nothing if the KDTree has not been initialized yet
    #     if not self.known_sa_list:
    #         print("KDTree has not been initialized, skipping the fill k nearest neighbor step")
    #         return
    #
    #     # Define and trim nn dict with distances,
    #     hsa = self.hash_state_action(s, a)
    #     knn_hsa = self._get_knn_hs_kd_with_action_tree(hsa, k=k)
    #
    #     for hsa, dist in knn_hsa.items():
    #         hs,a = self.unhash_state_action(hsa)
    #         if radius > dist:
    #             for ns in self.tD[hs][a]:
    #                 self.consume_transition((hs,a,ns,r,d))


    def fill_knn_for_s_a(self, s, a, k=None, within_radius = None):
        # Switch to default params if not provided
        k = k or self.mdp_build_k

        # Do nothing if the KDTree has not been initialized yet
        if not self.known_sa_list:
            print("KDTree has not been initialized, skipping the fill k nearest neighbor step")
            return

        # Define and trim nn dict with distances,
        hsa = self.hash_state_action(s, a)
        knn_hsa = self._get_knn_hs_kd_with_action_tree(hsa, k=k)
        if within_radius is not None and min(knn_hsa.values()) > within_radius :
            return

        # first Calculate R(s,a)
        all_knn_probs_ = self.get_kernel_probs(knn_hsa, delta=self.knn_delta)
        approx_reward = 0
        all_rewards = []
        for hsa, prob in all_knn_probs_.items():
            s_, a_ = self.unhash_state_action(hsa)
            approx_reward += prob * sum(self.known_rC[s_][a_].values()) / sum(self.known_tC[s_][a_].values())
            all_rewards.append(sum(self.known_rC[s_][a_].values()) / sum(self.known_tC[s_][a_].values()))
        # approx_reward = min(all_rewards)
        # self.cache_and_check(hsa, knn_hsa, incremental_update  = incremental_update)

        if self.calc_action_vector:
            self.process_knn_dict_from_action_vectors(knn_hsa, s, a, approx_reward)
        else:
            self.process_knn_dict_from_states(knn_hsa,s,a, approx_reward)


    def fill_fully_unknown_states(self):
        for s in tqdm(self._fully_unknown_states):
            for a in self.A:
                self.fill_knn_for_s_a(s,a)


    def process_knn_dict_from_action_vectors(self, knn_hsa ,s, a, approx_reward):
        # reset counter and remember transition reward
        self.reset_counts_for_sa(s,a)

        action_vec_list, end_state_transition = [], False
        for hsa in knn_hsa:
            for ns in self.known_tC[self.unhash_state_action(hsa)[0]][self.unhash_state_action(hsa)[1]]:
                if ns != "end_state":
                    action_vec_list.append(np.array(np.array(ns) - np.array(self.unhash_state_action(hsa)[0])))
                else:
                    end_state_transition = True

        if not end_state_transition:
            avg_action_vec = np.mean(action_vec_list, axis=0)  #if weighted_mean else np.mean(action_vec_list, axis=0)
            approx_ns = np.array(s) + avg_action_vec
            knn_hs = self._get_knn_hs_kdtree(tuple(approx_ns.squeeze()), k = self.mdp_build_k)
            all_knn_probs = self.get_kernel_probs(knn_hs, delta=self.knn_delta)

            if self.penalize_uncertainity:
                approx_reward = approx_reward if (s, a) in self.known_sa_dict \
                    else approx_reward - self.penalty_beta * np.mean(list(knn_hs.values()))

            # add respective counts
            for hs_, prob in all_knn_probs.items():
                if prob >0.01:
                    
                    
                    
                    self.tC[s][a][hs_] += int(prob * 100)
                    self.rC[s][a][hs_] += int(prob * 100) * approx_reward

        else:
            self.tC[s][a]["end_state"] = 1
            self.rC[s][a]["end_state"] = approx_reward

        self.filter_sa_count_for_max_ns_count(s,a)

        # update the probability ditribution
        self.update_mdp_for(s, a)

        assert 1.1 > sum(self.tD[s][a].values()) > 0.9


    def process_knn_dict_from_states(self, knn_hsa ,s, a,approx_reward):
        knn_hs = {self.unhash_state_action(hsa)[0]:dist for hsa, dist in knn_hsa.items()}
        all_knn_probs = self.get_kernel_probs(knn_hs, delta=self.knn_delta)

        # reset counter and remember transition reward
        self.reset_counts_for_sa(s,a)

        # add respective counts
        for hs_, prob in all_knn_probs.items():
            assert  len(self.known_tC[hs_][a]) != 0
            for ns in self.known_tC[hs_][a]:
                if prob >0.01:
                    self.tC[s][a][ns] += int( prob * 100 * self.known_tC[hs_][a][ns])
                    self.rC[s][a][ns] += int( prob * 100) * approx_reward

        self.filter_sa_count_for_max_ns_count(s,a)

        # update the probability ditribution
        self.update_mdp_for(s, a)

        assert 1.1 > sum(self.tD[s][a].values()) > 0.9


    def filter_sa_count_for_max_ns_count(self,s,a):
        # If the branching factor is too large remove the least occuring transition #Todo Set this as parameter
        while len(self.tC[s][a]) > self.MAX_NS_COUNT:
            self.delete_tran_from_counts(s, a, min(self.tC[s][a], key=self.tC[s][a].get))

    def reset_counts_for_sa(self,s,a):
        self.tC[s][a] = init2zero_def_dict()
        self.rC[s][a] = init2zero_def_dict()


    def sync_mdp_from_cpu_to_gpu(self,):
        # self.tranCountMatrix_gpu.gpudata.free()
        try:
            self.tranProbMatrix_gpu.gpudata.free()
            self.tranidxMatrix_gpu.gpudata.free()
            self.rewardMatrix_gpu.gpudata.free()
            self.e_rewardMatrix_gpu.gpudata.free()
        except:
            print("free failed")

        # self.tranCountMatrix_gpu = gpuarray.to_gpu(self.tranCountMatrix_cpu)
        self.tranProbMatrix_gpu = gpuarray.to_gpu(self.tranProbMatrix_cpu)
        self.tranidxMatrix_gpu = gpuarray.to_gpu(self.tranidxMatrix_cpu)
        self.rewardMatrix_gpu = gpuarray.to_gpu(self.rewardMatrix_cpu)
        self.e_rewardMatrix_gpu = gpuarray.to_gpu(self.e_rewardMatrix_cpu)


    def update_mdp_matrix_at_cpu_for(self, s, a):
        """
        updates GPU matrix for particular state action
        """
        for i, a in [(self.a2idx[a], a)]:
            for j, s in [(self.s2idx[s], s)]:
                self.tranProbMatrix_cpu[i][j] = np.array([self.tD[s][a][ns] for ns in self.tD[s][a]] + [0] * (self.MAX_NS_COUNT - len(self.tD[s][a]))).astype("float32")
                self.tranidxMatrix_cpu[i][j] = np.array([self.s2idx[ns] for ns in self.tD[s][a]] + [self.s2idx["unknown_state"]] * (self.MAX_NS_COUNT - len(self.tD[s][a]))).astype("float32")
                self.rewardMatrix_cpu[i][j] = np.array([self.rD[s][a]] * self.MAX_NS_COUNT).astype("float32")
                self.e_rewardMatrix_cpu[i][j] = np.array([self.get_rmax_reward_logic(s, a)] * self.MAX_NS_COUNT).astype("float32")
                assert len(self.tranProbMatrix_cpu[i][j]) == len(self.tranidxMatrix_cpu[i][j])


    def seed_for_new_state(self, s):
        """
        Checks if the state is not in the MDP, if so seeds with undefined state actions
        :param s: new state
        """

        if s not in self.tC:
            self._fully_unknown_states[s]=1
            curr_idx = len(self.tC)
            self.s2idx[s] = curr_idx
            self.idx2s[curr_idx] = s

            #check if current index is bigger than the capacity of the Transition Buffer
            u_ns = "end_state" if s ==  "end_state" else "unknown_state"
            ur = 0 if s in self.omit_list else self.ur
            for a in self.A:
                if a not in self.tC[s]:
                    self.tC[s][a][u_ns] = 1
                    self.rC[s][a][u_ns] = ur
                    self.update_mdp_for(s, a)

                # Fill unknown logic
                if s not in self.omit_list:
                    self.unknown_sa_dict[self.hash_state_action(s, a)] = 1


    def delete_tran_from_counts(self, s, a, ns):
        """
        deletes the given transition from the MDP
        """
        if s in self.tC and a in self.tC[s] and ns in self.tC[s][a]:
            del self.tC[s][a][ns]
            del self.rC[s][a][ns]

    def consume_transition(self, tran):
        """
        Adds the transition in the MDP
        """
        assert len(tran) == 5
        assert len(tran) == 5


        s, a, ns, r, d = tran
        ns = "end_state" if d else ns
        
        if s not in self.omit_list:
            self.known_tC[s][a][ns] += 1
            self.known_rC[s][a][ns] += r

        # if the next state is not in the MDP add a dummy transitions to unknown state for all state actions
        self.seed_for_new_state(s)
        self.seed_for_new_state(ns)

        # set to_commit stage
        for s_,a_ in [(s_,a_) for s_ in [s,ns] for a_ in self.A if s_ not in self.omit_list]:
            self.to_commit_sa_dict[self.hash_state_action(s_,a_)] = 1

        # delete seeded transition once an actual state action set is encountered
        self.delete_tran_from_counts(s, a, "unknown_state")
        self._fully_unknown_states.pop(s, "None")

        if s not in self.omit_list:
            self.unknown_sa_dict.pop(self.hash_state_action(s, a), None)
            self.known_sa_dict[self.hash_state_action(s,a)] = 1

        self.tC[s][a][ns] += 1
        self.rC[s][a][ns] += r

        self.filter_sa_count_for_max_ns_count(s, a)
        self.update_mdp_for(s,a)


    def fill_knn_from_to_commit_dict(self, update_kd_trees = True):
        if update_kd_trees:
            st = time.time()
            self._update_nn_kd_tree()
            self._update_nn_kd_with_action_tree()
            et = time.time()
            self.fill_engine_time.append(et - st)

        for hsa in tqdm(self.to_commit_sa_dict):
            s,a = self.unhash_state_action(hsa)
            self.fill_knn_for_s_a(s, a, False)

        assert self.unknown_state_action_count == 0
        self.to_commit_sa_dict = {}

    def get_rmax_reward_logic(self, s, a):
        # get the sum of distances for k nearest neighbor
        # pick the top 10%
        # set all unknwon sa for this s as rmax  actions as rmax.
        return 0 #todo add some logic here

        # if not self.known_sa_list:
        #     return 0
        #
        # if self.hash_state_action(s,a) in self.known_sa_list:
        #     return 0
        #
        # knn_hs = self._get_knn_hs_kdtree(s, k = self.def_k)
        # if sum(knn_hs.values()) > min(self.top_quantile_of_distances):
        #     return 1000
        # else:
        #     return 0

        # sa_count = sum(self.tC[s][a].values())
        # linearly_decreasing_rmax = self.vi_params["rmax_reward"] * (self.vi_params["rmax_thres"] - sa_count)
        # exponentially_decreasing_rmax = 100 * (math.e ** (int(-0.01 * sa_count)))
        # if s in ["end_state", "unknown_state"]:
        #     rmax_reward = 0
        # else:
        #     if self.vi_params["balanced_explr"]:
        #         rmax_reward = linearly_decreasing_rmax if sa_count < self.vi_params["rmax_thres"] \
        #             else exponentially_decreasing_rmax
        #     else:
        #         rmax_reward = linearly_decreasing_rmax if sa_count < 10 else self.rD[s][a]
        #
        # return rmax_reward

    def sync_opt_val_vectors_from_GPU(self):
        tmp_vD_cpu = self.vD_gpu.get()
        tmp_qD_cpu = self.qD_gpu.get()
        self.vD_cpu = tmp_vD_cpu
        self.qD_cpu = tmp_qD_cpu

    def sync_explr_val_vectors_from_GPU(self):
        tmp_e_vD_cpu = self.e_vD_gpu.get()
        tmp_e_qD_cpu = self.e_qD_gpu.get()
        self.e_vD_cpu = tmp_e_vD_cpu
        self.e_qD_cpu = tmp_e_qD_cpu

    def sync_safe_val_vectors_from_GPU(self):
        tmp_s_vD_cpu = self.s_vD_gpu.get()
        tmp_s_qD_cpu = self.s_qD_gpu.get()
        self.s_vD_cpu = tmp_s_vD_cpu
        self.s_qD_cpu = tmp_s_qD_cpu


    def convert_matrix_to_dicts(self):
        assert False


    def do_backup(self, mode, module, n_backups):
        bkp_fxn_dict = {"optimal": {"CPU": self.opt_bellman_backup_step_cpu, "GPU": self.opt_bellman_backup_step_gpu},
                "safe": {"CPU": self.safe_bellman_backup_step_cpu, "GPU": self.safe_bellman_backup_step_gpu},
                "exploration": {"CPU": self.explr_bellman_backup_step_cpu,"GPU": self.explr_bellman_backup_step_gpu}}
        sync_fxn_dict= {"optimal": {"GPU": self.sync_opt_val_vectors_from_GPU},
                "safe": {"GPU": self.sync_safe_val_vectors_from_GPU},
                "exploration": {"GPU": self.sync_explr_val_vectors_from_GPU}}

        if mode == "CPU":
            for _ in range(n_backups):
                bkp_fxn_dict[module]["CPU"]()
        elif mode == "GPU":
            self.sync_mdp_from_cpu_to_gpu()
            for _ in range(n_backups):
                bkp_fxn_dict[module]["GPU"]()
            sync_fxn_dict[module]["GPU"]()
        else:
            print("Illegal Mode: Not Specified")
            assert False

    def do_optimal_backup(self, mode="CPU", n_backups=1):
        self.do_backup(mode=mode, module="optimal", n_backups=n_backups)

    def do_safe_backup(self, mode="CPU", n_backups=1):
        self.do_backup(mode=mode, module="safe", n_backups=n_backups)

    def do_explr_backup(self, mode="CPU", n_backups=1):
        self.do_backup(mode=mode, module="exploration", n_backups=n_backups)




    #### Policy Functions ####
    def sample_action_from_qval_dict(self,qval_dict):
        return random.choices(list(qval_dict.keys()), list(qval_dict.values()), k=1)[0]

    def get_action_from_q_matrix(self, hs, qMatrix, smoothing = False, soft_q = False, weight_nn = False):
        if smoothing:
            if self.smooth_with_seen:
                qval_dict = {}
                for a in self.A:
                    knn_hsa = self._get_knn_hs_kd_with_action_tree(self.hash_state_action(hs, a), k=self.plcy_lift_k)
                    qval_a = np.mean([qMatrix[self.s2idx[nn_hs]][self.a2idx[a_]] for nn_hs , a_ in knn_hsa])
                    qval_dict[a] =  qval_a
            else:
                knn_hs = self._get_knn_hs_kdtree(hs, k=self.plcy_lift_k)
                if weight_nn:
                    knn_hs_normalized = self.get_kernel_probs(knn_hs, delta=self.knn_delta)
                    all_qval_dict = [{self.idx2a[i]: qval*norm_dist for i, qval in enumerate(qMatrix[self.s2idx[nn_hs]])} for nn_hs, norm_dist in knn_hs_normalized.items()]
                    qval_dict = {a:np.sum([_qval_dict[a] for _qval_dict in all_qval_dict]) for a in self.A}
                else:
                    all_qval_dict = [{self.idx2a[i]: qval for i, qval in enumerate(qMatrix[self.s2idx[nn_hs]])} for nn_hs in knn_hs]
                    qval_dict = {a: np.mean([_qval_dict[a] for _qval_dict in all_qval_dict]) for a in self.A}

        else:
            nn_hs = self._get_nn_hs_kdtree(hs)
            qval_dict = {self.idx2a[i]:qval for i, qval in enumerate(qMatrix[self.s2idx[nn_hs]])}

        if soft_q:
            return self.sample_action_from_qval_dict(qval_dict)
        else:
            return max(qval_dict, key=qval_dict.get)

    def get_opt_action(self, hs ,smoothing = False, soft_q = False, weight_nn = False):
        return self.get_action_from_q_matrix(hs,self.qD_cpu, smoothing=smoothing, soft_q=soft_q, weight_nn=weight_nn)

    def get_safe_action(self, hs, smoothing = False, soft_q = False, weight_nn = False):
        return self.get_action_from_q_matrix(hs,self.s_qD_cpu,  smoothing=smoothing, soft_q=soft_q, weight_nn=weight_nn)

    def get_explr_action(self, hs, smoothing = False, soft_q = False, weight_nn = False):
        return self.get_action_from_q_matrix(hs, self.e_qD_cpu, smoothing=smoothing, soft_q=soft_q, weight_nn=weight_nn)

    def get_state_count(self):
        return len(self.s2idx)

    def _update_nn_kd_tree(self):
        state_list = [s for s in self.tD if s not in self.omit_list]
        if not state_list:
            return

        self.state_list = state_list
        self.KDTree = KDTree(np.array(state_list), leaf_size=40)

    def _update_nn_kd_with_action_tree(self):
        # print("KD Tree Updated")
        self.known_sa_list = list(self.known_sa_dict.keys())
        if not self.known_sa_list:
            return
        self.factored_known_sa_list = {
            a: [self.unhash_state_action(hsa)[0] for hsa in self.known_sa_list if self.unhash_state_action(hsa)[1] == a]
            for a in self.A}

        self.KDActionTrees = {a: KDTree(np.array(st_list), leaf_size=40) for a, st_list in self.factored_known_sa_list.items()}

    def _get_nn_hs_kd_with_action_tree(self, hsa ):
        hs, a = self.unhash_state_action(hsa)
        if hs in self.omit_list:
            return hsa

        nn_dist, nn_idx = self.KDActionTrees[a].query(np.array([hs]), k=1)
        nearest_neighbor_hs = self.factored_known_sa_list[a][int(nn_idx.squeeze())]
        nearest_neighbor_hsa = self.hash_state_action(nearest_neighbor_hs,a)
        return nearest_neighbor_hsa

    def _get_knn_hs_kd_with_action_tree(self, hsa,k):
        hs, a = self.unhash_state_action(hsa)
        if hs in self.omit_list or not self.factored_known_sa_list:
            return {hsa:0}

        nn_dist, nn_idx = self.KDActionTrees[a].query(np.array([hs]), k=k)
        nn_dist, nn_idx = nn_dist.reshape(-1), nn_idx.reshape(-1)
        nn_dict = {self.hash_state_action(self.factored_known_sa_list[a][int(idx)], a): nn_dist[i]
                   for i, idx in enumerate(nn_idx)}
        return nn_dict

    def _get_nn_hs_kdtree(self, hs, return_dist = False):
        nn_hs, nn_dist = list(self._get_knn_hs_kdtree(hs, k=1).items())[0]
        if return_dist:
            return nn_hs, nn_dist
        else:
            return nn_hs

    def _get_knn_hs_kdtree(self, hs, k):
        if hs in self.omit_list or not self.state_list:
            return {hs:0}

        nn_dist, nn_idx = self.KDTree.query(np.array([hs]), k=k)
        nn_dist, nn_idx = nn_dist.reshape(-1), nn_idx.reshape(-1)
        nn_dict = {self.state_list[int(idx)]: nn_dist[i] for i, idx in enumerate(nn_idx)}
        return nn_dict


    def _get_radial_nn_hs_kdtree(self, hs, radius):
        nn_idxs, dist = self.KDTree.query_radius(np.array([hs]), r=radius,
                                                       return_distance=True, sort_results=True)
        nn_idxs, dist = nn_idxs[0], dist[0]
        nn_dict = {self.state_list[indx]:dist[i] for i , indx in enumerate(nn_idxs)}

        return nn_dict


    def solve(self, eps = 1e-5, mode = None, safe_bkp = False, explr_bkp = False, verbose = True):
        mode = mode or self.default_mode

        st = time.time()
        curr_error = self.curr_vi_error
        while self.curr_vi_error > eps:
            self.do_optimal_backup(mode=mode, n_backups=250)
            if safe_bkp:
                self.do_safe_backup(mode = mode, n_backups=250)
            if explr_bkp:
                self.do_explr_backup(mode = mode, n_backups=250)

            if self.curr_vi_error < curr_error/10 and verbose:
                print("Elapsed Time:{}s, VI Error:{}, #Backups: {}".format(int(time.time()-st), round(self.curr_vi_error,8), self.gpu_backup_counter))
                curr_error = self.curr_vi_error
        et = time.time()
        print("Time takedn to solve",et-st)


    @property
    def  qval_distribution(self):
        qvalDict = cpy(self.qvalDict)
        return [qsa for qs in  qvalDict for qsa in qs]

    @property
    def unknown_state_count(self):
        return sum([self.check_if_unknown(s) for s in self.tD])

    @property
    def unknown_state_action_count(self):
        return sum([self.check_if_unknown(s,a) for s in self.tD for a in self.tD[s]])

    @property
    def fully_unknown_state_count(self):
        return sum([all([self.check_if_unknown(s,a) for a in self.tD[s]]) for s in self.tD ])

    @property
    def fully_unknown_states(self):
        return [s for s in self.tD if all([self.check_if_unknown(s,a) for a in self.tD[s]])]

    def check_if_unknown(self, s, a = None):
        if s == "unknown_state":
            return False
        if a is not None:
            return "unknown_state" in self.tD[s][a]
        else:
            return sum([1 for a in self.A if "unknown_state" in self.tD[s][a]]) == len(self.A)

    def get_seen_action_count(self, s):
        return sum([1 for a in self.tC[s] if "unknown_state" not in self.tC[s][a]])

    def get_explored_action_count(self,s):
        return sum([1 for a in self.tC[s] if sum(self.tC[s][a].values())> self.vi_params["rmax_thres"]])

    @property
    def tran_prob_distribution(self):
        return [self.tD[s][a][ns] for s in self.tD for a in self.tD[s] for ns in self.tD[s][a]]

    @property
    def reward_distribution(self):
        return [self.rD[s][a] for s in self.rD for a in self.rD[s]]

    @property
    def state_action_count_distribution(self):
        return [sum(self.tC[s][a].values()) for s in self.tC for a in self.tC[s]]

    @property
    def state_action_fan_out_distribution(self):
        return [len(self.tD[s][a]) for s in self.tD for a in self.tD[s]]

    @property
    def state_action_fan_in_distribution(self):
        list_of_ns = [ns for s in self.tD for a in self.tD[s] for ns in self.tD[s][a] if ns not in self.omit_list]
        counter = Counter(list_of_ns)
        return list(counter.values())
        # count = defaultdict(init2zero)
        # for s in self.tD:
        #     for a in self.tD[s]:
        #         for ns in self.tD[s][a]:
        #             if ns in self.omit_list:
        #                 continue
        #             count[ns] += 1
        # state_action_in_distr = list(count.values())
        # return state_action_in_distr

    @property
    def explored_action_count_distribution(self):
        return [sum([1 for a in self.tC[s] if sum(self.tC[s][a].values()) > self.vi_params["rmax_thres"]]) for s in self.tC]

    @property
    def seen_action_count_distribution(self):
        return [sum([1 for a in self.tC[s] if "unknown_state" not in self.tC[s][a]]) for s in self.tC]

    @property
    def state_count_distribution(self):
        return [ sum([sum(self.tC[s][a].values()) for a in self.tC[s]]) for s in self.tC ]

    @property
    def self_loop_prob_distribution(self):
        return [self.tD[s][a][ns] for s in self.tD for a in self.tD[s] for ns in self.tD[s][a] if s == ns]

    @property
    def self_loop_count(self):
        return sum([1 for s in self.tD for a in self.tD[s] for ns in self.tD[s][a] if s == ns])

    @property
    def end_state_state_action_count(self):
        return sum([1 for s in self.tD for a in self.tD[s] if "end_state" in self.tD[s][a]])

    @staticmethod
    def get_bisimilarity_distance(m1, m2, disc1, disc2, dataset):
        """

        :param m1: Ground Abstraction
        :param m2: Current Choice of Abstraction
        :param disc1: h1()
        :param disc2: h2()
        :param dataset:
        :return:
        """
        transition_error = 0
        reward_error = 0
        approximation_error = 0
        estimation_error = 0

        _batch_size = 256

        start_end_indexes = get_iter_indexes(len(dataset.buffer), _batch_size)
        processed_transitions = []
        for start_i, end_i in tqdm(start_end_indexes):
            batch, info = dataset.sample_indices(list(range(start_i, end_i)))
            batch_s, batch_a, batch_ns, batch_r, batch_d = batch

            m1_batch_s_d, m1_batch_ns_d = disc1(batch_s), disc1(batch_ns)
            m2_batch_s_d, m2_batch_ns_d = disc2(batch_s), disc2(batch_ns)

            for i in range(len(m1_batch_s_d)):
                m1_hs, a, m1_hns = hAsh(m1_batch_s_d[i]), int(batch_a[i][0]), hAsh(m1_batch_ns_d[i])
                m2_hs, a, m2_hns = hAsh(m2_batch_s_d[i]), int(batch_a[i][0]), hAsh(m2_batch_ns_d[i])
                if (m1_hs, a, m1_hns) not in processed_transitions and m1_hns != "unknown_state":
                    processed_transitions.append((m1_hs, a, m1_hns))
                    transition_error += abs(m1.tD[m1_hs][a][m1_hns] - m2.tD[m2_hs][a][m2_hns]) # the first term is mostly 1, second term is varying.
                    reward_error += abs(m1.rD[m1_hs][a][m1_hns] - m2.rD[m2_hs][a][m2_hns]) # the first term is mostly deterministic
                else:
                    continue

        # approximation_error
        return transition_error


    def opt_bellman_backup_step_cpu(self):
        backup_error = 0
        for s in self.tD:
            for a in self.A:
                expected_ns_val = sum(self.tD[s][a][ns] * self.vD[ns] for ns in self.tD[s][a])
                self.qD[s][a] = self.rD[s][a] + self.vi_params["gamma"] * expected_ns_val

            new_val = max(self.qD[s].values())

            backup_error = max(backup_error, abs(new_val - self.vD[s]))
            self.vD[s] = new_val
            self.pD[s] = max(self.qD[s], key=self.qD[s].get)

        self.curr_vi_error = backup_error

    def explr_bellman_backup_step_cpu(self):
        backup_error = 0
        for s in self.tD:
            for a in self.A:
                self.e_rD[s][a] = self.get_rmax_reward_logic(s, a)
                self.e_qD[s][a] = self.e_rD[s][a] + self.vi_params["gamma"] * sum(
                    self.tD[s][a][ns] * self.e_vD[ns] for ns in self.tD[s][a])

            new_val = max(self.e_qD[s].values())

            backup_error = max(backup_error, abs(new_val - self.e_vD[s]))
            self.e_vD[s] = new_val
            self.e_pD[s] = max(self.e_qD[s], key=self.e_qD[s].get)
        self.e_curr_vi_error = backup_error

    def safe_bellman_backup_step_cpu(self):
        backup_error = 0
        for s in self.tD:
            next_explored_state_val_list = []
            for a in self.A:
                expected_ns_val = sum(self.tD[s][a][ns] * self.s_vD[ns] for ns in self.tD[s][a])
                self.s_qD[s][a] = self.rD[s][a] + self.vi_params["gamma"] * expected_ns_val
                if "unknown_state" not in self.tD[s][a]:
                    next_explored_state_val_list.append(self.s_qD[s][a])
            if next_explored_state_val_list:
                new_val = (1 - self.vi_params["slip_prob"]) * max(self.s_qD[s].values()) + self.vi_params[
                    "slip_prob"] * sum(next_explored_state_val_list) / len(next_explored_state_val_list)
            else:
                new_val = max(self.s_qD[s].values())

            backup_error = max(backup_error, abs(new_val - self.s_vD[s]))
            self.s_vD[s] = new_val
            self.s_pD[s] = max(self.s_qD[s], key=self.s_qD[s].get)
        self.s_curr_vi_error = backup_error


    def opt_bellman_backup_step_gpu(self):
        # Temporary variables
        ACTION_COUNT, ROW_COUNT, COL_COUNT = self.tranProbMatrix_gpu.shape
        MATRIX_SIZE = mth.ceil(mth.sqrt(ROW_COUNT))
        BLOCK_SIZE = 16

        # get the kernel code from the template
        kernel_code = vi_kernel_template % {
            'ROW_COUNT': ROW_COUNT,
            'COL_COUNT': COL_COUNT,
            'ACTION_COUNT': ACTION_COUNT,
            'MATRIX_SIZE': MATRIX_SIZE,
            'GAMMA': self.vi_params["gamma"],
            'SLIP_ACTION_PROB': 0,
            'RMIN': self.vi_params["rmin"]
        }

        # Get grid dynamically by specifying the constant MATRIX_SIZE
        if MATRIX_SIZE % BLOCK_SIZE != 0:
            grid = (MATRIX_SIZE // BLOCK_SIZE + 1, MATRIX_SIZE // BLOCK_SIZE + 1, 1)
        else:
            grid = (MATRIX_SIZE // BLOCK_SIZE, MATRIX_SIZE // BLOCK_SIZE, 1)

        # compile the kernel code and get the compiled module
        mod = compiler.SourceModule(kernel_code)
        matrixmul = mod.get_function("MatrixMulKernel")

        # Empty initialize Target Value and Q vectors
        tgt_vD_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, 1)).astype("float32"))
        tgt_qD_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, ACTION_COUNT)).astype("float32"))
        tgt_error_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, 1)).astype("float32"))

        try:
            matrixmul(
                # inputs
                self.tranProbMatrix_gpu, self.tranidxMatrix_gpu, self.rewardMatrix_gpu, self.vD_gpu,
                # output
                tgt_vD_gpu, tgt_qD_gpu, tgt_error_gpu,
                grid=grid,
                # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
                block=(BLOCK_SIZE, BLOCK_SIZE, 1)
            )
        except:
            if (input("d for debugging") == 'd'):
                print(BLOCK_SIZE, BLOCK_SIZE, 1)
                import pdb;
                pdb.set_trace()

        self.vD_gpu.gpudata.free()
        self.qD_gpu.gpudata.free()
        self.vD_gpu = tgt_vD_gpu
        self.qD_gpu = tgt_qD_gpu

        self.gpu_backup_counter += 1
        if (self.gpu_backup_counter + 1) % 25 == 0:
            # print("checkingggg for epsilng stop")
            max_error_gpu = gpuarray.max(tgt_error_gpu, stream=None)  # ((value_vector_gpu,new_value_vector_gpu)
            max_error = max_error_gpu.get()
            max_error_gpu.gpudata.free()
            self.curr_vi_error = float(max_error)
        tgt_error_gpu.gpudata.free()

    def safe_bellman_backup_step_gpu(self):
        # Temporary variables
        ACTION_COUNT, ROW_COUNT, COL_COUNT = self.tranProbMatrix_gpu.shape
        MATRIX_SIZE = mth.ceil(mth.sqrt(ROW_COUNT))
        BLOCK_SIZE = 16

        # get the kernel code from the template
        kernel_code = vi_kernel_template % {
            'ROW_COUNT': ROW_COUNT,
            'COL_COUNT': COL_COUNT,
            'ACTION_COUNT': ACTION_COUNT,
            'MATRIX_SIZE': MATRIX_SIZE,
            'GAMMA': self.vi_params["gamma"],
            'SLIP_ACTION_PROB': self.vi_params["slip_prob"],
            'RMIN': self.vi_params["rmin"]
        }

        # Get grid dynamically by specifying the constant MATRIX_SIZE
        if MATRIX_SIZE % BLOCK_SIZE != 0:
            grid = (MATRIX_SIZE // BLOCK_SIZE + 1, MATRIX_SIZE // BLOCK_SIZE + 1, 1)
        else:
            grid = (MATRIX_SIZE // BLOCK_SIZE, MATRIX_SIZE // BLOCK_SIZE, 1)

        # compile the kernel code and get the compiled module
        mod = compiler.SourceModule(kernel_code)
        matrixmul = mod.get_function("MatrixMulKernel")

        # Empty initialize Target Value and Q vectors
        tgt_vD_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, 1)).astype("float32"))
        tgt_qD_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, ACTION_COUNT)).astype("float32"))
        tgt_error_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, 1)).astype("float32"))

        try:
            matrixmul(
                # inputs
                self.tranProbMatrix_gpu, self.tranidxMatrix_gpu, self.rewardMatrix_gpu, self.s_vD_gpu,
                # output
                tgt_vD_gpu, tgt_qD_gpu, tgt_error_gpu,
                grid=grid,
                # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
                block=(BLOCK_SIZE, BLOCK_SIZE, 1)
            )
        except:
            if (input("d for debugging") == 'd'):
                print(BLOCK_SIZE, BLOCK_SIZE, 1)
                import pdb;
                pdb.set_trace()

        self.s_vD_gpu.gpudata.free()
        self.s_qD_gpu.gpudata.free()
        self.s_vD_gpu = tgt_vD_gpu
        self.s_qD_gpu = tgt_qD_gpu

        self.s_gpu_backup_counter += 1
        if (self.s_gpu_backup_counter + 1) % 25 == 0:
            max_error_gpu = gpuarray.max(tgt_error_gpu, stream=None)  # ((value_vector_gpu,new_value_vector_gpu)
            max_error = max_error_gpu.get()
            max_error_gpu.gpudata.free()
            self.s_curr_vi_error = float(max_error)
        tgt_error_gpu.gpudata.free()

    def explr_bellman_backup_step_gpu(self):
        # Temporary variables
        ACTION_COUNT, ROW_COUNT, COL_COUNT = self.tranProbMatrix_gpu.shape
        MATRIX_SIZE = mth.ceil(mth.sqrt(ROW_COUNT))
        BLOCK_SIZE = 16

        # get the kernel code from the template
        kernel_code = vi_kernel_template % {
            'ROW_COUNT': ROW_COUNT,
            'COL_COUNT': COL_COUNT,
            'ACTION_COUNT': ACTION_COUNT,
            'MATRIX_SIZE': MATRIX_SIZE,
            'GAMMA': self.vi_params["gamma"],
            'SLIP_ACTION_PROB': 0,
            'RMIN': self.vi_params["rmin"]
        }

        # Get grid dynamically by specifying the constant MATRIX_SIZE
        if MATRIX_SIZE % BLOCK_SIZE != 0:
            grid = (MATRIX_SIZE // BLOCK_SIZE + 1, MATRIX_SIZE // BLOCK_SIZE + 1, 1)
        else:
            grid = (MATRIX_SIZE // BLOCK_SIZE, MATRIX_SIZE // BLOCK_SIZE, 1)

        # compile the kernel code and get the compiled module
        mod = compiler.SourceModule(kernel_code)
        matrixmul = mod.get_function("MatrixMulKernel")

        # Empty initialize Target Value and Q vectors
        tgt_vD_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, 1)).astype("float32"))
        tgt_qD_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, ACTION_COUNT)).astype("float32"))
        tgt_error_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, 1)).astype("float32"))

        try:
            matrixmul(
                # inputs
                self.tranProbMatrix_gpu, self.tranidxMatrix_gpu, self.e_rewardMatrix_gpu, self.e_vD_gpu,
                # output
                tgt_vD_gpu, tgt_qD_gpu, tgt_error_gpu,
                grid=grid,
                # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
                block=(BLOCK_SIZE, BLOCK_SIZE, 1)
            )
        except:
            if (input("d for debugging") == 'd'):
                print(BLOCK_SIZE, BLOCK_SIZE, 1)
                import pdb;
                pdb.set_trace()

        self.e_vD_gpu.gpudata.free()
        self.e_qD_gpu.gpudata.free()
        self.e_vD_gpu = tgt_vD_gpu
        self.e_qD_gpu = tgt_qD_gpu

        self.e_gpu_backup_counter += 1
        if (self.e_gpu_backup_counter + 1) % 25 == 0:
            max_error_gpu = gpuarray.max(tgt_error_gpu, stream=None)  # ((value_vector_gpu,new_value_vector_gpu)
            max_error = float(max_error_gpu.get())
            max_error_gpu.gpudata.free()
            self.e_curr_vi_error = max_error
        tgt_error_gpu.gpudata.free()