from pycuda import compiler, gpuarray
from collections import defaultdict
import time
import math as mth
from bigmdp.mdp.kernels import complex_vi_kernel_code_template as vi_kernel_template
import numpy as np
from tqdm import tqdm
from copy import deepcopy as cpy
from collections import deque
import random
from sklearn.neighbors import KDTree
from collections import Counter
import pycuda.autoinit


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
                            "rmin": -1000},
                 knn_delta=0.01,
                 default_mode="GPU"):

        """
        :param A: Action Space of the MDP
        :param ur: reward for undefined state action pair
        """
        self.omit_list = ["end_state", "unknown_state"]

        self.known_sa_dict = defaultdict(init2zero_def_dict)
        self.KDTree = None
        self.KDActionTrees = {a: None for a in A}
        self.known_sa_list = []
        self.s_list_D = []

        self.consumption_count = 0
        self.knn_delta = knn_delta

        # KNN Params
        self.known_tC = defaultdict(init2zero_def_def_dict)
        self.known_rC = defaultdict(init2zero_def_def_dict)

        # VI CPU/GPU parameters
        self.vi_params = vi_params
        self.MAX_S = MAX_S_COUNT  # Maximum number of state that can be allocated in GPU
        self.MAX_NS = MAX_NS_COUNT  # MAX number of next states for a single state action pair
        self.curr_vi_error = float("inf")  # Backup error
        self.e_curr_vi_error = float("inf")  # exploration Backup error
        self.s_curr_vi_error = float("inf")  # safe  Backup error

        # MDP Parameters
        self.trD = defaultdict(init2zero_def_def_dict)  # Transition reverse dynamics tracker to delete missing links
        self.ur = ur
        self.A = A
        self.default_mode = default_mode

        # MDP dict to matrix api Parameters # i = index
        self.s2i = {"unknown_state": 0, "end_state": 1}
        self.i2s = {0: "unknown_state", 1: "end_state"}
        self.a2i = {a: i for i, a in enumerate(self.A)}
        self.i2a = {i: a for i, a in enumerate(self.A)}
        self.free_i = list(reversed(range(2, self.MAX_S)))
        self.filled_mask = np.zeros((self.MAX_S,)).astype('uint')
        self.filled_mask[[0, 1]] = 1  # set filled for unknown and end state

        # MDP matrices CPU
        m_shape = (len(self.A), self.MAX_S, self.MAX_NS)
        self.tranCountMatrix_cpu = np.zeros(m_shape).astype('float32')
        self.tranidxMatrix_cpu = np.zeros(m_shape).astype('uint')
        self.tranProbMatrix_cpu = np.zeros(m_shape).astype('float32')
        self.rewardCountMatrix_cpu = np.zeros(m_shape).astype('float32')
        self.rewardMatrix_cpu = np.zeros(m_shape).astype('float32')

        # MDP matrices GPU
        self.tranProbMatrix_gpu = gpuarray.to_gpu(np.zeros(m_shape).astype('float32'))
        self.tranidxMatrix_gpu = gpuarray.to_gpu(np.zeros(m_shape).astype('float32'))
        self.rewardMatrix_gpu = gpuarray.to_gpu(np.zeros(m_shape).astype('float32'))

        # Initialize for unknown and end states
        self._initialize_end_and_unknown_state()

        # Help :
        # self.tranCountMatrix_cpu = count Matrix [a_idx, s_idx, ns_id_idx] [Holds tran Counts] ,
        # self.tranidxMatrix_cpu = id Matrix [a_idx, s_idx, ns_id_idx] [Holds ns_idx]
        # self.tranProbMatrix_cpu =  prob Matrix [a_idx, s_idx, ns_id_idx] [Holds tran probabilities]
        # self.rewardCountMatrix_cpu = count Matrix [a_idx, s_idx, ns_id_idx] [Holds reward Counts]
        # self.rewardMatrix_cpu = reward Matrix [a_idx, s_idx, ns_id_idx] [Holds normalized rewards]

        # Optimal Policy  parameters
        self.pD_cpu = np.zeros((self.MAX_S,)).astype('uint')  # Optimal Policy Vector
        self.vD_cpu = np.zeros((self.MAX_S, 1)).astype('float32')
        self.qD_cpu = np.zeros((self.MAX_S, len(self.A))).astype('float32')
        self.vD_gpu = gpuarray.to_gpu(np.zeros((self.MAX_S, 1)).astype('float32'))  # value vector in gpu
        self.qD_gpu = gpuarray.to_gpu(np.zeros((self.MAX_S, len(self.A))).astype('float32'))  # q matrix in gpu
        self.gpu_backup_counter = 0

        # Exploration Policy parameters
        self.e_pD_cpu = np.zeros((self.MAX_S,)).astype('uint')  # Optimal Policy Vector
        self.e_vD_gpu = gpuarray.to_gpu(np.zeros((self.MAX_S, 1)).astype('float32'))
        self.e_qD_gpu = gpuarray.to_gpu(np.zeros((self.MAX_S, len(self.A))).astype('float32'))
        self.e_rewardMatrix_gpu = gpuarray.to_gpu(np.zeros(m_shape).astype('float32'))
        self.e_vD_cpu = np.zeros((self.MAX_S, 1)).astype('float32')
        self.e_qD_cpu = np.zeros((self.MAX_S, len(self.A))).astype('float32')
        self.e_rewardMatrix_cpu = np.zeros(m_shape).astype('float32')
        self.e_gpu_backup_counter = 0

        # Safe Policy GPU parameters
        self.s_pD_cpu = np.zeros((self.MAX_S,)).astype('uint')  # Optimal Policy Vector
        self.s_vD_gpu = gpuarray.to_gpu(np.zeros((self.MAX_S, 1)).astype('float32'))
        self.s_qD_gpu = gpuarray.to_gpu(np.zeros((self.MAX_S, len(self.A))).astype('float32'))
        self.s_vD_cpu = np.zeros((self.MAX_S, 1)).astype('float32')
        self.s_qD_cpu = np.zeros((self.MAX_S, len(self.A))).astype('float32')
        self.s_gpu_backup_counter = 0

        # Policy Search , nn parameters
        self._total_calls = 0
        self._nn_calls = 0

        # Contraction Operator Data structures
        self.reversetD = defaultdict(init2zero_def_def_dict)

        # Track fully unknown_states
        self._fully_unknown_states = {}

        # self._update_nn_kd_tree()
        # self._update_nn_kd_with_action_tree()

        # cached items
        # self.refresh_cache_dicts()

    def _initialize_end_and_unknown_state(self):
        self.tranCountMatrix_cpu[:, :, 0] = 1  # [a_idx, s_idx, ns_id_idx] # everything goes to unknown state
        self.tranProbMatrix_cpu[:, :, 0] = 1  # [a_idx, s_idx, ns_id_idx] # everything goes to unknown state
        self.rewardCountMatrix_cpu[:, :, 0] = self.ur  # [a_idx, s_idx, ns_id_idx] # everything  has ur rewards
        self.rewardMatrix_cpu[:, :, 0] = self.ur  # [a_idx, s_idx, ns_id_idx] # everything  has ur rewards

        self.tranidxMatrix_cpu[:, 0, 0] = 0  # unknown state has a self loop
        self.tranCountMatrix_cpu[:, 0, 0] = 1  # unknown state has a self loop
        self.tranProbMatrix_cpu[:, 0, 0] = 1  # unknown state has a self loop
        self.rewardCountMatrix_cpu[:, 0, 0] = 0  # unknown state self loop has no rewards
        self.rewardMatrix_cpu[:, 0, 0] = 0  # unknown state self loop has no rewards

        self.tranidxMatrix_cpu[:, 1, 0] = 1  # end state has a self loop
        self.tranCountMatrix_cpu[:, 1, 0] = 1  # end state has a self loop
        self.tranProbMatrix_cpu[:, 1, 0] = 1  # end state has a self loop
        self.rewardCountMatrix_cpu[:, 1, 0] = 0  # end state self loop has no rewards
        self.rewardMatrix_cpu[:, 1, 0] = 0  # end state has self loop no rewards

    def refresh_cache_dicts(self):
        self.tD = defaultdict(init2zero_def_def_dict)  # Transition Probabilities
        self.rD = defaultdict(init2zero_def_def_dict)  # Transition Probabilities

        for s, s_i in self.s2i.items():
            for a in self.A:
                for ns_slot, ns_i in enumerate(self.tranidxMatrix_cpu[self.a2i[a], s_i]):
                    ns, a_i = self.i2s[ns_i], self.a2i[a]
                    if self.tranProbMatrix_cpu[a_i, s_i, ns_slot] > 0:
                        # print(f"ns:{ns},ns_i:{ns_i}, a_i:{a_i}, a:{a}, s:{s}, s_i:{s_i} ,  ns_slot:{ns_slot}")
                        self.tD[s][a][ns] = self.tranProbMatrix_cpu[a_i, s_i, ns_slot]
                        self.rD[s][a][ns] = self.rewardMatrix_cpu[a_i, s_i, ns_slot]

    def __len__(self):
        return np.sum(self.filled_mask)

    @property
    def missing_state_action_count(self):
        return sum([1 for s in self.rD for a in self.rD[s] if self.ur == self.rD[s][a]])

    @property
    def total_state_action_count(self):
        return len(self.tD) * len(self.A)

    @property
    def missing_state_action_percentage(self):
        return round(self.missing_state_action_count / self.total_state_action_count, 4)

    @property
    def rmax_state_action_count(self):
        return sum([1 for s in self.rD for a in self.rD[s] if self.vi_params["rmax_reward"] >= self.rD[s][a]])

    @property
    def valueDict(self):
        return {s: float(self.vD_cpu[i]) for s, i in self.s2i.items()}

    @property
    def s_valueDict(self):
        return {s: float(self.s_vD_cpu[i]) for s, i in self.s2i.items()}

    @property
    def qvalDict(self):
        return {s: {a: self.qD_cpu[i][j] for a, j in self.a2i.items()} for s, i in self.s2i.items()}

    @property
    def s_qvalDict(self):
        return {s: {a: self.s_qD_cpu[i][j] for a, j in self.a2i.items()} for s, i in self.s2i.items()}

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

    def get_free_index(self):
        indx = self.free_i.pop()
        self.filled_mask[indx] = 1
        return indx

    def get_ns_i_slot(self, s_i: int, a_i: int, ns_i: int):
        # returns occupied slot if already exists
        # returns new slot if does not exist
        # returns slot_with_least_count if full
        # returns slot, override_flag
        slot_counts = {}
        for slot, _ns_i in enumerate(self.tranidxMatrix_cpu[a_i, s_i]):
            if _ns_i == ns_i:  # i.e. ns_i is already allocated a slot
                return slot, 0
            elif _ns_i == 0:  # i.e. slot not allocated, this slot is free:
                return slot, 1
            else:
                slot_counts[slot] = self.tranCountMatrix_cpu[a_i, s_i, slot]

        # i.e all slots are full
        return min(slot_counts, key=slot_counts.get), 1

    def index_if_new_state(self, s):
        is_new = s not in self.s2i
        if is_new:
            i = self.get_free_index()
            self.s2i[s], self.i2s[i] = i, s
        return is_new

    def consume_transition(self, tran):
        """
        Adds the transition in the MDP
        """
        assert len(tran) == 5
        # pre-process transition
        s, a, ns, r, d = tran
        ns = "end_state" if d else ns
        if s not in self.omit_list:
            self.known_sa_dict[s][a] = 1
            self.known_tC[s][a][ns] += 1
            self.known_rC[s][a][ns] += r

        # Index states and get slot for transition
        self.index_if_new_state(s)
        self.index_if_new_state(ns)
        s_i, a_i, ns_i = self.s2i[s], self.a2i[a], self.s2i[ns]

        # Update MDP with new transition
        slot, override = self.get_ns_i_slot(s_i, a_i, ns_i)
        self.update_count_matrices(s_i, a_i, ns_i, r, slot, override)
        self.update_prob_matrices(s_i, a_i)

    def update_count_matrices(self, s_i, a_i, ns_i, r, slot, override):
        if override:
            self.tranidxMatrix_cpu[a_i, s_i, slot] = ns_i
            self.tranCountMatrix_cpu[a_i, s_i, slot] = 1
            self.rewardCountMatrix_cpu[a_i, s_i, slot] = r
        else:
            self.tranCountMatrix_cpu[a_i, s_i, slot] += 1
            self.rewardCountMatrix_cpu[a_i, s_i, slot] += r

    def update_prob_matrices(self, s_i, a_i):
        # Normalize count Matrix
        self.tranProbMatrix_cpu[a_i, s_i] = self.tranCountMatrix_cpu[a_i, s_i] / np.sum(
            self.tranCountMatrix_cpu[a_i, s_i])
        self.rewardMatrix_cpu[a_i, s_i] = self.rewardCountMatrix_cpu[a_i, s_i] / (
                self.tranCountMatrix_cpu[a_i, s_i] + 1e-12)
        self.e_rewardMatrix_cpu[a_i, s_i] = np.array([self.get_rmax_reward_logic(s_i, a_i)] * self.MAX_NS).astype(
            "float32")
        # assert len(self.tranProbMatrix_cpu[i][j]) == len(self.tranidxMatrix_cpu[i][j])

    def hash_state_action(self, s, a):
        return tuple([s, a])

    def unhash_state_action(self, sa):
        s, a = sa
        return s, a

    def sync_mdp_from_cpu_to_gpu(self, ):
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
        self.tranidxMatrix_gpu = gpuarray.to_gpu(self.tranidxMatrix_cpu.astype("float32"))
        self.rewardMatrix_gpu = gpuarray.to_gpu(self.rewardMatrix_cpu)
        self.e_rewardMatrix_gpu = gpuarray.to_gpu(self.e_rewardMatrix_cpu)

    def delete_tran_from_counts(self, s, a, ns):
        """
        deletes the given transition from the MDP
        """
        if s in self.tC and a in self.tC[s] and ns in self.tC[s][a]:
            del self.tC[s][a][ns]
            del self.rC[s][a][ns]

        if ns in self.trD and a in self.trD[s] and s in self.trD[ns][a]:
            del self.trD[ns][a][s]

    def delete_state_from_MDP(self, s):
        del self.tC[s]
        del self.rC[s]

        del self.tD[s]
        del self.rD[s]

        # give the index back to the pool
        i = self.s2i[s]
        self.free_i.append(i)
        del self.s2i[s]

        # find the reverse states and update teh transition Matrices
        fan_in_states = []
        for a in self.A:
            for fan_in_s in list(self.trD[s][a].keys()):
                del self.tC[fan_in_s][a][s]
                self.update_mdp_for(fan_in_s, a)

            fan_in_states += list(self.trD[s][a].keys())

    def get_rmax_reward_logic(self, s, a):
        # get the sum of distances for k nearest neighbor
        # pick the top 10%
        # set all unknwon sa for this s as rmax  actions as rmax.
        return 0  # todo add some logic here

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

    def do_backup(self, mode, module, n_backups):
        bkp_fxn_dict = {"optimal": {"CPU": self.opt_bellman_backup_step_cpu, "GPU": self.opt_bellman_backup_step_gpu},
                        "safe": {"CPU": self.safe_bellman_backup_step_cpu, "GPU": self.safe_bellman_backup_step_gpu},
                        "exploration": {"CPU": self.explr_bellman_backup_step_cpu,
                                        "GPU": self.explr_bellman_backup_step_gpu}}
        sync_fxn_dict = {"optimal": {"GPU": self.sync_opt_val_vectors_from_GPU},
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
    def sample_action_from_qval_dict(self, qval_dict):
        return random.choices(list(qval_dict.keys()), list(qval_dict.values()), k=1)[0]

    def get_action_from_q_matrix(self, hs, qMatrix, soft=False, weight_nn=False, plcy_k=1, kNN_on_sa=False):
        qval_dict = {}
        if kNN_on_sa:
            for a in self.A:
                knn_hsa = self._get_knn_hs_kd_with_action_tree((hs, a), k=plcy_k)
                knn_hs_norm = self.get_kernel_probs(knn_hsa, delta=self.knn_delta) \
                    if weight_nn else {k: 1 / len(knn_hsa) for k in knn_hsa}
                qval_dict[a] = np.sum([qMatrix[self.s2i[sa[0]], self.a2i[sa[1]]] * p for sa, p in knn_hs_norm.items()])
        else:
            knn_hs = self._get_knn_hs_kdtree(hs, k=plcy_k)
            knn_hs_norm = self.get_kernel_probs(knn_hs, delta=self.knn_delta) \
                if weight_nn else {k: 1 / len(knn_hs) for k in knn_hs}
            for a in self.A:
                qval_dict[a] = np.sum([qMatrix[self.s2i[s], self.a2i[a]] * p for s, p in knn_hs_norm.items()])

        if soft:
            return self.sample_action_from_qval_dict(qval_dict)
        else:
            return max(qval_dict, key=qval_dict.get)

    def get_opt_action(self, hs, soft=False, weight_nn=False, plcy_k=1, kNN_on_sa=False):
        return self.get_action_from_q_matrix(hs, self.qD_cpu, soft=soft, weight_nn=weight_nn,
                                             plcy_k=plcy_k, kNN_on_sa=kNN_on_sa)

    def get_safe_action(self, hs, soft=False, weight_nn=False, plcy_k=1, kNN_on_sa=False):
        return self.get_action_from_q_matrix(hs, self.s_qD_cpu, soft=soft, weight_nn=weight_nn,
                                             plcy_k=plcy_k, kNN_on_sa=kNN_on_sa)

    def get_explr_action(self, hs, soft=False, weight_nn=False, plcy_k=1, kNN_on_sa=False):
        return self.get_action_from_q_matrix(hs, self.e_qD_cpu, soft=soft, weight_nn=weight_nn,
                                             plcy_k=plcy_k, kNN_on_sa=kNN_on_sa)

    def get_kernel_probs(self, knn_dist_dict, delta=None):
        # todo Add a choice to do exponential averaging here.
        delta = delta or self.knn_delta
        all_knn_kernels = {nn: 1 / (dist + delta) for nn, dist in knn_dist_dict.items()}
        all_knn_probs = {nn: knn_kernel / sum(all_knn_kernels.values()) for nn, knn_kernel in all_knn_kernels.items()}
        return all_knn_probs

    def get_state_count(self):
        return len(self.s2i)

    def _update_nn_kd_tree(self):
        assert len(self.s2i) > 2
        self.state_list = [s for s in self.s2i if s not in self.omit_list]
        self.KDTree = KDTree(np.array(self.state_list), leaf_size=40)

    def _update_nn_kd_with_action_tree(self):
        self.s_list_D = {a: [] for a in self.A}
        for s in self.known_sa_dict:
            for a in self.known_sa_dict[s]:
                self.s_list_D[a].append(s)

        self.KDActionTrees = {a: KDTree(np.array(st_list), leaf_size=40) for a, st_list in self.s_list_D.items()}

    def _get_nn_hs_kd_with_action_tree(self, sa):
        s, a = sa
        if s in self.omit_list:
            return hsa

        nn_dist, nn_idx = self.KDActionTrees[a].query(np.array([s]), k=1)
        nearest_neighbor_hs = self.s_list_D[a][int(nn_idx.squeeze())]
        nearest_neighbor_hsa = self.hash_state_action(nearest_neighbor_hs, a)
        return nearest_neighbor_hsa

    def _get_knn_hs_kd_with_action_tree(self, sa, k):
        s, a = sa
        if s in self.omit_list or not self.s_list_D:
            return {sa: 0}
        nn_dist, nn_idx = self.KDActionTrees[a].query(np.array([s]), k=k)
        nn_dist, nn_idx = nn_dist.reshape(-1), nn_idx.reshape(-1)
        nn_dict = {self.hash_state_action(self.s_list_D[a][int(idx)], a): nn_dist[i]
                   for i, idx in enumerate(nn_idx)}
        return nn_dict

    def _get_nn_hs_kdtree(self, hs, return_dist=False):
        nn_hs, nn_dist = list(self._get_knn_hs_kdtree(hs, k=1).items())[0]
        if return_dist:
            return nn_hs, nn_dist
        else:
            return nn_hs

    def _get_knn_hs_kdtree(self, s, k):
        if s in self.omit_list or not self.state_list:
            return {s: 0}

        nn_dist, nn_idx = self.KDTree.query(np.array([s]), k=k)
        nn_dist, nn_idx = nn_dist.reshape(-1), nn_idx.reshape(-1)
        nn_dict = {self.state_list[int(idx)]: nn_dist[i] for i, idx in enumerate(nn_idx)}
        return nn_dict

    def _get_radial_nn_hs_kdtree(self, hs, radius):
        nn_idxs, dist = self.KDTree.query_radius(np.array([hs]), r=radius,
                                                 return_distance=True, sort_results=True)
        nn_idxs, dist = nn_idxs[0], dist[0]
        nn_dict = {self.state_list[indx]: dist[i] for i, indx in enumerate(nn_idxs)}

        return nn_dict

    def solve(self, eps=1e-5, mode=None, safe_bkp=False, explr_bkp=False, verbose=True):
        mode = mode or self.default_mode

        st = time.time()
        curr_error = self.curr_vi_error
        while abs(self.curr_vi_error) > eps:
            self.do_optimal_backup(mode=mode, n_backups=250)
            if safe_bkp:
                self.do_safe_backup(mode=mode, n_backups=250)
            if explr_bkp:
                self.do_explr_backup(mode=mode, n_backups=250)

            if self.curr_vi_error < curr_error / 10 and verbose:
                print("Elapsed Time:{}s, VI Error:{}, #Backups: {}".format(int(time.time() - st),
                                                                           round(self.curr_vi_error, 8),
                                                                           self.gpu_backup_counter))
                curr_error = self.curr_vi_error
        et = time.time()
        if verbose: print("Time takedn to solve", et - st)

    @property
    def qval_distribution(self):
        qvalDict = cpy(self.qvalDict)
        return [qsa for qs in qvalDict for qsa in qs]

    @property
    def unknown_state_count(self):
        return sum([self.check_if_unknown(s) for s in self.tD])

    @property
    def unknown_state_action_count(self):
        return sum([self.check_if_unknown(s, a) for s in self.tD for a in self.tD[s]])

    @property
    def fully_unknown_state_count(self):
        return sum([all([self.check_if_unknown(s, a) for a in self.tD[s]]) for s in self.tD])

    @property
    def fully_unknown_states(self):
        return [s for s in self.tD if all([self.check_if_unknown(s, a) for a in self.tD[s]])]

    def check_if_unknown(self, s, a=None):
        if s == "unknown_state":
            return False
        if a is not None:
            return "unknown_state" in self.tD[s][a]
        else:
            return sum([1 for a in self.A if "unknown_state" in self.tD[s][a]]) == len(self.A)

    def get_seen_action_count(self, s):
        return sum([1 for a in self.tC[s] if "unknown_state" not in self.tC[s][a]])

    def get_explored_action_count(self, s):
        return sum([1 for a in self.tC[s] if sum(self.tC[s][a].values()) > self.vi_params["rmax_thres"]])

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
        return [sum([1 for a in self.tC[s] if sum(self.tC[s][a].values()) > self.vi_params["rmax_thres"]]) for s in
                self.tC]

    @property
    def seen_action_count_distribution(self):
        return [sum([1 for a in self.tC[s] if "unknown_state" not in self.tC[s][a]]) for s in self.tC]

    @property
    def state_count_distribution(self):
        return [sum([sum(self.tC[s][a].values()) for a in self.tC[s]]) for s in self.tC]

    @property
    def self_loop_prob_distribution(self):
        return [self.tD[s][a][ns] for s in self.tD for a in self.tD[s] for ns in self.tD[s][a] if s == ns]

    @property
    def self_loop_count(self):
        return sum([1 for s in self.tD for a in self.tD[s] for ns in self.tD[s][a] if s == ns])

    @property
    def end_state_state_action_count(self):
        return sum([1 for s in self.tD for a in self.tD[s] if "end_state" in self.tD[s][a]])

    def opt_bellman_backup_step_cpu(self):
        backup_error = 0
        for s, s_i in self.s2i.items():
            for a, a_i in self.a2i.items():
                ns_values = np.array([self.vD_cpu[ns_i] for ns_i in self.tranidxMatrix_cpu[a_i, s_i]]).squeeze()
                expected_ns_val = np.sum(self.tranProbMatrix_cpu[a_i, s_i] * ns_values)
                expected_reward = np.sum(self.tranProbMatrix_cpu[a_i, s_i] * self.rewardMatrix_cpu[a_i, s_i])
                self.qD_cpu[s_i, a_i] = expected_reward + self.vi_params["gamma"] * expected_ns_val

            new_val = np.max(self.qD_cpu[s_i])

            backup_error = max(backup_error, abs(new_val - self.vD_cpu[s_i]))
            self.vD_cpu[s_i] = new_val
            self.pD_cpu[s_i] = np.argmax(self.qD_cpu[s_i])

        self.curr_vi_error = backup_error
        return backup_error

    def safe_bellman_backup_step_cpu(self):
        backup_error = 0
        for s, s_i in self.s2i.items():
            for a, a_i in self.a2i.items():
                ns_values = np.array([self.s_vD_cpu[ns_i] for ns_i in self.tranidxMatrix_cpu[a_i, s_i]]).squeeze()
                expected_ns_val = np.sum(self.tranProbMatrix_cpu[a_i, s_i] * ns_values)
                expected_reward = np.sum(self.tranProbMatrix_cpu[a_i, s_i] * self.rewardMatrix_cpu[a_i, s_i])
                self.s_qD_cpu[s_i, a_i] = expected_reward + self.vi_params["gamma"] * expected_ns_val

            max_q, sum_q = np.max(self.s_qD_cpu[s_i]), np.sum(self.s_qD_cpu[s_i])
            new_val = (1 - self.vi_params["slip_prob"]) * max_q + self.vi_params["slip_prob"] * (sum_q - max_q)

            backup_error = max(backup_error, abs(new_val - self.s_vD_cpu[s_i]))
            self.s_vD_cpu[s_i] = new_val
            self.s_pD_cpu[s_i] = np.argmax(self.s_qD_cpu[s_i])

        self.s_curr_vi_error = backup_error

    def explr_bellman_backup_step_cpu(self):
        backup_error = 0
        for s, s_i in self.s2i.items():
            for a, a_i in self.a2i.items():
                ns_values = np.array([self.e_vD_cpu[ns_i] for ns_i in self.tranidxMatrix_cpu[a_i, s_i]]).squeeze()
                expected_ns_val = np.sum(self.tranProbMatrix_cpu[a_i, s_i] * ns_values)
                expected_reward = np.sum(self.tranProbMatrix_cpu[a_i, s_i] * self.e_rewardMatrix_cpu[a_i, s_i])
                self.e_qD_cpu[s][a] = expected_reward + self.vi_params["gamma"] * expected_ns_val

            new_val = np.max(self.qD_cpu[s_i])

            backup_error = max(backup_error, abs(new_val - self.e_vD_cpu[s]))
            self.e_vD_cpu[s_i] = new_val
            self.e_pD_cpu[s_i] = max(self.e_qD_cpu[s], key=self.e_qD_cpu[s].get)

        self.e_curr_vi_error = backup_error



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
