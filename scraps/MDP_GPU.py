from pycuda import compiler, gpuarray
import numpy
import math as mth
from bigmdp.mdp.kernels import complex_vi_kernel_code_template as vi_kernel_template
from bigmdp.mdp.kernels import NN_kernel_code_template
# -- initialize the device
from copy import deepcopy as cpy
from collections import Counter
import random
from scraps.tmp_vi_helper import *

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
                 vi_params={"gamma": 0.99, "slip_prob": 0, "rmax_reward": 1000, "rmax_thres": 10,
                            "balanced_explr": False},
                 policy_params={"unhash_array_len": 2}, weight_transitions=True, default_mode="GPU"):
        """
        :param A: Action Space of the MDP
        :param ur: reward for undefined state action pair
        """
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
        self.weight_transitions = weight_transitions
        self.default_mode = default_mode

        # MDP GPU Parameters
        self.s2idx = {s: i for i, s in enumerate(self.tD)}
        self.idx2s = {i: s for i, s in enumerate(self.tD)}
        self.a2idx = {s: i for i, s in enumerate(self.A)}
        self.idx2a = {i: s for i, s in enumerate(self.A)}

        self.tranCountMatrix_cpu = np.zeros((len(self.A), self.MAX_S_COUNT, self.MAX_NS_COUNT)).astype('float32')
        self.tranProbMatrix_cpu = np.zeros((len(self.A), self.MAX_S_COUNT, self.MAX_NS_COUNT)).astype('float32')
        self.tranidxMatrix_cpu = np.zeros((len(self.A), self.MAX_S_COUNT, self.MAX_NS_COUNT)).astype('float32')
        self.rewardMatrix_cpu = np.zeros((len(self.A), self.MAX_S_COUNT, self.MAX_NS_COUNT)).astype('float32')

        self.tranCountMatrix_gpu = gpuarray.to_gpu(np.zeros((len(self.A), self.MAX_S_COUNT, self.MAX_NS_COUNT)).astype('float32'))
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
        self.s_pD = {}  # Exploration Policy Vector

        # Policy Search , nn parameters
        self._unhash_dict = {}
        self._total_calls = 0
        self._nn_calls = 0
        self._unhash_idx2s = {}
        self.nn_searchMatrix_gpu = gpuarray.to_gpu(
            np.ones((self.MAX_S_COUNT, policy_params["unhash_array_len"])).astype("float32") * 9999)

        # Track fully unknown_states
        self.fully_unknown_states  = {}

        self._initialize_end_and_unknown_state()

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
    def qvalDict(self):
        return {s: {a:self.qD_cpu[i][j] for a,j in self.a2idx.items()} for s, i in self.s2idx.items()}

    @property
    def polDict(self):
        qval = cpy(self.qvalDict)
        return {s: max(qval[s], key=qval[s].get) for s in qval}

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

    def fill_mdp_for_unknown_states(self):
        for s in tqdm(self.fully_unknown_states):
            top_20_nearest_neighbors = self._get_knn_hs_gpu(s, 20)
            for nn_hs, dist in top_20_nearest_neighbors:
                if sum([1 for a in self.A if "unknown_state" in self.tD[nn_hs][a]]) < len(self.A):
                    for a in self.A:
                        self.tC[s][a] = cpy(self.tC[nn_hs][a])
                        self.rC[s][a] = cpy(self.rC[nn_hs][a])
                        self.update_mdp_for(s,a)
            assert sum([1 for a in self.A if "unknown_state" in self.tD[s][a]]) < len(self.A)
        self.fully_unknown_states = {}

    def update_mdp_at_gpu_for_all(self,):
        """
        updates GPU matrix for particular state action
        """
        for s in tqdm(self.tD):
            for a in self.A:
                self.update_mdp_matrix_at_gpu_for(s, a)

    def sync_mdp_from_cpu_to_gpu(self,):
        self.tranCountMatrix_gpu.gpudata.free()
        self.tranProbMatrix_gpu.gpudata.free()
        self.tranidxMatrix_gpu.gpudata.free()
        self.rewardMatrix_gpu.gpudata.free()
        self.e_rewardMatrix_gpu.gpudata.free()

        self.tranCountMatrix_gpu = gpuarray.to_gpu(self.tranCountMatrix_cpu)
        self.tranProbMatrix_gpu = gpuarray.to_gpu(self.tranProbMatrix_cpu)
        self.tranidxMatrix_gpu = gpuarray.to_gpu(self.tranidxMatrix_cpu)
        self.rewardMatrix_gpu = gpuarray.to_gpu(self.rewardMatrix_cpu)
        self.e_rewardMatrix_gpu = gpuarray.to_gpu(self.e_rewardMatrix_cpu)



    def update_mdp_matrix_at_gpu_for(self, s, a):
        """
        updates GPU matrix for particular state action
        """
        for i, a in [(self.a2idx[a], a)]:
            for j, s in [(self.s2idx[s], s)]:
                self.tranProbMatrix_gpu[i][j] = gpuarray.to_gpu(np.array(
                    [self.tD[s][a][ns] for ns in self.tD[s][a]] + [0] * (self.MAX_NS_COUNT - len(self.tD[s][a]))).astype("float32"))
                self.tranidxMatrix_gpu[i][j] = gpuarray.to_gpu(np.array([self.s2idx[ns] for ns in self.tD[s][a]] + [self.s2idx["unknown_state"]] * (self.MAX_NS_COUNT - len(self.tD[s][a]))).astype("float32"))
                self.rewardMatrix_gpu[i][j] = gpuarray.to_gpu(np.array([self.rD[s][a]] * self.MAX_NS_COUNT).astype("float32"))
                self.e_rewardMatrix_gpu[i][j] = gpuarray.to_gpu(np.array([self.get_rmax_reward_logic(s, a)] * self.MAX_NS_COUNT).astype("float32"))
                assert len(self.tranProbMatrix_gpu[i][j]) == len(self.tranidxMatrix_gpu[i][j])

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
            # update unhashdict for policy
            if s not in ["end_state", "unknown_state"]:
                self._unhash_dict[s] = unhAsh(s)
                self._unhash_idx2s[len(self._unhash_dict) - 1] = s
                if self.default_mode == "GPU":
                    self.nn_searchMatrix_gpu[len(self._unhash_dict) - 1] = gpuarray.to_gpu(
                        np.array(unhAsh(s)).astype("float32"))

            self.fully_unknown_states[s]=1
            curr_idx = len(self.tC)
            self.s2idx[s] = curr_idx
            self.idx2s[curr_idx] = s

            for a_ in self.A:
                if a_ not in self.tC[s]:
                    self.tC[s][a_]["unknown_state"] = 1
                    self.rC[s][a_]["unknown_state"] = self.ur
                    self.update_mdp_for(s, a_)


    def reset_searchMatric_gpu(self):
        self._unhash_dict = {s: unhAsh(s) for s in self.tC if s not in ["end_state", "unknown_state"]}
        for i, s in enumerate(self._unhash_dict):
            self.nn_searchMatrix_gpu[i] = gpuarray.to_gpu(np.array(unhAsh(s)).astype("float32"))

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

        # if the next state is not in the MDP add a dummy transitions to unknown state for all state actions
        self.seed_for_new_state(ns)
        self.seed_for_new_state(s)

        # delete seeded transition once an actual state action set is encountered
        self.delete_tran_from_counts(s, a, "unknown_state")
        self.fully_unknown_states.pop(s, "None")

        # account for the seen transition
        self.tC[s][a][ns] += 1
        self.rC[s][a][ns] += r

        # If the branching factor is too large remove the least occuring transition #Todo Set this as parameter
        if len(self.tC[s][a]) > self.MAX_NS_COUNT:
            self.delete_tran_from_counts(s, a, min(self.tC[s][a], key=self.tC[s][a].get))

        # update the transition probability for all other next states.
        self.update_mdp_for(s, a)

    def get_rmax_reward_logic(self, s, a):
        sa_count = sum(self.tC[s][a].values())
        linearly_decreasing_rmax = self.vi_params["rmax_reward"] * (self.vi_params["rmax_thres"] - sa_count)
        exponentially_decreasing_rmax = 100 * (math.e ** (int(-0.01 * sa_count)))
        if s in ["end_state", "unknown_state"]:
            rmax_reward = 0
        else:
            if self.vi_params["balanced_explr"]:
                rmax_reward = linearly_decreasing_rmax if sa_count < self.vi_params["rmax_thres"] \
                    else exponentially_decreasing_rmax
            else:
                rmax_reward = linearly_decreasing_rmax if sa_count < 10 else self.rD[s][a]

        return rmax_reward

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


    def do_optimal_backup(self, mode="CPU", n_backups=1):
        if mode == "CPU":
            for _ in range(n_backups):
                self.opt_bellman_backup_step_cpu()
        elif mode == "GPU":
            self.sync_mdp_from_cpu_to_gpu()
            for _ in range(n_backups):
                    self.opt_bellman_backup_step_gpu()
            self.sync_opt_val_vectors_from_GPU()
        else:
            print("Illegal Mode: Not Specified")
            assert False

    def do_safe_backup(self, mode="CPU", n_backups=1):
        if mode == "CPU":
            for _ in range(n_backups):
                self.safe_bellman_backup_step_cpu()
        elif mode == "GPU":
            self.sync_mdp_from_cpu_to_gpu()
            for _ in range(n_backups):
                self.safe_bellman_backup_step_gpu()
            self.sync_safe_val_vectors_from_GPU()
        else:
            print("Illegal Mode: Not Specified")
            assert False

    def do_explr_backup(self, mode="CPU", n_backups=1):
        if mode == "CPU":
            for _ in range(n_backups):
                self.explr_bellman_backup_step_cpu()
        elif mode == "GPU":
            self.sync_mdp_from_cpu_to_gpu()
            for _ in range(n_backups):
                self.explr_bellman_backup_step_gpu()
            self.sync_explr_val_vectors_from_GPU()
        else:
            print("Illegal Mode: Not Specified")
            assert False

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
                new_val = (1 - self.vi_params["slip_prob"]) * max(self.s_qD[s].values()) + self.vi_params["slip_prob"] *  sum(next_explored_state_val_list)/len(next_explored_state_val_list)
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

    #### Policy Functions ####

    def get_opt_action(self, hs, mode="GPU"):
        nn_hs = self._get_nn_hs_gpu(hs) if mode == "GPU" else self._get_nn_hs(hs)
        return self.idx2a[np.argmax(self.qD_cpu[self.s2idx[nn_hs]])]

    def get_soft_opt_action(self, hs, mode="GPU"):
        nn_hs = self._get_nn_hs_gpu(hs) if mode == "GPU" else self._get_nn_hs(hs)
        q_values = self.qD_cpu[self.s2idx[nn_hs]]
        action = random.choices(list(range(len(q_values))), q_values, k=1)[0]
        return action

    # def get_knn_opt_action(self, hs, mode="GPU"):
    #     nn_hs = self._get_knn_hs_gpu(hs) if mode == "GPU" else self._get_knn_hs(hs)
    #     return self.idx2a[np.argmax(self.qD_cpu[self.s2idx[nn_hs]])]

    def get_safe_action(self, hs, mode="GPU"):
        nn_hs = self._get_nn_hs_gpu(hs) if mode == "GPU" else self._get_nn_hs(hs)
        return self.idx2a[np.argmax(self.s_qD_cpu[self.s2idx[nn_hs]])]

    def get_explr_action(self, hs, mode="GPU"):
        nn_hs = self._get_nn_hs_gpu(hs) if mode == "GPU" else self._get_nn_hs(hs)
        return self.idx2a[np.argmax(self.e_qD_cpu[self.s2idx[nn_hs]])]

    def _get_nn_hs(self, hs):
        self._total_calls += 1
        if hs in self.s2idx:
            nearest_neighbor_hs = hs
        else:
            self._nn_calls += 1
            s = unhAsh(hs)
            hm_dist_dict = {s_hat: hm_dist(s, self._unhash_dict[s_hat]) for s_hat in self.s2idx if
                            s_hat not in ["end_state", "unknown_state"]}
            nearest_neighbor_hs = min(hm_dist_dict.keys(),
                                      key=(lambda k: hm_dist_dict[k])) if hm_dist_dict else "end_state"
        return nearest_neighbor_hs

    def get_state_count(self):
        return len(self._unhash_dict)

    def _update_nn_kd_tree(self):
        pass

    def _get_nn_hs_kdtree(self, hs):
        pass

    def _get_knn_hs_kdtree(self, hs):
        pass

    def _get_nn_hs_gpu(self, hs):
        self._total_calls += 1
        if hs in self.s2idx:
            nearest_neighbor_hs = hs
        else:
            self._nn_calls += 1
            distance_vector = self.calc_dist_vect_gpu(self.nn_searchMatrix_gpu, np.array(unhAsh(hs)))
            min_id = np.argmin(distance_vector)
            nearest_neighbor_hs = self.idx2s[min_id + 2]
        return nearest_neighbor_hs

    def _get_knn_hs_gpu(self, hs, k):
        distance_vector = self.calc_dist_vect_gpu(self.nn_searchMatrix_gpu, np.array(unhAsh(hs)))
        # nearest_neighbor_hs = self._unhash_idx2s[np.argmin(distance_vector)]
        hm_dist_dict = {s_hat: distance_vector[self.s2idx[s_hat]-2] for s_hat in self.s2idx if s_hat not in ["end_state", "unknown_state"]}
        most_common_ns = Counter(hm_dist_dict).most_common()[:-k-1:-1]

        return most_common_ns

    def calc_dist_vect_gpu(self, searchMatrix, queryVector):
        # Define kernel code template
        POP_COUNT, VEC_SIZE = searchMatrix.shape
        InVector = gpuarray.to_gpu(np.array(queryVector, dtype= numpy.float32))
        OutVector = gpuarray.to_gpu(np.zeros((POP_COUNT,), dtype=numpy.float32))
        SearcVector = gpuarray.to_gpu(searchMatrix)

        MATRIX_SIZE = mth.ceil(mth.sqrt(POP_COUNT))
        BLOCK_SIZE = 16

        if MATRIX_SIZE % BLOCK_SIZE != 0:
            grid = (MATRIX_SIZE // BLOCK_SIZE + 1, MATRIX_SIZE // BLOCK_SIZE + 1, 1)
        else:
            grid = (MATRIX_SIZE // BLOCK_SIZE, MATRIX_SIZE // BLOCK_SIZE, 1)

        # Define actual kernel
        kernel_code = NN_kernel_code_template % {
            'MATRIX_SIZE': MATRIX_SIZE,
            'VEC_SIZE': VEC_SIZE,
            'POP_COUNT': POP_COUNT
        }

        # Compile and get function
        mod = compiler.SourceModule(kernel_code)
        get_nearest_fxn = mod.get_function("NNKernel")

        # call the function
        get_nearest_fxn(  # inputs
            InVector, OutVector, SearcVector,
            # grid
            grid=grid,
            # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
            block=(BLOCK_SIZE, BLOCK_SIZE, 1))

        OutVector_cpu = OutVector.get()
        SearcVector.gpudata.free()
        InVector.gpudata.free()
        OutVector.gpudata.free()
        # return out vector
        return OutVector_cpu


    def solve(self, eps = 1e-5, mode = None):
        mode = mode or self.default_mode

        st = time.time()
        curr_error = self.curr_vi_error
        while self.curr_vi_error > eps:
            self.do_optimal_backup(mode=mode, n_backups=250)
            self.do_explr_backup(mode = mode, n_backups=250)
            if self.curr_vi_error < curr_error/10:
                print("Elapsed Time:{}s, VI Error:{}, #Backups: {}".format(int(time.time()-st), round(self.curr_vi_error,8), self.gpu_backup_counter))
                curr_error = self.curr_vi_error
        et = time.time()
        print("Time takedn to solve",et-st)


    @property
    def unknown_state_count(self):
        c = 0
        for s in mdp_T.tD:
            if sum([1 for a in mdp_T.A if "unknown_state" in mdp_T.tD[s][a]]) == len(mdp_T.A):
                c += 1
        return c

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
        count = defaultdict(init2zero)
        for s in self.tD:
            for a in self.tD[s]:
                for ns in self.tD[s][a]:
                    count[ns] += 1
        state_action_in_distr = list(count.values())
        return state_action_in_distr

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

