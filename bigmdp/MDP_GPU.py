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
from async_vi.utils.kernel_templates import complex_vi_kernel_code_template as vi_kernel_template
from async_vi.utils.kernel_templates import NN_kernel_code_template
import pycuda
# -- initialize the device
from statistics import mean
from async_vi.utils.tmp_vi_helper import *


def init2dict():
    return {}


def init2zero():
    return 0


def init2zero_def_dict():
    return defaultdict(init2zero)


def init2zero_def_def_dict():
    return defaultdict(init2zero_def_dict)


class FullMDP(object):
    def __init__(self, A, ur=-1000, MAX_S_COUNT=1000000, MAX_NS_COUNT =20 ,
                 vi_params={"gamma": 0.99,"slip_prob": 0,"rmax_reward": 1000,"rmax_thres": 10,"balanced_explr": False},
                 policy_params= {"unhash_array_len":2}):
        """
        :param A: Action Space of the MDP
        :param ur: reward for undefined state action pair
        """
        # VI CPU/GPU parameters
        self.vi_params = vi_params
        self.MAX_S_COUNT = MAX_S_COUNT  # Maximum number of state that can be allocated in GPU
        self.MAX_NS_COUNT = MAX_NS_COUNT  # MAX number of next states for a single state action pair
        self.curr_vi_error = float("inf")  # Backup error
        self.e_curr_vi_error = float("inf")  # Backup error
        self.s_curr_vi_error = float("inf")  # Backup error

        # MDP Parameters
        self.tC = defaultdict(init2zero_def_def_dict)  # Transition Counts
        self.rC = defaultdict(init2zero_def_def_dict)  # Reward Counts
        self.tD = defaultdict(init2zero_def_def_dict)  # Transition Probabilities
        self.rD = init2zero_def_def_dict()  # init2def_def_dict() # Reward Expected
        self.ur = ur
        self.A = A

        # MDP GPU Parameters
        self.s2idx = {s: i for i, s in enumerate(self.tD)}
        self.idx2s = {i: s for i, s in enumerate(self.tD)}
        self.tranCountMatrix_gpu = gpuarray.to_gpu(np.zeros((len(self.A), self.MAX_S_COUNT, self.MAX_NS_COUNT)).astype('float32'))
        self.tranProbMatrix_gpu = gpuarray.to_gpu(np.zeros((len(self.A), self.MAX_S_COUNT, self.MAX_NS_COUNT)).astype('float32'))
        self.tranidxMatrix_gpu = gpuarray.to_gpu(np.zeros((len(self.A), self.MAX_S_COUNT, self.MAX_NS_COUNT)).astype('float32'))
        self.rewardMatrix_gpu = gpuarray.to_gpu(np.zeros((len(self.A), self.MAX_S_COUNT, self.MAX_NS_COUNT)).astype('float32'))

        # Optimal Policy GPU parameters
        self.vD_gpu = gpuarray.to_gpu(np.zeros((self.MAX_S_COUNT, 1)).astype('float32'))
        self.qD_gpu = gpuarray.to_gpu(np.zeros((self.MAX_S_COUNT, len(self.A))).astype('float32'))
        self.gpu_backup_counter = 0

        # Exploration Policy GPU parameters
        self.e_vD_gpu = gpuarray.to_gpu(np.zeros((self.MAX_S_COUNT, 1)).astype('float32'))
        self.e_qD_gpu = gpuarray.to_gpu(np.zeros((self.MAX_S_COUNT, len(self.A))).astype('float32'))
        self.e_rewardMatrix_gpu = gpuarray.to_gpu(np.zeros((len(self.A), self.MAX_S_COUNT, self.MAX_NS_COUNT)).astype('float32'))
        self.e_gpu_backup_counter = 0

        # Safe Policy GPU parameters
        self.s_vD_gpu = gpuarray.to_gpu(np.zeros((self.MAX_S_COUNT, 1)).astype('float32'))
        self.s_qD_gpu = gpuarray.to_gpu(np.zeros((self.MAX_S_COUNT, len(self.A))).astype('float32'))
        self.s_gpu_backup_counter = 0

        # Optimal Policy Parameters
        self.vD = init2zero_def_dict()  # Optimal Value Vector
        self.qD = init2zero_def_def_dict()  # Optimal Q Value Matrix
        self.pD = {}  # Optimal Policy Vector

        # Exploration Policy Parameters
        self.e_vD = init2zero_def_dict()  # Exploration Value Vector
        self.e_qD = init2zero_def_def_dict()  # Exploration Q Value Matrix\
        self.e_pD = {}  # Exploration Policy Vector
        self.e_rD = init2zero_def_def_dict()  # exploration reward to check which state is visited.

        # Safe Policy Parameters
        self.s_vD = init2zero_def_dict()  # Exploration Value Vector
        self.s_qD = init2zero_def_def_dict()  # Exploration Q Value Matrix\
        self.s_pD = {}  # Exploration Policy Vector


        # Policy Search , nn parameters
        self._unhash_dict = {}
        self._total_calls = 0
        self._nn_calls = 0
        self._unhash_idx2s = {}
        self.nn_searchMatrix_gpu = gpuarray.to_gpu(np.ones((self.MAX_S_COUNT,policy_params["unhash_array_len"])).astype("float32")*9999)

        self._initialize_end_and_unknown_state()


    def _initialize_end_and_unknown_state(self):
        """
        Initializes the MDP With "end_state" and a "unknown_state"
        "end_state": Abstract state for all terminal State
        "unknown_state": Abstract target state for all undefined State action pairs
        """
        for a in self.A:
            self.consume_transition(["unknown_state", a, "unknown_state", 0, 0])  # [s, a, ns, r, d]
            self.consume_transition(["end_state", a, "end_state", 0, 1])  # [s, a, ns, r, d]

    def update_state_indexes(self):
        """
        updates all indexes for GPU-CPU communication
        """
        self.s2idx = {s: i for i, s in enumerate(self.tD)}
        self.idx2s = {i: s for i, s in enumerate(self.tD)}
        self.a2idx = {s: i for i, s in enumerate(self.A)}
        self.idx2a = {i: s for i, s in enumerate(self.A)}

    def update_mdp_at_gpu_for(self, s, a):
        """
        updates GPU matrix for particular state action
        """
        for i, a in [(self.a2idx[a], a)]:
            for j, s in [(self.s2idx[s], s)]:
                self.tranProbMatrix_gpu[i][j] = gpuarray.to_gpu(np.array(  [self.tD[s][a][ns] for ns in self.tD[s][a]] + [0] * (self.MAX_NS_COUNT - len(self.tD[s][a]))).astype("float32"))
                self.tranidxMatrix_gpu[i][j] = gpuarray.to_gpu(np.array([self.s2idx[ns] for ns in self.tD[s][a]] + [self.s2idx["unknown_state"]] * ( self.MAX_NS_COUNT - len(self.tD[s][a]))).astype("float32"))
                self.rewardMatrix_gpu[i][j] = gpuarray.to_gpu(  np.array([self.rD[s][a]] * self.MAX_NS_COUNT).astype("float32"))

                self.e_rewardMatrix_gpu[i][j] = gpuarray.to_gpu(np.array([self.get_rmax_reward_logic(s,a)] * self.MAX_NS_COUNT).astype("float32"))
                assert len(self.tranProbMatrix_gpu[i][j]) == len(self.tranidxMatrix_gpu[i][j])

    def update_mdp_for(self, s, a):
        """
        updates transition probabilities as well as reward as per the transition counts for the passed state action pair
        """

        self.tD[s][a] = init2zero_def_dict()
        self.update_state_indexes()

        for ns_ in self.tC[s][a]:
            self.tD[s][a][ns_] = self.tC[s][a][ns_] / sum(self.tC[s][a].values())
            self.rD[s][a] = sum(self.rC[s][a].values()) / sum(self.tC[s][a].values())
            # self.rd[s][a][ns_] = self.rc[s][a][ns_] / sum(self.tc[s][a][ns_])  # for state action nextstate

            self.update_mdp_at_gpu_for(s, a)

    def sync_opt_val_vectors_from_GPU(self):
        for i, v in enumerate(self.vD_gpu):
            if i not in self.idx2s:
                break
            self.vD[self.idx2s[i]] = float(v.get())

        for i, qs in enumerate(self.qD_gpu):
            if i not in self.idx2s:
                break
            for j, qsa in enumerate(self.qD_gpu[i]):
                self.qD[self.idx2s[i]][self.idx2a[j]] = float(qsa.get())

        for s in self.qD:
            self.pD[s] = max(self.qD[s], key=self.qD[s].get)

    def sync_explr_val_vectors_from_GPU(self):
        for i, v in enumerate(self.e_vD_gpu):
            if i not in self.idx2s:
                break
            self.e_vD[self.idx2s[i]] = float(v.get())

        for i, qs in enumerate(self.e_qD_gpu):
            if i not in self.idx2s:
                break
            for j, qsa in enumerate(self.e_qD_gpu[i]):
                self.e_qD[self.idx2s[i]][self.idx2a[j]] = float(qsa.get())

        for s in self.e_qD:
            self.e_pD[s] = max(self.e_qD[s], key=self.e_qD[s].get)

    def sync_safe_val_vectors_from_GPU(self):
        for i, v in enumerate(self.s_vD_gpu):
            if i not in self.idx2s:
                break
            self.s_vD[self.idx2s[i]] = float(v.get())

        for i, qs in enumerate(self.s_qD_gpu):
            if i not in self.idx2s:
                break
            for j, qsa in enumerate(self.s_qD_gpu[i]):
                self.s_qD[self.idx2s[i]][self.idx2a[j]] = float(qsa.get())

        for s in self.s_qD:
            self.s_pD[s] = max(self.s_qD[s], key=self.s_qD[s].get)

    def seed_for_new_state(self, s):
        """
        Checks if the state is not in the MDP, if so seeds with undefined state actions
        :param s: new state
        """

        if s not in self.tC:
            for a_ in self.A:
                if a_ not in self.tC[s]:
                    self.tC[s][a_]["unknown_state"] = 1
                    self.rC[s][a_]["unknown_state"] = self.ur
                    self.update_mdp_for(s, a_)

            # update unhashdict for policy
            if s not in ["end_state", "unknown_state"]:
                self._unhash_dict[s] = unhAsh(s)
                self._unhash_idx2s[len(self._unhash_dict)-1]=s
                self.nn_searchMatrix_gpu[len(self._unhash_dict)-1] = gpuarray.to_gpu(np.array(unhAsh(s)).astype("float32"))

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
        s, a, ns, r, d = tran
        ns = "end_state" if d else ns

        # if the next state is not in the MDP add a dummy transitions to unknown state for all state actions
        self.seed_for_new_state(ns)
        self.seed_for_new_state(s)

        # delete seeded transition once an actual state action set is encountered
        self.delete_tran_from_counts(s, a, "unknown_state")

        # account for the seen transition
        self.tC[s][a][ns] += 1
        self.rC[s][a][ns] += r

        # If the branching factor is too large remove the least occuring transition #Todo Set this as parameter
        if len(self.tC[s][a]) > self.MAX_NS_COUNT:
            self.delete_tran_from_counts(s, a, min(self.tC[s][a], key=self.tC[s][a].get))

        # update the transition probability for all other next states.
        self.update_mdp_for(s, a)

    def do_optimal_backup(self, mode="CPU", n_backups=1):
        if mode == "CPU":
            for _ in range(n_backups):
                self.opt_bellman_backup_step_cpu()
        elif mode == "GPU":
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

    def explr_bellman_backup_step_cpu(self):
        backup_error = 0
        for s in self.tD:
            for a in self.A:
                self.e_rD[s][a] = self.get_rmax_reward_logic(s,a)
                self.e_qD[s][a] = self.e_rD[s][a] + self.vi_params["gamma"] * sum(self.tD[s][a][ns] * self.e_vD[ns] for ns in self.tD[s][a])

            new_val = max(self.e_qD[s].values())

            backup_error = max(backup_error, abs(new_val - self.e_vD[s]))
            self.e_vD[s] = new_val
            self.e_pD[s] = max(self.e_qD[s], key=self.e_qD[s].get)
        self.e_curr_vi_error = backup_error

    def safe_bellman_backup_step_cpu(self):
        backup_error = 0
        for s in self.tD:
            for a in self.A:
                expected_ns_val = sum(self.tD[s][a][ns] * self.s_vD[ns] for ns in self.tD[s][a])
                self.s_qD[s][a] = self.rD[s][a] + self.vi_params["gamma"] * expected_ns_val

            new_val = (1 - self.vi_params["slip_prob"]) * max(self.s_qD[s].values()) + \
                      self.vi_params["slip_prob"] * mean([qsa for qsa in self.s_qD[s].values()])

            backup_error = max(backup_error, abs(new_val - self.s_vD[s]))
            self.s_vD[s] = new_val
            self.s_pD[s] = max(self.s_qD[s], key=self.s_qD[s].get)
        self.s_curr_vi_error = backup_error

    # def update_mdp_at_gpu_for_all(self):
    #     for i, a in enumerate(self.A):
    #         for j, s in enumerate(self.tD):
    #             self.tranProbMatrix_gpu[i][j] = gpuarray.to_gpu(np.array( [self.tD[s][a][ns] for ns in self.tD[s][a]] + [0] * (self.max_ns_count - len(self.tD[s][a]))).astype("float32"))
    #             self.tranidxMatrix_gpu[i][j] = gpuarray.to_gpu(np.array([self.s2idx[ns] for ns in self.tD[s][a]] + [self.s2idx["unknown_state"]] * (self.max_ns_count - len(self.tD[s][a]))).astype("float32"))
    #             self.rewardMatrix_gpu[i][j] = gpuarray.to_gpu(np.array([self.rD[s][a]] * self.max_ns_count).astype("float32"))
    #             assert len(self.tranProbMatrix_gpu[i][j]) == len(self.tranidxMatrix_gpu[i][j])
    #
    # def update_mdp_at_gpu_for(self,s,a):
    #     for i, a in [(self.a2idx[a],a)]:
    #         for j, s in [(self.s2idx[s],s)]:
    #             self.tranProbMatrix_gpu[i][j] = gpuarray.to_gpu(np.array( [self.tD[s][a][ns] for ns in self.tD[s][a]] + [0] * (self.max_ns_count - len(self.tD[s][a]))).astype("float32"))
    #             self.tranidxMatrix_gpu[i][j] = gpuarray.to_gpu(np.array([self.s2idx[ns] for ns in self.tD[s][a]] + [self.s2idx["unknown_state"]] * (self.max_ns_count - len(self.tD[s][a]))).astype("float32"))
    #             self.rewardMatrix_gpu[i][j] = gpuarray.to_gpu(np.array([self.rD[s][a]] * self.max_ns_count).astype("float32"))
    #             assert len(self.tranProbMatrix_gpu[i][j]) == len(self.tranidxMatrix_gpu[i][j])

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
            'SLIP_ACTION_PROB': 0
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
        tgt_vD_gpu = gpuarray.empty((ROW_COUNT, 1), np.float32)
        tgt_qD_gpu = gpuarray.empty((ROW_COUNT, ACTION_COUNT), np.float32)
        tgt_error_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, 1)))

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
            print("checkingggg for epsilng stop")
            max_error_gpu = gpuarray.max(tgt_error_gpu,stream=None)  # ((value_vector_gpu,new_value_vector_gpu)
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
            'SLIP_ACTION_PROB': self.vi_params["slip_prob"]
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
        tgt_vD_gpu = gpuarray.empty((ROW_COUNT, 1), np.float32)
        tgt_qD_gpu = gpuarray.empty((ROW_COUNT, ACTION_COUNT), np.float32)
        tgt_error_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, 1)))

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
            max_error_gpu = gpuarray.max(tgt_error_gpu,
                                         stream=None)  # ((value_vector_gpu,new_value_vector_gpu)
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
            'SLIP_ACTION_PROB': 0
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
        tgt_vD_gpu = gpuarray.empty((ROW_COUNT, 1), np.float32)
        tgt_qD_gpu = gpuarray.empty((ROW_COUNT, ACTION_COUNT), np.float32)
        tgt_error_gpu = gpuarray.to_gpu(np.zeros((ROW_COUNT, 1)))

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
        if (self.e_gpu_backup_counter+1) % 25 == 0:
            max_error_gpu = gpuarray.max(tgt_error_gpu, stream=None)  # ((value_vector_gpu,new_value_vector_gpu)
            max_error = float(max_error_gpu.get())
            max_error_gpu.gpudata.free()
            self.e_curr_vi_error = max_error
        tgt_error_gpu.gpudata.free()

    #### Policy Functions

    def get_opt_action(self, hs):
        nn_hs = self._get_nn_hs(hs)
        return self.pD[nn_hs]

    def get_safe_action(self, hs):
        nn_hs = self._get_nn_hs(hs)
        return self.s_pD[nn_hs]

    def get_explr_action(self, hs):
        nn_hs = self._get_nn_hs(hs)
        return self.e_pD[nn_hs]

    def _get_nn_hs(self, hs):
        self._total_calls += 1
        if hs in self.pD:
            nearest_neighbor_hs = hs
        else:
            self._nn_calls += 1
            s = unhAsh(hs)
            hm_dist_dict = {s_hat: hm_dist(s, self._unhash_dict[s_hat]) for s_hat in self.pD if
                            s_hat not in ["end_state", "unknown_state"]}
            nearest_neighbor_hs = min(hm_dist_dict.keys(),
                                      key=(lambda k: hm_dist_dict[k])) if hm_dist_dict else "end_state"
        return nearest_neighbor_hs

    def get_state_count(self):
        return len(self._unhash_dict)

    def _get_nn_hs_gpu(self, hs):
        self._total_calls += 1
        if hs in self.pD:
            nearest_neighbor_hs = hs
        else:
            self._nn_calls += 1
            distance_vector = self.calc_dist_vect_gpu(self.nn_searchMatrix_gpu, np.array(unhAsh(hs)))
            nearest_neighbor_hs = self._unhash_idx2s[np.argmin(distance_vector)]
        return nearest_neighbor_hs

    def calc_dist_vect_gpu(self, searchMatrix, queryVector):
        # Define kernel code template
        import pycuda.autoinit
        POP_COUNT, VEC_SIZE = searchMatrix.shape
        InVector = gpuarray.to_gpu(queryVector)
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