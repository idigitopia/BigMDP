import torch
from collections import defaultdict
from collections import deque
import math

def init2dict():
    return {}

def init2zero():
    return 0

def init2zero_def_dict():
    return defaultdict(init2zero)

def init2zero_def_def_dict():
    return defaultdict(init2zero_def_dict)


class FullMDP(object):
    def __init__(self, A, ur=-1000, vi_params= {"gamma":0.99, "slip_prob":0, "rmax_reward":1000, "rmax_thres":10, "balanced_explr":False}):
        """
        :param A: Action Space of the MDP
        :param ur: reward for undefined state action pair
        """
        # MDP Parameters
        self.tC = defaultdict(init2zero_def_def_dict) # Transition Counts
        self.rC = defaultdict(init2zero_def_def_dict) # Reward Counts
        self.tD = defaultdict(init2zero_def_def_dict) # Transition Probabilities
        self.rD = init2zero_def_def_dict()  # init2def_def_dict() # Reward Expected
        self.ur = ur
        self.A = A

        # VI parameters
        self.vi_params = vi_params

        # Optimal Policy Parameters
        self.vD = init2zero_def_dict() # Optimal Value Vector
        self.qD = init2zero_def_def_dict()  # Optimal Q Value Matrix
        self.pD = {} # Optimal Policy Vector
        self.curr_vi_error = float("inf") # Backup Error

        # Exploration Policy Parameters
        self.e_vD = init2zero_def_dict()  # Exploration Value Vector
        self.e_qD = init2zero_def_def_dict()  # Exploration Q Value Matrix\
        self.e_pD = {}  # Exploration Policy Vector
        self.e_rD = init2zero_def_def_dict()  # exploration reward to check which state is visited.
        self.e_curr_vi_error = float("inf") # Backup error

        # Safe Policy Parameters
        self.s_vD = init2zero_def_dict()  # Exploration Value Vector
        self.s_qD = init2zero_def_def_dict()  # Exploration Q Value Matrix\
        self.s_pD = {}  # Exploration Policy Vector
        self.s_curr_vi_error = float("inf") # Backup error

        self._seed_transitions()

    def _seed_transitions(self):
        """
        Initializes the MDP With "end_state" and a "unknown_state"
        "end_state": Abstract state for all terminal State
        "unknown_state": Abstract target state for all undefined State action pairs
        """
        for a in self.A:
            self.consume_transition(["end_state", a, "end_state", 0, 1]) # [s, a, ns, r, d]
            self.consume_transition(["unknown_state", a, "unknown_state", 0, 0]) # [s, a, ns, r, d]

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
                    self.update_mdp_for(s,a_)

    def update_mdp_for(self, s, a):
        """
        updates transition probabilities as well as reward as per the transition counts for the passed state action pair
        """
        self.tD[s][a] = init2zero_def_dict()
        for ns_ in self.tC[s][a]:
            self.tD[s][a][ns_] = self.tC[s][a][ns_] / sum(self.tC[s][a].values())
            self.rD[s][a] = sum(self.rC[s][a].values()) / sum(self.tC[s][a].values())
            # self.rd[s][a][ns_] = self.rc[s][a][ns_] / sum(self.tc[s][a][ns_])  # for state action nextstate

    def delete_tran_from_counts(self,s,a,ns):
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
        s, a, ns, r, d =  tran
        ns = "end_state" if d else ns

        # if the next state is not in the MDP add a dummy transitions to unknown state for all state actions
        self.seed_for_new_state(ns)
        self.seed_for_new_state(s)

        # delete seeded transition once an actual state action set is encountered
        self.delete_tran_from_counts(s,a,"unknown_state")

        # account for the seen transition
        self.tC[s][a][ns] += 1
        self.rC[s][a][ns] += r

        # If the branching factor is too large remove the least occuring transition #Todo Set this as parameter
        if len(self.tC[s][a]) > 20:
            self.delete_tran_from_counts(s,a,min(self.tC[s][a], key = self.tC[s][a].get))

        # update the transition probability for all other next states.
        self.update_mdp_for(s,a)


    def do_single_optimal_backup(self):
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


    def do_single_exploration_backup(self):
        backup_error = 0
        for s in self.tD:
            for a in self.A:
                if s in ["end_state", "unknown_state"]:
                    new_val = 0 
                    self.e_qD[s][a] = 0
                else:
                    sa_count = sum(self.tC[s][a].values())
                    if self.vi_params["balanced_explr"]:
                        self.e_rD[s][a] = self.vi_params["rmax_reward"] * (self.vi_params["rmax_thres"] - sa_count) if sa_count<10 else 100*(math.e**(-0.01*sa_count))
                    else:
                        self.e_rD[s][a] = self.vi_params["rmax_reward"] * (self.vi_params["rmax_thres"] - sa_count) if sa_count<10 else self.rD[s][a]


                    self.e_qD[s][a] = self.e_rD[s][a] + self.vi_params["gamma"]*sum(self.tD[s][a][ns] * self.e_vD[ns] for ns in self.tD[s][a])

            new_val = max(self.e_qD[s].values())

            backup_error = max(backup_error, abs(new_val - self.e_vD[s]))
            self.e_vD[s] = new_val
            self.e_pD[s] = max(self.e_qD[s], key=self.e_qD[s].get)
        self.e_curr_vi_error = backup_error


    def do_single_safe_backup(self):
        backup_error = 0
        for s in self.tD:
            for a in self.A:
                expected_ns_val = sum(self.tD[s][a][ns] * self.vD[ns] for ns in self.tD[s][a])
                self.s_qD[s][a] = self.rD[s][a] + self.vi_params["gamma"] * expected_ns_val

            new_val = (1 - self.vi_params["slip_prob"]) * max(self.s_qD[s].values()) + \
                           self.vi_params["slip_prob"] * sum([qsa for qsa in self.s_qD[s].values()])

            backup_error = max(backup_error, abs(new_val - self.s_vD[s]))
            self.s_vD[s] = new_val
            self.s_pD[s] = max(self.s_qD[s], key=self.s_qD[s].get)
        self.s_curr_vi_error = backup_error


