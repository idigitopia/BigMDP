import ray
import time
import math 

ray.shutdown()
ray.init()

from collections import deque
from statistics import mean
from async_vi.utils.tmp_vi_helper import *

# Totally in discrete state land ##
@ray.remote
class SharedStorage(object):
    def __init__(self):
        # Optimal Policy Parameters
        self.curr_value_Vector = {}
        self.curr_Q_Matrix = {}
        self.curr_Policy_Vector = {}
        self.curr_vi_error = float("inf")

        # Exploration Policy Parameters
        self.curr_e_value_Vector = {}
        self.curr_e_Q_Matrix = {}
        self.curr_e_Policy_Vector = {}
        self.curr_e_vi_error = float("inf")

        # Safe Policy Parameters
        self.curr_s_value_Vector = {}
        self.curr_s_Q_Matrix = {}
        self.curr_s_Policy_Vector = {}
        self.curr_s_vi_error = float("inf")

        self.solve_flag = False
        self.unhash_dict_ = {}

        self.transition_queue = deque()
        self._total_calls = 0 
        self._nn_calls = 0

    def start_value_iteration(self):
        self.solve_flag = True
        
    def stop_value_iteration(self):
        self.solve_flag = False

    def keep_solving(self):
        return self.solve_flag

    def add_to_transition_queue(self, tran):
        self.transition_queue.append(tran)
      
    def pop_from_transition_queue(self):
        if self.transition_queue:
            return self.transition_queue.popleft()
        else:
            return False
        
    def pop_batch_from_transition_queue(self):
        tran_batch = []
        while True:
            if self.transition_queue:
                tran_batch.append(self.transition_queue.popleft())
            else:
                break
        return tran_batch

    def update_curr_vi_errors(self, e, e_e, s_e):
        self.curr_vi_error = e
        self.curr_e_vi_error = e_e
        self.curr_s_vi_error = s_e
        
    def update_value_vectors(self, optimal_vectors, exploration_vectors, safe_vectors):
        self.curr_value_Vector, self.curr_Q_Matrix, self.curr_Policy_Vector = optimal_vectors
        self.curr_e_value_Vector, self.curr_e_Q_Matrix, self.curr_e_Policy_Vector = exploration_vectors
        self.curr_s_value_Vector, self.curr_s_Q_Matrix, self.curr_s_Policy_Vector = safe_vectors
        self._update_unhas_dict()
        
    def _update_unhas_dict(self):    
        for s in self.curr_value_Vector:
            if s not in self.unhash_dict_ and s not in ["end_state", "unknown_state"]:
                self.unhash_dict_[s]= unhAsh(s)

    def get_value_vectors(self):
        return (self.curr_value_Vector, self.curr_Q_Matrix, self.curr_Policy_Vector), \
               (self.curr_e_value_Vector, self.curr_e_Q_Matrix, self.curr_e_Policy_Vector), \
               (self.curr_s_value_Vector, self.curr_s_Q_Matrix, self.curr_s_Policy_Vector)

    def get_curr_vi_errors(self):
        return self.curr_vi_error, self.curr_e_vi_error, self.curr_s_vi_error
        
    def get_opt_action(self,hs):
        nn_hs = self._get_nn_hs(hs)
        return self.curr_Policy_Vector[nn_hs]
    
    def get_safe_action(self,hs):
        nn_hs = self._get_nn_hs(hs)
        return self.curr_s_Policy_Vector[nn_hs]
        
    def get_explr_action(self,hs):
        nn_hs = self._get_nn_hs(hs)
        return self.curr_e_Policy_Vector[nn_hs]
    
    def _get_nn_hs(self,hs):
        self._total_calls +=1
        pi = self.curr_Policy_Vector
        if hs in pi:
            nearest_neighbor_hs = hs
        else:
            self._nn_calls +=1
            s = unhAsh(hs)
            hm_dist_dict = {s_hat: hm_dist(s, self.unhash_dict_[s_hat]) for s_hat in pi if
                            s_hat not in ["end_state", "unknown_state"]}
            nearest_neighbor_hs = min(hm_dist_dict.keys(), key=(lambda k: hm_dist_dict[k])) if hm_dist_dict else "end_state"
        return nearest_neighbor_hs
    
    def get_state_count(self):
        return len(self.unhash_dict_)
        

@ray.remote
class ContiniousSolver(object):
    def __init__(self, shared_storage, mdp, backups_per_sync=5, backup_flags = {"expl":True, "opt":True, "safe":True}):
        self.mdp = mdp
        self.shared_storage = shared_storage
        self.backup_iter = 0
        self.backups_per_sync = backups_per_sync
        self.backup_flags = backup_flags
        self._initialize_storage()
        
    def get_mdp(self):
        return self.mdp
    
    def _initialize_storage(self):
        self.shared_storage.update_value_vectors.remote((self.mdp.vD, self.mdp.qD, self.mdp.pD),
                                                        (self.mdp.e_vD, self.mdp.e_qD, self.mdp.e_pD),
                                                        (self.mdp.s_vD, self.mdp.s_qD, self.mdp.s_pD))
        self.shared_storage.update_curr_vi_errors.remote(self.mdp.curr_vi_error,
                                                         self.mdp.e_curr_vi_error,
                                                         self.mdp.s_curr_vi_error)
        

    def start(self):

        
        st = time.time()
        backup_time, consumption_time , broad_cast_time, consumption_rate = [], [], [], []
        while True:
            if ray.get(self.shared_storage.keep_solving.remote()):

                st2 = time.time()
                tran_batch = ray.get(self.shared_storage.pop_batch_from_transition_queue.remote())
                for tran in tran_batch:
                    self.mdp.consume_transition(tran)
                consumption_time.append(time.time()-st2)
                consumption_rate.append(len(tran_batch))

                st2 = time.time()
                for _ in range(self.backups_per_sync):
                    if self.backup_flags["opt"]:
                        self.mdp.do_single_optimal_backup()
                    if self.backup_flags["expl"]:
                        self.mdp.do_single_exploration_backup()
                    if self.backup_flags["safe"]:
                        self.mdp.do_single_safe_backup()
                backup_time.append(time.time()-st2)

                st2 = time.time()
                self.shared_storage.update_value_vectors.remote((self.mdp.vD, self.mdp.qD, self.mdp.pD),
                                                                (self.mdp.e_vD, self.mdp.e_qD, self.mdp.e_pD),
                                                                (self.mdp.s_vD, self.mdp.s_qD, self.mdp.s_pD))
                self.shared_storage.update_curr_vi_errors.remote(self.mdp.curr_vi_error,
                                                                self.mdp.e_curr_vi_error,
                                                                self.mdp.s_curr_vi_error)
                broad_cast_time.append(time.time()-st2)

                self.backup_iter += 1

                ############ HouseKeeping ############
                if(self.backup_iter%100 ==0   ):
                    print("Continious Solver at:", self.backup_iter,
                          "total call speed:", 100/(time.time()-st),
                          "backups speed" , 1/mean(backup_time),
                          "consumption speed", 1/mean(consumption_time),
                          "sync speed", 1/mean(broad_cast_time),
                          "consumption_volume:", mean(consumption_rate))
                    backup_time, consumption_time, broad_cast_time, consumption_rate = [], [], [], []
                    st = time.time()

            else:
                break


                


