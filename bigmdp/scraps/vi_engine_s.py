import numpy as np
# A : Action Space
# S : State Space 

def simple_value_iteration(S, A, reward_dict, tran_dict, seed_value=None, unknown_value=0, true_action_prob=0.8,
                           beta=0.99, epsilon=0.01, workers_num=4, verbose=True):
    slip_prob = 1 - true_action_prob
    slip_action_prob = slip_prob / len(A)

    V_t = {s: 0 for s in S} if seed_value is None else seed_value
    error = float("inf")

    while error > epsilon:
        V_tplus1 = {s: 0 for s in S}
        max_vals = {s: float("-inf") for s in S}

        max_error = 0

        for s in S:
            for a in tran_dict[s]:
                expected_ns_val = 0
                for ns in tran_dict[s][a]:
                    try:
                        expected_ns_val += tran_dict[s][a][ns] * V_t[ns]
                    except:
                        expected_ns_val += tran_dict[s][a][ns] * unknown_value

                expect_s_val = reward_dict[s][a] + beta * expected_ns_val
                max_vals[s] = max(max_vals[s], expect_s_val)
                V_tplus1[s] += slip_action_prob * expect_s_val
            V_tplus1[s] += (true_action_prob - slip_action_prob) * max_vals[s]

            max_error = max(max_error, abs(V_tplus1[s] - V_t[s]))

        V_t.update(V_tplus1)
        error = max_error

        if (verbose):
            print("Error:", error)

    pi = get_pi_from_value(V_t, A, tran_dict, reward_dict, beta)

    return V_t, pi


def get_pi_from_value(V, list_of_actions, tran_dict, reward_dict, beta):
    v_max = {s: float('-inf') for s in V}
    pi = {}

    for s in V:
        for a in tran_dict[s]:
            expected_val = 0
            for ns in tran_dict[s][a]:
                try:
                    expected_val += tran_dict[s][a][ns] * V[ns]
                except:
                    expected_val += tran_dict[s][a][ns] * 0
            expect_s_val = reward_dict[s][a] + beta * expected_val
            if expect_s_val > v_max[s]:
                v_max[s] = expect_s_val
                pi[s] = a

    return pi
