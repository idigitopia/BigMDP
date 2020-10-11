import argparse

def print_args(args, to_show_args=[], title="Local"):
    if not to_show_args:
        title = "All Arguments"

    header = "#" * 50 + " " * 4 + title + " " * 4 + "#" * 50
    print(header)
    iter_args = iter(list(vars(args)))  # small hack to iter 2 from a list at a time
    for arg1 in iter_args:
        if to_show_args and arg1 not in to_show_args:
            continue
        try:
            arg2 = next(iter_args)
            while to_show_args and arg2 not in to_show_args:
                arg2 = next(iter_args)
            print(arg1.ljust(30), ":", str(getattr(args, arg1)).ljust(30), arg2.ljust(30), ":", getattr(args, arg2))
        except:
            print(arg1.ljust(30), ":", str(getattr(args, arg1)).ljust(30))
    print("#" * len(header))

def get_argument_parser():
    # Get all Arguments
    parser = argparse.ArgumentParser()
    argumentDict = {}
    envArgs = parser.add_argument_group('Environment Specification')
    envArgs.add_argument("--env_name", help="environment name", type=str, default="CartPole-v1")
    envArgs.add_argument("--seed", help="choice of seed to use for single start state.", type=int, default=4444)
    argumentDict.update({"envArgs": ["env_name", "seed",]})

    # MDP Build parameters
    mdpBuildArgs = parser.add_argument_group("MDP build arguments")
    mdpBuildArgs.add_argument("--unknown_transition_reward", help="default reward for unknown transitions to absorbing state", type=int, default=-1000)
    mdpBuildArgs.add_argument("--rmax_reward", help="Default reward for RMAX reward", type=int, default= 10000)
    mdpBuildArgs.add_argument("--balanced_exploration", help="Try to go to all states equally often", type=int, default= 0)
    mdpBuildArgs.add_argument("--rmax_threshold", help="Number of travesal before annealing rmax reward", type=int, default= 2)
    mdpBuildArgs.add_argument("--MAX_S_COUNT", help="maximum state count  for gpu rewource allocation", type=int, default= 250000)
    mdpBuildArgs.add_argument("--MAX_NS_COUNT", help="maximum nest state count  for gpu rewource allocation", type=int, default=20)
    mdpBuildArgs.add_argument("--def_device", help="Default device to use for building the MDP", type=str, default= "GPU")
    mdpBuildArgs.add_argument("--fill_with", help="Define how to fill missing state actions", type=str, default="0Q_src-KNN", choices=["0Q_src-KNN", "1Q_dst-KNN","kkQ_dst-1NN", "none"])
    mdpBuildArgs.add_argument("--mdp_build_k", help="Number of Nearest neighbor to consider k", type=int, default= 1)
    mdpBuildArgs.add_argument("--knn_delta", help="Define the bias parmeter for nearest neighbor distance", type=float, default=1e-8)
    mdpBuildArgs.add_argument("--penalty_type", help="penalized predicted rewards based on the distance to the state", type=str, default="linear", choices=["none", "linear", "exponential"])
    mdpBuildArgs.add_argument("--penalty_beta", help="beta multiplyer for penalizing rewards based on distance", type=float, default= 1)
    mdpBuildArgs.add_argument("--filter_with_abstraction", help="Set to true, to filter the states to be added based on the radius.", type=int, default= 0)
    mdpBuildArgs.add_argument("--normalize_by_distance", help="set it on if the transition probabilities should be normalized by distance.", action = "store_true")
    argumentDict.update({"mdpBuildArgs": ["unknown_transition_reward", "rmax_reward", "balanced_exploration" , "rmax_threshold", "MAX_S_COUNT", "def_device", "weight_transitions",
                        "fill_with" , "mdp_build_k", "knn_delta", "penalty_type" ,"penalty_beta", "within_radius", "filter_with_abstraction", "normalize_by_distance"]})

    # MDP solve and lift up parameters
    mdpSolveArgs = parser.add_argument_group("MDP build arguments")
    mdpSolveArgs.add_argument("--gamma", help="Discount Factor for Value iteration", type=float, default= 0.99)
    mdpSolveArgs.add_argument("--slip_probability", help="Slip probability for safe policy", type=float, default= 0.1)
    mdpSolveArgs.add_argument("--target_vi_error", help="target belllman backup error for considering solved", type=float, default= 0.001)
    mdpSolveArgs.add_argument("--bellman_backup_every", help="Do a bellman backups every __k frames", type=int, default= 100)
    mdpSolveArgs.add_argument("--n_backups", help="The number of backups for every backup step", type=int, default= 10)
    argumentDict.update({"mdpSolveArgs":["gamma", "slip_probability", "target_vi_error","bellman_backup_every", "n_backups", "policy_k",]})

    # Evaluation Parameters
    evalArgs =parser.add_argument_group("Evaluation Arguments")
    evalArgs.add_argument("--eval_episode_count", help="Number of episodes to evaluate the policy", type=int, default=249)
    evalArgs.add_argument("--soft_q", help="Sample according to Q values rather than max action", action="store_true")
    evalArgs.add_argument("--smooth_with_seen", help="do nearest neighbor on seen state action pair than seen states", action="store_true")
    evalArgs.add_argument("--policy_k", help="List the lift up parameter policy_k you want to test with", nargs="+", type= int, default=[1])
    argumentDict.update({"evalArgs":["build_mdp", "load_mdp", "save_mdp","test", "generate_video", "eval_episode_count","test_path_following",
                                     "video_count", "smoothing", "soft_q", "smooth_with_seen","policy_k"]})
    # parser.add_argument("--all_gammas", help="Name of the Environment to guild", type=int, default= "[0.1 ,0.9 ,0.99 ,0.999]")

    return parser, argumentDict