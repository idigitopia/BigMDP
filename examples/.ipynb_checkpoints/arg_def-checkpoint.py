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

    # Houskeeping Parameters
    houseKeepingArgs = parser.add_argument_group("House Keeping Parameters")
    houseKeepingArgs.add_argument("--project", help="Name of the project(used for wandb logging)", type=str, default="Batch_Flat_Test")
    houseKeepingArgs.add_argument("--run_id", help="Set the run id for the  wandb and tensorboard logger, leave to default to configure automatically", type=str, default="default")
    houseKeepingArgs.add_argument("--ts_in_run_id", help = "set to 0 if you want to see time_stamp in run id", type=int, default=1)
    houseKeepingArgs.add_argument("--exp_meta", help="Set something sensible for a simple experiment with a small number of runs, used for Sweep generation", type=str, default="Default_Exp_Name")
    houseKeepingArgs.add_argument("--exp_id", help="Used for grouping the runs, add a hash of the experiment used for sweep generation", type=str, default="E404")
    houseKeepingArgs.add_argument("--run_number", help="appended in run id if number set to bigger than 1", type=int, default = 0)
    # houseKeepingArgs.add_argument("--load_time_string", help="ime string to load.", type=str, default="none")
    houseKeepingArgs.add_argument("--run_name", help="Additional Houskeeping paramter for atari experiments.", type=str, default="none")
    # houseKeepingArgs.add_argument("--log_every", help="log the running rewards every __ frames", type=int, default=1000)
    # houseKeepingArgs.add_argument("--evaluate_every", help="Evaluate the policy every __ frames", type=int, default=10000)
    houseKeepingArgs.add_argument("--log_wandb", help="Train ppo agent", action="store_true")
    # houseKeepingArgs.add_argument("--train_eval_episodes", help="Name of the Environment to guild", type=int, default=9)
    argumentDict.update({"houseKeepingArgs": ["project", "run_id", "exp_id", "exp_meta","run_name", "ts_in_run_id", "run_number", "experiment_meta", "log_wandb"]})

    # Environment  arguments
    envArgs = parser.add_argument_group('Environment Specification')
    envArgs.add_argument("--env_name", help="environment name", type=str, default="CartPole-v1")
    envArgs.add_argument("--max_episode_length", help="specify the maximum episodle length", type=int, default=250)
    envArgs.add_argument("--single_start_state", help="set this to set starting state distribution to singular", action="store_true")
    envArgs.add_argument("--seed", help="choice of seed to use for single start state.", type=int, default=4444)
    envArgs.add_argument("--stack_count", help="Choice of the number of frames to stack", type=int, default=4)
    envArgs.add_argument("--skip_count", help="choies of every _th frame to stack", type=int, default=1)
    envArgs.add_argument("--action_repeat", help="Action repeat for environment", type=int, default=1)
    envArgs.add_argument("--image_size", help="Reshaped Image size for the Environment for gymImage-v? environments", type = int, nargs="+", default=[84, 84])
    envArgs.add_argument("--random_goal_pos", help="set this to set the goal position to random place in the room for mini gridworld", action="store_true")
    envArgs.add_argument("--layout", help="choose layout of the environment", type= str, default="Simple")
    envArgs.add_argument("--crop_frame", help="set to crop the frame", action="store_true")

    # parser.add_argument("--shaped_reward", help="size", type=int, default=0)
    argumentDict.update({"envArgs": ["env_name", "max_episode_length", "single_start_state", "seed", "stack_count", "skip_count", "action_repeat", "image_size", "random_goal_pos","layout"]})

    # Seed Dataset Parameters
    dataArgs = parser.add_argument_group('Dataset Specification')
    dataArgs.add_argument("--algorithm", help="Choose algorithm", type=str, default="DQN", choices=["DDQN","DQN", "PPO"])
    dataArgs.add_argument("--gen_data", help="set it on if the dataset is to be generated, previously stored dataset will be discarded.", action = "store_true")
    dataArgs.add_argument("--gen_frame_count", help="Total size of the dataset to be generated, will be saves as one big dump", type = int, default=int(2e5))
    dataArgs.add_argument("--dont_load_dataset", help="set it on if the dataset is not to be loaded", action = "store_true")
    dataArgs.add_argument("--data_version", help="Name of the Dataset to be loaded", type=str, default="batch_rl" )
    dataArgs.add_argument("--data_sub_version", help="Name of the Dataset sub version to be loaded", type=str, default="OnePercent" )
    dataArgs.add_argument("--data_suffices",  help="int offsets for the data", type = int, nargs="+", default=[1])
    dataArgs.add_argument("--sample_size", help="Percentage of Dataset to be sampled for the later pipeline, default is 1", type=float, default=1)
    dataArgs.add_argument("--test_size", help="Percentage of Dataset to be used as test dataset, default is 0.3 ", type=float, default=0.3)
    dataArgs.add_argument("--split_on_episodes", help="Set to 1 if we want to fitler sample the dataset based on episodes", type=int, default=0)
    dataArgs.add_argument("--data2oracle", help="convert the observation datset to oracle state dataset", action = "store_true")
    argumentDict.update({"dataArgs":["algorithm","gen_data","gen_frame_count", "dont_load_dataset", "data_version", "data_suffices", "sample_size", "test_size", "split_on_episodes", "data2oracle", "data_sub_version"]})

    # Prediction function Parameters
    predFxnArgs = parser.add_argument_group("Prediction function Arguments")
    predFxnArgs.add_argument("--ns_prediction", help="get the next state prediction from __?", type=str, default="oracle", choices=["oracle" ,"model", "non_parametric"])
    predFxnArgs.add_argument("--r_prediction", help="get the reward prediction from __?", type=str, default="oracle", choices=["oracle", "model", "non_parametric"])
    predFxnArgs.add_argument("--d_prediction", help="get the next state prediction from  __?", type=str, default="oracle", choices=["oracle", "model", "non_parametric"])
    argumentDict.update({"predFxnArgs":["ns_prediction", "r_prediction", "d_prediction"]})

    # MDP Build parameters
    mdpBuildArgs = parser.add_argument_group("MDP build arguments")
    mdpBuildArgs.add_argument("--unknown_transition_reward", help="default reward for unknown transitions to absorbing state", type=int, default=-1000)
    mdpBuildArgs.add_argument("--rmax_reward", help="Default reward for RMAX reward", type=int, default= 10000)
    mdpBuildArgs.add_argument("--balanced_exploration", help="Try to go to all states equally often", type=int, default= 0)
    mdpBuildArgs.add_argument("--rmax_threshold", help="Number of travesal before annealing rmax reward", type=int, default= 2)
    mdpBuildArgs.add_argument("--MAX_S_COUNT", help="maximum state count  for gpu rewource allocation", type=int, default= 250000)
    mdpBuildArgs.add_argument("--MAX_NS_COUNT", help="maximum nest state count  for gpu rewource allocation", type=int, default=20)
    mdpBuildArgs.add_argument("--def_device", help="Default device to use for building the MDP", type=str, default= "GPU")
    mdpBuildArgs.add_argument("--weight_transitions", help="Caluclate transition prob based on transition frequencies?", type=int, default= 1)
    mdpBuildArgs.add_argument("--weight_neighbors", help="Caluclate transition prob based on neighbor distances ?", type=int, default= 1)
    mdpBuildArgs.add_argument("--fill_with", help="Define how to fill missing state actions", type=str, default="0Q_src-KNN", choices=["0Q_src-KNN", "1Q_dst-KNN","kkQ_dst-1NN", "none"])
    mdpBuildArgs.add_argument("--mdp_build_k", help="Number of Nearest neighbor to consider k", type=int, default= 1)
    mdpBuildArgs.add_argument("--knn_delta", help="Define the bias parmeter for nearest neighbor distance", type=float, default=1e-8)
    mdpBuildArgs.add_argument("--penalty_type", help="penalized predicted rewards based on the distance to the state", type=str, default="linear", choices=["none", "linear", "exponential"])
    mdpBuildArgs.add_argument("--penalty_beta", help="beta multiplyer for penalizing rewards based on distance", type=float, default= 1)
    mdpBuildArgs.add_argument("--within_radius", help="Radius to cap the prediction to absorbing state", type=int, default= 100)
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

    # Network parameters
    netDefArgs =parser.add_argument_group("Network Definition arguments")
    netDefArgs.add_argument("--hidden_state_size", help="Size of hidden state for MLP", type=int, default= 64)
    netDefArgs.add_argument("--encoder_type", help = "set to one of the choices to change network encoder type", type = str, choices = ["conv", "conv_bn", "conv_small", "linear", "none"], default = "none")
    netDefArgs.add_argument("--bottleneck_size", help="Size of the latent space.", type=int, default= 16)
    netDefArgs.add_argument("--do_pca", help="set to do principal component analysis when using dqn representation, peertinent for Atari for now",action = "store_true")
    netDefArgs.add_argument("--latent_type", help="Set appropriately amont the choices contrastive and Q learning", type = str, default="Contrastive") # none simply means that it cannot be trained, loss function wont be defined, used for random projection
    argumentDict.update({"netDefArgs":["hidden_state_size", "encoder_type", "bottleneck_size","learnt_latent", "latent_type", ]})

    # Network Train Parameters
    netTrainArgs =parser.add_argument_group("Network Train arguments")
    netTrainArgs.add_argument("--load_from_library", help="Load the network from a separate library of networks trained and cahced. ?", action = "store_true")
    netTrainArgs.add_argument("--save_to_library", help="set to 1 to save the trained network to the library. ", action = "store_true")
    netTrainArgs.add_argument("--train", help="train the network , set  to train.", action = "store_true")
    netTrainArgs.add_argument("--validate", help="validate the network , set  to validate.", action = "store_true")
    netTrainArgs.add_argument("--log_train_variables", help="set if we wuold like to track network weights as well as other network variables.", action="store_true")
    netTrainArgs.add_argument("--dropout_ratio", help="set to 0 if you dont want any dropout in the netwrok", type=float, default= 0)
    netTrainArgs.add_argument("--use_priority_buffer", help="set to use a priorirty buffer to train the model",action = "store_true")
    # netTrainArgs.add_argument("--train_loss_wts", help="Name of the Environment to guild", type=int, default=-1000)
    netTrainArgs.add_argument("--train_epochs", help="Training epochs for the network", type=int, default= 100)
    netTrainArgs.add_argument("--batch_size", help="Training BatchSize for the network", type=int, default = 64)
    netTrainArgs.add_argument("--learning_rate", help="Learning rate for the Adam optimizer", type=float, default = 3e-5)
    netTrainArgs.add_argument("--schedule_learning_rate", help="Set to schedule the learning rate for the Optimizer",action = "store_true")
    netTrainArgs.add_argument("--weight_decay_rate", help="decay rate for weight decay in Adam", type=float, default=0)
    argumentDict.update({"netTrainArgs":["load_from_library", "save_to_library", "train","validate", "log_train_variables",
                                         "dropout_ratio", "use_priority_buffer", "train_epochs",
                                         "batch_size", "learning_rate", "schedule_learning_rate", "weight_decay_rate"]})

    # Evaluation Parameters
    evalArgs =parser.add_argument_group("Evaluation Arguments")
    evalArgs.add_argument("--build_mdp", help="set to build the MDP from the data and the network",action = "store_true")
    evalArgs.add_argument("--load_mdp", help="set to load the MDP from memory",action = "store_true")
    evalArgs.add_argument("--save_mdp", help="set to save the built MDP",action = "store_true")
    evalArgs.add_argument("--test", help="set to test the policy in the environment", action = "store_true")
    evalArgs.add_argument("--test_path_following", help="set to test the policy in the environment", action = "store_true")
    evalArgs.add_argument("--video_count", help="Set to greater than 0 to log video", type=int, default=0)
    evalArgs.add_argument("--eval_episode_count", help="Number of episodes to evaluate the policy", type=int, default=249)
    evalArgs.add_argument("--smoothing", help="Use more than one K NN for lifting up the policy", action="store_true")
    evalArgs.add_argument("--soft_q", help="Sample according to Q values rather than max action", action="store_true")
    evalArgs.add_argument("--smooth_with_seen", help="do nearest neighbor on seen state action pair than seen states", action="store_true")
    evalArgs.add_argument("--policy_k", help="List the lift up parameter policy_k you want to test with", nargs="+", type= int, default=[1])
    argumentDict.update({"evalArgs":["build_mdp", "load_mdp", "save_mdp","test", "generate_video", "eval_episode_count","test_path_following",
                                     "video_count", "smoothing", "soft_q", "smooth_with_seen","policy_k"]})
    # parser.add_argument("--all_gammas", help="Name of the Environment to guild", type=int, default= "[0.1 ,0.9 ,0.99 ,0.999]")

    # Data Collection Phase Parameters
    dataColllectArgs =parser.add_argument_group("Data Collection Arguments")
    dataColllectArgs.add_argument("--online_data_collection", help="set to collect data in online fashion while trianing", action= "store_true")
    dataColllectArgs.add_argument("--inner_loop_frame_count", help="Total number of frames to collect int he data collection phase", type=int, default=100000)
    dataColllectArgs.add_argument("--update_repr_every", help="update the prepresentation every ___? frames", type=int, default= 25000)
    argumentDict.update({"dataColllectArgs":["inner_loop_frame_count", "update_repr_every" ]})


    return parser, argumentDict