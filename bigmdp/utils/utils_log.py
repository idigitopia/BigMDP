import logging
import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from bigmdp.utils import utils_directory as dh


def get_simple_log_dir_and_logger(env_name = "test_env", exp_prefix = "this_exp", EXP_PARAMS = "None", load_time_string = "None" , file_logger = False ):
    ENVNAME = env_name
    # args.video = False
    # Experiment housekeeping
    # EXP_NAME = "Model_and_QBN"
    EXP_PREFIX = exp_prefix
    EXP_NAME = ENVNAME
    EXP_PARAMS = EXP_PARAMS

    log_dirs = {}
    current_time = load_time_string if load_time_string != "None" else datetime.now().strftime(
        '%b%d_%H-%M-%S')
    log_dirs["py_log_dir"] = os.path.join('logs', 'py_logs', EXP_PREFIX,ENVNAME, EXP_PARAMS, "_" + current_time)
    log_dirs["tb_train_log_dir"] = os.path.join('logs', 'tb_logs', EXP_PREFIX,ENVNAME, EXP_PARAMS, "_" + current_time + "_train")
    log_dirs["tb_valid_log_dir"] = os.path.join('logs', 'tb_logs', EXP_PREFIX,ENVNAME,EXP_PARAMS, "_" + current_time + "_valid")
    log_dirs["tb_fitted_log_dir"] = os.path.join('logs', 'tb_logs', EXP_PREFIX,ENVNAME, EXP_PARAMS,
                                                 "_" + current_time + "_fitted")
    log_dirs["tb_vi_log_dir"] = os.path.join('logs', 'tb_logs', EXP_PREFIX,ENVNAME,EXP_PARAMS, "_" + current_time + "_vi")

    dh.create_hierarchy(log_dirs["py_log_dir"])

    # print(log_dirs["tb_train_log_dir"],"\n", log_dirs["tb_valid_log_dir"])

    # python Logging Basics
    py_logger = logging.getLogger("mylogger")
    py_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

    if file_logger:
        py_file_handler = logging.FileHandler(log_dirs["py_log_dir"] + '/all_logs.log')
        py_file_handler.setLevel(logging.DEBUG)
        py_file_handler.setFormatter(formatter)
        py_logger.addHandler(py_file_handler)

    py_stream_handler = logging.StreamHandler()
    py_stream_handler.setFormatter(formatter)
    py_stream_handler.setLevel(logging.DEBUG)

    py_logger.addHandler(py_stream_handler)

    # Tensorflow Logging Basics
    logger = {}
    logger["py_logger"] = py_logger
    logger["tb_train_logger"] = SummaryWriter(log_dir=log_dirs["tb_train_log_dir"])
    logger["tb_valid_logger"] = SummaryWriter(log_dir=log_dirs["tb_valid_log_dir"])
    logger["tb_vi_logger"] = SummaryWriter(log_dir=log_dirs["tb_vi_log_dir"])
    logger["tb_fitted_logger"] = SummaryWriter(log_dir=log_dirs["tb_fitted_log_dir"])
    # args.video = False
    return log_dirs, logger


def get_advanced_log_dir_and_logger(ROOT_FOLDER ="default", EXP_ID ="default", EXP_PARAMS ="_", load_time_string ="None", tb_log_keys = ["def_logger"]):

    log_dirs = {}
    current_time = load_time_string if load_time_string != "None" else datetime.now().strftime(
        '%b%d_%H-%M-%S')
    log_dirs["py_log_dir"] = os.path.join('logs', 'py_logs', ROOT_FOLDER,EXP_ID, EXP_PARAMS, "_" + current_time)
    for tb_log_key in tb_log_keys:
        log_dirs[tb_log_key] = os.path.join('logs', 'tb_logs', ROOT_FOLDER,EXP_ID, EXP_PARAMS, "_" + current_time + tb_log_key)

    dh.create_hierarchy(log_dirs["py_log_dir"])

    # python Logging Basics
    py_logger = logging.getLogger("mylogger")
    py_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

    py_file_handler = logging.FileHandler(log_dirs["py_log_dir"] + '/all_logs.log')
    py_file_handler.setLevel(logging.DEBUG)
    py_file_handler.setFormatter(formatter)

    py_stream_handler = logging.StreamHandler()
    py_stream_handler.setFormatter(formatter)
    py_stream_handler.setLevel(logging.DEBUG)

    py_logger.addHandler(py_file_handler)
    py_logger.addHandler(py_stream_handler)

    # Tensorflow Logging Basics
    logger = {}
    logger["py_logger"] = py_logger
    for tb_log_key in tb_log_keys:
        logger[tb_log_key] = SummaryWriter(log_dir=log_dirs[tb_log_key])
    # args.video = False
    return log_dirs, logger