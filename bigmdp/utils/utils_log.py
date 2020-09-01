import logging
import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from bigmdp.utils import utils_directory as dh




def get_advanced_log_dir_and_logger(ROOT_FOLDER ="default", EXP_ID ="default", EXP_PARAMS ="_", load_time_string ="None", tb_log_keys = ["def_logger"]):

    log_dirs = {}
    current_time = load_time_string if load_time_string != "None" else datetime.now().strftime(
        '%b%d_%H-%M-%S')
    log_dirs["py_log_dir"] = os.path.join('logs', 'py_logs', ROOT_FOLDER,EXP_ID, EXP_PARAMS, current_time)
    for tb_log_key in tb_log_keys:
        log_dirs[tb_log_key] = os.path.join('logs', 'tb_logs', ROOT_FOLDER,EXP_ID, EXP_PARAMS,  current_time + tb_log_key)

    dh.create_hierarchy(log_dirs["py_log_dir"])

    # python Logging Basics
    py_logger = logging.getLogger("mylogger")
    py_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

    # py_file_handler = logging.FileHandler(log_dirs["py_log_dir"] + '/all_logs.log')
    # py_file_handler.setLevel(logging.DEBUG)
    # py_file_handler.setFormatter(formatter)

    py_stream_handler = logging.StreamHandler()
    py_stream_handler.setFormatter(formatter)
    py_stream_handler.setLevel(logging.DEBUG)

    # py_logger.addHandler(py_file_handler)
    py_logger.addHandler(py_stream_handler)

    # Tensorflow Logging Basics
    logger = {}
    logger["py_logger"] = py_logger
    for tb_log_key in tb_log_keys:
        logger[tb_log_key] = SummaryWriter(log_dir=log_dirs[tb_log_key])
    # args.video = False
    return log_dirs, logger