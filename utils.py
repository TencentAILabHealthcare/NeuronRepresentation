import torch
import shutil
import math
import logging
import numpy as np
import random

def set_seed(seed):
    """
    Setting of Global Seed

    """
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # gpu

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        pth = '/'.join(filename.split('/')[:-1])
        shutil.copyfile(filename, f'{pth}/model_best.pth')

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "openselfsup".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(__name__.split('.')[0])  # i.e., openselfsup
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s - %(message)s'
    logging.basicConfig(format=format_str, level=log_level)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used. Some
            special loggers are:
            - "root": the root logger obtained with `get_root_logger()`.
            - "silent": no message will be printed.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif logger == 'root':
        _logger = get_root_logger()
        _logger.log(level, msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger != 'silent':
        raise TypeError(
            'logger should be either a logging.Logger object, "root", '
            '"silent" or None, but got {}'.format(logger))
