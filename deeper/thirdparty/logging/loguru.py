import sys

from loguru import logger

from deeper.utils.dist import is_main_process


def log_in_main_process_only(record):
    return is_main_process()


def get_default_logger():
    logger.remove()
    logger.add(sys.stdout, filter=log_in_main_process_only, format='[{time:YYYY-MM-DD HH:mm:ss} '
                                                                   '{file} line {line}] {message}')
    return logger


def get_colored_logger():
    logger.remove()
    logger.add(sys.stdout, filter=log_in_main_process_only, format='<green>[{time:YYYY-MM-DD HH:mm:ss}</green> '
                                                                   '<cyan>{file} line {line}]</cyan> {message}')
    return logger


def get_custom_logger(**kwargs):
    logger.remove()
    logger.add(**kwargs)
    return logger


default_logger = get_colored_logger()
