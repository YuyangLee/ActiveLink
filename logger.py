import logging
import os
import sys
import time


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(asctime)s %(levelname)s  %(message)s")

    time_tag = int(time.time())
    log_dir = "logs/{0}".format(time_tag)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    file_handler = logging.FileHandler(log_dir + "/log.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    
    return log_dir, time_tag
