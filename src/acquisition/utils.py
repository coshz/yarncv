import logging 
import os
from datetime import datetime
import time


def make_logger(
    log_name, 
    log_level=logging.DEBUG,
    cmd_level=logging.INFO,
    formatter=logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S")
):
    """
    @desc: Create a logger associated with a file and terminal 
        - log_name: file name 
        - log_level, cmd_level: logging level for file, terminal 
        - formatter: logging format
    """
    fh = logging.FileHandler(f"{log_name}.log")
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setLevel(cmd_level)
    sh.setFormatter(formatter)

    logger = logging.Logger(log_name)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def make_timestamp():
    # now = datetime.now()
    msts = round(1000 * time.time())
    # ts = now.strftime("%y%m%d%H%M%S") + f"{now.microsecond // 1000 :03d}"
    return str(msts)


def make_image_name(width, height, frame_num, timestamp, ext):
    return f"Image_w{width}_h{height}_fn{frame_num}_{timestamp}.{ext}"


if __name__ == '__main__':
    print(make_timestamp())