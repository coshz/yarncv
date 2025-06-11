import logging 


def make_logger(
    log_name, 
    log_level=logging.DEBUG,
    cmd_level=logging.INFO,
    formatter=logging.Formatter(
        "%(asctime)s - %(levelname)s: %(message)s",
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