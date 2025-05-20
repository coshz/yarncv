import os 
import yaml

class C:
    OUT_DIR     = 'out/'
    LOG_DIR     = "log/" 
    CKPT_DIR    = "ckpt/"

    # data parameters
    DATA_DIR    = "../../data"
    DATA_TRAIN  = "refined/yarn-refined.train.csv"
    DATA_TEST   = "refined/yarn-refined.test.csv"

    # model parameters
    MODEL_NAME = "efficientnet"
    
    # train parameters
    EPOCHS = 100
    LEARNING_RATE = 1E-3
    BATCH_SIZE = 4
    VAL_INTERVAL = 2

    MODE = 'train' # train | test

    @classmethod
    def init_from(cls,config):
        if not os.path.exists(config):
            raise FileNotFoundError(f"Config file not found at: \"{os.path.abspath(config)}\"")
        
        # load from config file
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            key = k.upper()
            if hasattr(C, key):
                setattr(C, key, v)
            else:
                raise KeyError(f"Config key {key} not found in class C.")
        
        # creating & absolute directories
        def ca_(*ds): 
            d = os.path.join(*ds)
            if not os.path.exists(d): os.makedirs(d)
            return os.path.abspath(d)
        cls.LOG_DIR = ca_(cls.OUT_DIR, cls.LOG_DIR)
        cls.CKPT_DIR = ca_(cls.OUT_DIR, cls.CKPT_DIR)

        # validate data csv 
        cls.DATA_TRAIN = os.path.join(cls.DATA_DIR, cls.DATA_TRAIN)
        cls.DATA_TEST = os.path.join(cls.DATA_DIR, cls.DATA_TEST)
        if not os.path.exists(cls.DATA_TRAIN):
            raise FileNotFoundError(f"train data not found at: \"{os.path.abspath(cls.DATA_TRAIN)}\"")
        if not os.path.exists(cls.DATA_TEST):
            raise FileNotFoundError(f"test data not found at: \"{os.path.abspath(cls.DATA_TEST)}\"")