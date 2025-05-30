import os 
import yaml


class C:
    OUT_DIR     = './out/'                          # relative to `__file__`
    LOG_DIR     = "log/"                            # relative to OUT_DIR
    CKPT_DIR    = "ckpt/"                           # relative to OUT_DIR

    # data parameters 
    DATA_DIR    = "../../data/"                     # relative to `__file__`
    DATA_TRAIN  = "refined/yarn-refined.train.csv"  # relative to DATA_DIR
    DATA_TEST   = "refined/yarn-refined.test.csv"   # relative to DATA_DIR

    # model parameters
    MODEL_NAME = None
    
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
        cls.postprocess()
    
    @classmethod
    def postprocess(cls):
        def validate_dirs(*ds):
            for d in ds: 
                if not os.path.exists(d): 
                    os.makedirs(d)
        def validate_files(*fs):
            for f in fs:
                if not os.path.exists(f): 
                    raise FileNotFoundError(f"file not found at: \"{os.path.abspath(f)}\"")
        
        C.OUT_DIR = os.path.join(os.path.dirname(__file__), C.OUT_DIR)
        C.DATA_DIR = os.path.join(os.path.dirname(__file__), C.DATA_DIR)
        C.LOG_DIR = os.path.join(C.OUT_DIR, C.LOG_DIR)
        C.CKPT_DIR = os.path.join(C.OUT_DIR, C.CKPT_DIR)
        C.DATA_TRAIN = os.path.join(C.DATA_DIR, C.DATA_TRAIN)
        C.DATA_TEST = os.path.join(C.DATA_DIR, C.DATA_TEST)

        # validate_dirs(C.OUT_DIR, C.LOG_DIR, C.CKPT_DIR)
        # validate_files(C.DATA_TRAIN, C.DATA_TEST)

C.init_from(os.path.join(os.path.dirname(__file__), "config/default.yaml"))
