import argparse
from train import YarnTrainer
from config import C


def make_argparser():
    parser = argparse.ArgumentParser(description="Image recognition")
    parser.add_argument("--out", help="output directory of log and ckpt")
    parser.add_argument("--bs", type=int, help="batch size", )
    parser.add_argument("--config", default="config/default.yaml", help="config file", )
    parser.add_argument("--model", help="model name")
    parser.add_argument("--test", nargs="?", const='', help="run in test model")
    return parser


def init_config():
    """value priorities:
    1. command line args
    2. config file
    3. default values
    """
    args = make_argparser().parse_args()
    C.init_from(args.config)
    if args.model: C.MODEL_NAME = args.model
    if args.bs: C.BATCH_SIZE = args.bs
    if args.out: C.OUT_DIR = args.out
    return args


if __name__ == '__main__':
    args = init_config()
    trainer = YarnTrainer(train=(args.test is None))
    if args.test is not None:
        trainer.test(args.test)
    else:
        trainer.train_and_validate(C.EPOCHS, C.VAL_INTERVAL, resume=True)
        trainer.test()