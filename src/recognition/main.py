import argparse
from .dataset import create_yarn_dataloaders
from .model import YarnModel
from .train import YarnTrainer
from .inference import YarnPredictor
from .config import C


def make_argparser():
    parser = argparse.ArgumentParser(description="Yarn image recognition")
    parser.add_argument("--train", action="store_true", help="run in train mode")
    parser.add_argument("--test", nargs="?", const="", help="run in test mode from the specified checkpoint")
    parser.add_argument("--infer", nargs="+", help="inference the label(s) of the given image(s)")
    parser.add_argument("--model", help="model name")
    parser.add_argument("--out", help="output directory of log and ckpt")
    parser.add_argument("--bs", type=int, help="batch size", )
    parser.add_argument("--config", default="config/default.yaml", help="config file", )
    return parser


def init_config_from_parser(parser):
    """value priorities:
    1. command line args
    2. config file
    3. default values
    """
    args = parser.parse_args()
    if args.model: C.MODEL_NAME = args.model
    if args.bs: C.BATCH_SIZE = args.bs
    if args.out: C.OUT_DIR = args.out
    C.init_from(args.config)
    return args


if __name__ == '__main__':
    parser = make_argparser()
    args = init_config_from_parser(parser)

    if not any([args.train,args.test is not None,args.infer]):
        parser.print_help()
        exit()

    model = YarnModel(C.MODEL_NAME, 4)

    if args.infer:
        predictor = YarnPredictor(model)
        print(predictor.predict_from_files(args.infer).tolist())
    else:
        dataloaders = create_yarn_dataloaders(C.DATA_DIR, C.DATA_TRAIN, C.DATA_TEST,C.BATCH_SIZE)
        trainer = YarnTrainer(model,dataloaders)
        if args.train: 
            trainer.train_and_validate(C.EPOCHS, C.VAL_INTERVAL, resume=True)
            trainer.test()
        else:
            trainer.test(args.test) 