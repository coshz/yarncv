from .grabber import Grabber
from ..common import make_logger
import argparse


def make_parser():
    parser = argparse.ArgumentParser(description="Image acquisition from camera")
    parser.add_argument("--log", help="filename of log",default="yarn-acq")
    parser.add_argument("--out", help="output directory",default="img/")
    parser.add_argument("--fps", help="frames per second", type=float, default=30)
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    logger = make_logger(args.log)
    grabber = Grabber(logger)
    grabber.start(fps=30,out_dir=args.out)