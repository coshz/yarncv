import os
import random 
from PIL import Image
import argparse
from base_ import LABELS


def make_argparser():
    parser = argparse.ArgumentParser(description="Image argumentation through rotation")
    parser.add_argument("--dir", help="dataset to argumentate")
    parser.add_argument("--out", help="output directory")
    return parser


def data_argumentation_by_rotation(data_dir, out_dir):
    for d in os.listdir(data_dir):
        if d not in LABELS.keys(): continue
        os.mkdir(os.path.join(out_dir, d))
        for f in os.listdir(os.path.join(data_dir, d)):
            if not f.endswith('.png'): continue
            img_path = os.path.join(data_dir, d, f)
            img = Image.open(img_path)
            angles = [0] + random.sample(range(1, 360), 3)
            for angle in angles:
                rotated_img = img.rotate(angle)
                rotated_img.save(os.path.join(out_dir, d, f'{f[:-4]}_R{angle:03d}.png'))


if __name__ == '__main__':
    args = make_argparser().parse_args()
    if not os.path.exists(args.out): os.makedirs(args.out)
    data_argumentation_by_rotation(args.dir, args.out)

