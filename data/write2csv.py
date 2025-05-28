import pandas as pd
import os
import random 
import argparse
from base_ import LABELS


def make_argparser():
    parser = argparse.ArgumentParser(description="Generate CSV descripting the dataset")
    parser.add_argument("--dir", help="directory of dataset")
    return parser


def write2csv(data_dir, out_dir=None, test_ratio=0.1):
    """split and write (img_path,label_id) to CSV files"""

    def split_sample(lst, ratio):
        sampled = random.sample(lst, int(len(lst) * ratio))
        remained = list(set(lst) - set(sampled))
        return sampled, remained 
    
    l_train,l_test = [],[]
    out_dir = out_dir or data_dir

    for d in sorted(os.listdir(data_dir)):
        if d not in LABELS.keys(): continue
        label_id = LABELS[d][0]
        lst = sorted(os.listdir(os.path.join(data_dir, d)))
        sampled, remained = split_sample(lst, test_ratio)
        excluded = ['.DS_Store', 'Thumbs.db']
        l_test.extend([(os.path.join(data_dir,d,s), label_id) for s in sampled if s not in excluded ])
        l_train.extend((os.path.join(data_dir,d,r), label_id) for r in remained if r not in excluded)
    
    csv_name = f"yarn-{os.path.basename(data_dir)}.{{mode}}.csv"
    train_csv = os.path.join(out_dir, csv_name.format(mode='train'))
    test_csv = os.path.join(out_dir, csv_name.format(mode='test'))

    pd.DataFrame(l_train, columns=['img_path','label_id']).to_csv(train_csv,index=False)
    pd.DataFrame(l_test, columns=['img_path','label_id']).to_csv(test_csv,index=False)


if __name__ == '__main__':
    args = make_argparser().parse_args()
    write2csv(args.dir, out_dir=args.dir, test_ratio=1)