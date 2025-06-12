import pandas as pd
import os 

from concurrent.futures import ThreadPoolExecutor
from .inference import YarnPredictor


def test_acc(predicator, data_dir, test_csv):
    """
    Desc: test_csv records all lines (img_path, label_id) where test_csv and img_path are relative paths to data_dir
    """
    def process_row_(row):
        img_path, img_label = row[0], row[1]
        img_path = os.path.join(data_dir, img_path)
        res = predicator(img_path)
        if int(res) != int(img_label): print("N => ", img_path, '=>', res)
        else: print("Y => ", img_path)
        return int(res) == int(img_label)
    
    df = pd.read_csv(os.path.join(data_dir, test_csv), header=0)
    rows = list(df.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=4) as executor:
        res = list(executor.map(process_row_, rows))
    acc = sum(res) / len(res)
    print(f"Acc: {acc:.2f}")


if __name__ == '__main__':
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/data-2'))
    test_csv = 'test/yarn-test.test.csv'
    # predicator = YarnPredictor('yarn_sim2',out_dim=4)
    # test_csv = os.path.join(DATA_DIR, 'test/yarn-test.test.csv')
    predicator = YarnPredictor('yarn_sim2',out_dim=3)
    # test_csv = os.path.join('/Volumes/Gen/repo/yarn/0605/img/s2', 'refined@s2/yarn-refined@s2.test.csv')
    test_acc(predicator, data_dir, test_csv)