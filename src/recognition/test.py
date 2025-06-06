import pandas as pd
import os 

from concurrent.futures import ThreadPoolExecutor
from .inference import YarnPredictor
from .model import YarnModel


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/data-2'))


def test_acc(predicator, test_csv):

    def process_row_(row):
        img_path, img_label = row[0], row[1]
        img_path = os.path.join(DATA_DIR, img_path)
        res = predicator.predict_from_files([img_path])[0]
        if int(res) != int(img_label): print("N => ", img_path, '=>', res)
        else: print("Y => ", img_path)
        return int(res) == int(img_label)
    
    df = pd.read_csv(os.path.join(DATA_DIR, test_csv), header=0)
    rows = list(df.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=4) as executor:
        res = list(executor.map(process_row_, rows))
    acc = sum(res) / len(res)
    print(f"Acc: {acc:.2f}")


if __name__ == '__main__':
    predicator = YarnPredictor(YarnModel(model_name='yarn_sim2',out_dim=4))
    test_csv = os.path.join(DATA_DIR, 'test/yarn-test.test.csv')
    test_acc(predicator, test_csv)