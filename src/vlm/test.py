import pandas as pd
import os
from .inference import QwenPredictor 
from concurrent.futures import ThreadPoolExecutor


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))


def test_acc(predicator, test_csv):

    # pred = QwenPredictor(model_name)

    def process_row_(row):
        img_path, img_label = row[0], row[1]
        img_path = os.path.join(DATA_DIR, img_path)
        if img_label > 3: img_label -= 1 
        res = predicator(img_path)
        return int(res) == int(img_label)
    
    df = pd.read_csv(os.path.join(DATA_DIR, test_csv), header=0)
    rows = list(df.itertuples(index=False))
    rows = list(filter(lambda x: x[1] <= 2, rows)) # skip labels 3 and 4
    with ThreadPoolExecutor(max_workers=4) as executor:
        res = list(executor.map(process_row_, rows))
    acc = sum(res) / len(res)
    print(f"Acc: {acc:.2f}")


if __name__ == "__main__":
    # model_name = 'qwen2-vl-72b-instruct' # Acc=0.58
    model_name = 'qwen2.5-vl-72b-instruct'
    predictor = QwenPredictor(model_name)
    test_csv = os.path.join(DATA_DIR, 'img-test/yarn-img-test.test.csv')
    test_acc(predictor, test_csv)
    