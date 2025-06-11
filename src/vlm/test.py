import pandas as pd
import os
from .inference import QwenPredictor 
from concurrent.futures import ThreadPoolExecutor


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/data-2'))


def test_acc(predicator, test_csv):

    # pred = QwenPredictor(model_name)
    # cache_f = os.path.abspath(os.path.join(os.path.dirname(__file__), './txt'))
    cache = []
    # with open(cache_f) as f:
    #     for line in f.readlines():
    #         cache.append(line.strip())
    # print(cache)
    # return 

    def process_row_(row):
        img_path, img_label = row[0], row[1]
        img_path = os.path.join(DATA_DIR, img_path)
        res = predicator(img_path)
        if int(res) != int(img_label): print("N => ", img_path)
        else: print("Y => ", img_path)
        return int(res) == int(img_label)
    
    df = pd.read_csv(os.path.join(DATA_DIR, test_csv), header=0)
    rows = list(df.itertuples(index=False))
    rows = list(filter(lambda x: x[1] <= 2, rows)) # skip labels 3 and 4
    rows = list(filter(lambda row: os.path.abspath(os.path.join(DATA_DIR, row[0])) not in cache, rows))
    with ThreadPoolExecutor(max_workers=4) as executor:
        res = list(executor.map(process_row_, rows))
    acc = sum(res) / len(res)
    print(f"Acc: {acc:.2f}")


if __name__ == "__main__":
    # model_name = 'qwen2-vl-72b-instruct' # Acc=0.58
    # model_name = 'qwen2.5-vl-72b-instruct'
    # model_name = 'qwen-vl-max-2025-04-08'
    model_name = 'qwen-vl-max-latest'
    # predictor = QwenPredictor(model_name,preset_id=3)
    # test_csv = os.path.join(DATA_DIR, 'test/yarn-test.test.csv')  # 97% 
    predictor = QwenPredictor(model_name,preset_id=4)
    test_csv = os.path.join(DATA_DIR, 'test/yarn-test.test.csv')
    # test_csv = os.path.join(DATA_DIR, 'test/tmp.csv')
    test_acc(predictor, test_csv)
    