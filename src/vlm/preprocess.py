import os 
import pandas as pd 
import json
import shutil
import tempfile


def create_chatml_dataset(data_dir, csv_path, zip_path):
    tmp_dir = tempfile.mkdtemp()
    jsonl_path = os.path.join(tmp_dir, "data.jsonl")

    prompt = """
Detect yarn defects: 0=normal, 1~3=abnormal. \
Given a yarn image, plase point out its label (0,1,2,3)."""
    def make_inp_res_pair(img_name, label_id):
        return \
        [{
            "role": "user",
            "content": [
                { "text": prompt },
                { "image": os.path.basename(img_path) }
            ]
        }, {
            "role": "assistant",
            "content": [
                { "text": str(label_id) }
            ]
        }]
    
    f_jsonl = open(jsonl_path, "w")
    df = pd.read_csv(csv_path, header=0,dtype={'img_path':str,'label_id':int})
    for img_path, label_id in df.itertuples(index=False):
        img_name = os.path.basename(img_path)
        shutil.copyfile(os.path.join(data_dir,img_path), os.path.join(tmp_dir,img_name))
        f_jsonl.write(json.dumps({"messages": make_inp_res_pair(img_name, label_id)}) + "\n")
    f_jsonl.close()

    shutil.make_archive(zip_path, 'zip', tmp_dir)
    shutil.rmtree(tmp_dir)


def main():
    DATA_DIR = '../../data/'

    train_csv = DATA_DIR + 'refined-2/yarn-refined-2.train.csv'
    test_csv = DATA_DIR + 'img-test/yarn-img-test.test.csv'
   
    # generate datasets
    create_chatml_dataset(DATA_DIR, train_csv, DATA_DIR + 'data-sft/train')
    create_chatml_dataset(DATA_DIR, test_csv, DATA_DIR + 'data-sft/test')


if __name__ == '__main__':
    main()
