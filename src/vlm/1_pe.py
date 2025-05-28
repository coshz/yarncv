"""
:Prompt engineering
"""
import os
from openai import OpenAI
import base64
import pandas as pd
import ast
from concurrent.futures import ThreadPoolExecutor


def encode_image(img_path):
    with open(img_path, "rb") as img_file:
        base64_string = base64.b64encode(img_file.read()).decode()
        return f"data:image/png;base64,{base64_string}"


def expected_response(label):
    label = int(label)
    if label == 0:
        return "[1.0, 0.0, 0.0, 0.0]"
    elif label == 1:
        return "[0.0, 1.0, 0.0, 0.0]"
    elif label == 2:
        return "[0.0, 0.0, 1.0, 0.0]"
    elif label == 3:
        return "[0.0, 0.0, 0.0, 1.0]"
    else:
        raise ValueError("Invalid label")


def conversation_preset(img_path, label=None):
    res = [{
        "role": "user", 
        "content": [ 
            { "type": "image_url", "image_url": { "url": encode_image(img_path) } }  # encode_image(img_path)
        ] 
    },]
    if label: res += [ {"role": "assistant", "content": expected_response(label)} ]
    return res
        

DATA_DIR = '../../data'
df_ = pd.read_csv('preset.csv',header=0,dtype={'img_path':str,'label_id':int})
PRESET = [ (row[0], row[1]) for row in df_.itertuples(index=False) ]


def make_client():
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return client


def infer_from_client(client, model, img_path):
    prompt = \
"""
You are an expert for detecting the exception of yarn; \
there are four classes, label 0 is normal, and 1~3 are abnormal; \
given a yarn image, answer the normalized probability 4-tuple on label (0,1,2,3).
"""
    completion = client.chat.completions.create(
        model=model, 
        messages=[
            {"role": "system", "content": prompt},
        ] 
        + sum([conversation_preset(os.path.join(DATA_DIR,img), label) for img, label in PRESET], [])
        + conversation_preset(img_path, None)
    )
    return completion.choices[0].message.content


def test_acc(client, model, test_csv):
    def process_row_(row):
        img_path, img_label = row[0], row[1]
        img_path = os.path.join(DATA_DIR, img_path)
        if img_label > 3: img_label -= 1 
        res = infer_from_client(client, model, img_path)
        max_i, max_v = max(enumerate(ast.literal_eval(res)), key=lambda x:x[1])
        return max_i == int(img_label)
    
    df = pd.read_csv(os.path.join(DATA_DIR, test_csv), header=0)
    rows = list(df.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=4) as executor:
        res = list(executor.map(process_row_, rows))
    acc = sum(res) / len(res)
    print(f"Acc: {acc:.2f}")


def main():
    client = make_client() 
    model = "qwen-vl-max"
    test_csv = 'img-test/yarn-img-test.test.csv'
    test_acc(client, model, test_csv)


if __name__ == "__main__":
    main()
    