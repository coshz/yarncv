"""
inference based on Qwen 
"""
import os
from openai import OpenAI
import base64
import pandas as pd
from functools import lru_cache
import json
from ..base import PredicatorBase


presets_g = json.load(open(os.path.join(os.path.dirname(__file__), 'presets.json'), 'r'))['presets']


@lru_cache(maxsize=64)
def encode_image(img: str|bytes):
    if isinstance(img, str):
        with open(img, "rb") as f:
            img_bytes = f.read()
    elif isinstance(img, bytes):
        img_bytes = img
    else:
        raise TypeError("image type not supported: expect str or bytes")
    
    b64_str = base64.b64encode(img_bytes).decode()
    return "data:image/png;base64," + b64_str


class QwenPredictor(PredicatorBase):
    def __init__(self, model_name, preset_id=1, api_key=None):
        self.client = OpenAI(
            api_key = api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model_name
        self.prompt = presets_g[preset_id]['prompt']
        self.pairs = presets_g[preset_id]['pairs']
        self.data_dir = presets_g[preset_id]['dir']
        self.conversation = self.make_conversation(self.pairs)
    
    def predict_from_bytes(self, img: bytes) -> int:
        return self.predict_from_mime_(encode_image(img))
    
    def predict_from_path(self, img: str) -> int:
        return self.predict_from_mime_(encode_image(img))
    
    def predict_from_mime_(self, img_mime_str: str)->int:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages= [
                {"role": "system", "content": self.prompt},
                *self.conversation,
                {"role": "user",  "content": [ { "type": "image_url", "image_url": { "url": img_mime_str } } ] }
            ]
        )
        label = completion.choices[0].message.content or ""
        label = int(label) if label.isdigit() else -1
        return label
    
    def make_conversation(self, csv_or_pairs: str|list):
        def make_qa_(b64_img, label=None):
            res = [{
                "role": "user", 
                "content": [ 
                    { "type": "image_url", "image_url": { "url": b64_img  } } 
                ] 
            }]
            if label is not None: res += [ {"role": "assistant", "content": str(label)} ]
            return res
        if isinstance(csv_or_pairs, str):
            il_pairs = pd.read_csv(csv_or_pairs,header=0,dtype={'img_path':str,'label_id':int}).itertuples(index=False)
        elif isinstance(csv_or_pairs, list):
            il_pairs = [(it['image'], it['label']) for it in csv_or_pairs]
        else:
            raise TypeError("csv_or_pairs type not supported: expect str or dict")
        conversation = [qa for img, label in il_pairs for qa in make_qa_(encode_image(os.path.join(self.data_dir,img)), label)]
        return conversation