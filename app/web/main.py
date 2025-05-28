from typing import Union, List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException

import os
import sys 
# make fastapi happy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from app.api import g_predictor


app = FastAPI(tile="Yarn model service")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health_check():
    return { "status": "ok" }


@app.post("/predict")
async def predict_images(files: List[UploadFile] = File(...)):
    resp = {}
    f_valid, f_invalid= [], []
    for file in files:
        if file.content_type is None or \
        (isinstance(file.content_type,str) and not file.content_type.startswith("image/")):
            f_invalid.append(file)
        else:
            f_valid.append(file)
    if len(f_valid) > 0:
        data: Dict[str,int]=dict()
        img_bytes_list = [await file.read() for file in f_valid]
        labels = g_predictor.predict_from_files(img_bytes_list, return_probs=False)
        for file, label in zip(f_valid, labels):
            # data[file.filename] = label.item()
            data.update({ f"{file.filename}": int(label.item())})
        resp.update({"data": data})
    if len(f_invalid) > 0:
        resp.update({'invalid_files': [file.filename for file in f_invalid]})
    return resp

