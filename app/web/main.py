from typing import Union, List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException

import os
import sys 
# make fastapi happy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.api import g_predictor, qwen_predicator
from app.web.responder import ResponderBasic


app = FastAPI(tile="Yarn model service")

responder_qwen = ResponderBasic(qwen_predicator)
responder_local = ResponderBasic(g_predictor)


@app.get("/health")
async def health_check():
    return { "status": "ok" }


@app.post("/predict/qwen")
async def prediction_by_qwen(files: List[UploadFile] = File(...)):
    return await responder_qwen.respond(files)


@app.post("/predict/local")
async def prediction_by_local(files: List[UploadFile] = File(...)):
    return await responder_local.respond(files)

@app.post("/dummy")
async def prediction(files: List[UploadFile] = File(...)):
    return files[0].filename