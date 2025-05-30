from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Union, List, Dict


class ResponderBasic:
    def __init__(self, predictor):
        self.predictor = predictor

    @staticmethod
    def make_data(files, labels):
        return dict(zip(files, labels))
    
    @staticmethod
    def file_validation(files: List[UploadFile] = File(...)):
        f_valid, f_invalid= [], []
        for file in files:
            if file.content_type is None or \
            (isinstance(file.content_type,str) and not file.content_type.startswith("image/")):
                f_invalid.append(file)
            else:
                f_valid.append(file)
        return f_valid, f_invalid
    
    async def respond(self, files: List[UploadFile] = File(...)):
        f_valid, f_invalid= self.file_validation(files)
        r = {}
        if len(f_valid) > 0:
            img_bytes_list = [await file.read() for file in f_valid]
            files = [file.filename for file in f_valid]
            labels = self.predictor.predict_from_files(img_bytes_list)
            r.update({"data": self.make_data(files, labels)})
        if len(f_invalid) > 0:
            r.update({'invalid_files': [file.filename for file in f_invalid]})
        return r