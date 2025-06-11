import torch 
from .utils import read_images, search_checkpoint
from .config import C
from ..base import PredicatorBase
from .model import YarnModel


class YarnPredictor(PredicatorBase):
    def __init__(self, model_name:str, out_dim:int=4):
        self.model = YarnModel(model_name, out_dim)
        self.model.eval()
        self.load_checkpoint_()

    def predict_from_bytes(self, img: bytes) -> int:
        return self.predict_from_tensor_(read_images([img]).to(self.model.device()))[0]

    def predict_from_path(self, img: str) -> int:
        return self.predict_from_tensor_(read_images([img]).to(self.model.device()))[0]
    
    def predict_from_tensor_(self, imgs: torch.Tensor):
        with torch.no_grad():
            return self.model.predict(imgs.to(self.model.device()))

    def load_checkpoint_(self, ckpt_path=''):
        if not ckpt_path: 
            ckpt_path = search_checkpoint(model_name=C.MODEL_NAME, ckpt_dir=C.CKPT_DIR, best=True)
            if not ckpt_path:
                raise FileNotFoundError(f"Checkpoint file not found.")
        ckpt = torch.load(ckpt_path, map_location=self.model.device())
        self.model.load_state_dict(ckpt['model'])