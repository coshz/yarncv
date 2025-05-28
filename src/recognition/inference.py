import torch 
from .utils import read_images, search_checkpoint
from .config import C


class YarnPredictor:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.load_checkpoint_()

    def __call__(self, imgs):
        return self.predict(imgs) 
    
    def predict(self, imgs, return_probs=False):
        with torch.no_grad(): logits = self.model(imgs)
        if return_probs:
            return torch.softmax(logits, 1)
        else:
            return torch.argmax(logits, 1)
    
    def predict_from_files(self, files:list[str]|list[bytes], return_probs=False):
        return self.predict(read_images(files).to(self.model.device()), return_probs)
    
    def load_checkpoint_(self, ckpt_path=''):
        if not ckpt_path: 
            ckpt_path = search_checkpoint( model_name=C.MODEL_NAME, ckpt_dir=C.CKPT_DIR, best=True)
            if not ckpt_path:
                raise FileNotFoundError(f"Checkpoint file not found.")
        ckpt = torch.load(ckpt_path, map_location=self.model.device())
        self.model.load_state_dict(ckpt['model'])