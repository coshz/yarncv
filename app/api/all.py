from src.recognition import YarnPredictor
from src.vlm import QwenPredictor 
import yaml


def init():
    global local_predictor
    global qwen_predictor

    with open('app/config.yaml') as f:
        cfg = yaml.safe_load(f)

    local_predictor = YarnPredictor(cfg['model_name'], cfg['model_out_dim'])

    qwen_predictor = QwenPredictor(cfg['qwen_name'])

init()