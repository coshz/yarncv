import logging 
import torch
import os
import re
import random
from torchvision import transforms
from PIL import Image
from io import BytesIO


def img_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


def img_augment():
    return transforms.Compose([
        transforms.RandomRotation(random.randint(0, 360))
    ])


def read_image(img:str|bytes, aug=False)->torch.Tensor:
    if isinstance(img, str):
        data = Image.open(img).convert("RGB")
    elif isinstance(img, bytes):
        data = Image.open(BytesIO(img)).convert("RGB")
    else:
        raise TypeError("image type not supported: expect str or bytes")
    
    data = img_transform()(data)
    if aug: data = img_augment()(data)
    return data


def read_images(imgs: list[str]|list[bytes], aug=False):
    return torch.stack([read_image(img, aug) for img in imgs])


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def search_checkpoint(model_name, ckpt_dir, best=False) -> str | None:    
    """search best or latest checkpoint"""
    ckpt_file = None
    ckpt_regex = rf"{model_name}-(\d+\.\d+).pth"
    if best: 
        best_file, best_acc = None, 0.0
        for file in os.listdir(ckpt_dir):
            ckpt_match = re.search(ckpt_regex, file)
            if ckpt_match:
                acc = float(ckpt_match.group(1))
                if acc > best_acc:
                    best_acc = acc
                    best_file = os.path.join(ckpt_dir, file)
        ckpt_file = best_file
    else:
        latest_file, latest_time = None, 0
        for file in os.listdir(ckpt_dir):
            ckpt_match = re.search(ckpt_regex, file)
            if not ckpt_match: continue
            file_time = os.path.getmtime(os.path.join(ckpt_dir, file))
            if file_time > latest_time:
                latest_time = file_time
                latest_file = os.path.join(ckpt_dir, file)
        ckpt_file = latest_file
    return os.path.join(ckpt_dir, ckpt_file) if ckpt_file else None