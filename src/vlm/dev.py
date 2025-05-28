from transformers import AutoProcessor,Qwen2VLForConditionalGeneration
import torch
from PIL import Image

model_name = "Qwen/Qwen2-VL-2B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

image = Image.open('tac.jpg').convert("RGB")
inputs = processor(images=image, text="你猜这是什么", return_tensors="pt")
with torch.no_grad():
      outputs = model.generate(**inputs)

print(processor.decode(outputs[0]))