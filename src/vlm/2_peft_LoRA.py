"""
Parameter efficient fine-tuning (based on LoRA)
"""

from peft import LoraConfig 
from trl import AutoModelForCausalLMWithValueHead

model_id = 'Qwen/Qwen2-VL-2B'

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.5,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_id, torch_dtype='auto', device_map='auto', lora=lora_config
)

# 
# python trl/scripts/sft.py --output_dir out/ --model_name Qwen/Qwen2-VL-2B --dataset_name ../../data/data-sft/train --load_in_4bit --use_peft --per_device_train_batch_size 4 --gradient_accumulation_steps 2