from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

base_model_path = r"D:\models\Qwen3-4B-Instruct-2507"
adapter_path = r"D:\code\graduation_project\coal_blending_system_model_tuning\outputs\adapters\qwen-coal-lora"
merged_model_path = r"D:\code\graduation_project\coal_blending_system_model_tuning\outputs\merged\qwen3-coal-merged"

os.makedirs(merged_model_path, exist_ok=True)

print("正在加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    local_files_only=True
)

print("正在加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

print("正在加载 LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    local_files_only=True
)

print("正在合并 LoRA...")
model = model.merge_and_unload()

print("正在保存合并后的模型...")
model.save_pretrained(
    merged_model_path,
    safe_serialization=True
)

tokenizer.save_pretrained(merged_model_path)

print("LoRA 合并完成：", merged_model_path)