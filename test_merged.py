#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent


def resolve_path(value: str | Path) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else ROOT / path)


def default_dtype() -> torch.dtype:
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a quick generation test for the merged coal model.")
    parser.add_argument("--model-path", default="outputs/merged/qwen2.5-1.5b-coal-merged")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    model_path = resolve_path(args.model_path)
    dtype = default_dtype()
    if torch.cuda.is_available():
        device_map = "auto"
    elif torch.backends.mps.is_available():
        device_map = {"": "mps"}
    else:
        device_map = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model.eval()
    if device_map is None:
        model.to("cpu")

    prompt = """
请根据以下订单生成配煤方案：
需求量：5000吨
目标灰分：不高于18%
目标硫分：不高于0.8%
目标发热量：不低于5000 kcal/kg

可用煤种：
1. 山西低硫贫煤：灰分12.5%，硫分0.35%，发热量5100 kcal/kg，价格520元/吨，库存3000吨
2. 内蒙古长焰煤：灰分20.0%，硫分0.60%，发热量4600 kcal/kg，价格390元/吨，库存6000吨
3. 陕西高热值烟煤：灰分15.0%，硫分0.75%，发热量5600 kcal/kg，价格610元/吨，库存2000吨
4. 高硫经济煤：灰分16.5%，硫分1.50%，发热量5200 kcal/kg，价格330元/吨，库存5000吨

请输出推荐配比、使用量、预测质量、成本估算和方案解释。
"""

    messages = [
        {"role": "system", "content": "你是煤矿智能配煤系统助手。"},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )
    print(response)


if __name__ == "__main__":
    main()
