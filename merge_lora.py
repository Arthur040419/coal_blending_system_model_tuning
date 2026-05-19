#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent


def resolve_path(value: str | Path) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else ROOT / path)


def default_dtype() -> torch.dtype:
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def tokenizer_kwargs_for(model_name_or_path: str, local_files_only: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
        "use_fast": True,
    }
    cfg_path = Path(model_name_or_path) / "tokenizer_config.json"
    if not cfg_path.exists():
        return kwargs
    try:
        tokenizer_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return kwargs
    if isinstance(tokenizer_cfg.get("extra_special_tokens"), list):
        kwargs["extra_special_tokens"] = {}
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge the coal LoRA adapter into the Qwen base model.")
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model id or local base-model path.",
    )
    parser.add_argument(
        "--adapter",
        default="outputs/adapters/qwen2.5-1.5b-coal-lora",
        help="LoRA adapter directory.",
    )
    parser.add_argument(
        "--output",
        default="outputs/merged/qwen2.5-1.5b-coal-merged",
        help="Directory for the merged model.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load files that already exist locally.",
    )
    args = parser.parse_args()

    adapter_path = resolve_path(args.adapter)
    output_path = resolve_path(args.output)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    dtype = default_dtype()
    if torch.cuda.is_available():
        device_map = "auto"
    elif torch.backends.mps.is_available():
        device_map = {"": "mps"}
    else:
        device_map = None

    print("正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        **tokenizer_kwargs_for(args.base_model, args.local_files_only),
    )

    print("正在加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    print("正在加载 LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        local_files_only=args.local_files_only,
    )

    print("正在合并 LoRA...")
    model = model.merge_and_unload()

    print("正在保存合并后的模型...")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print("LoRA 合并完成：", output_path)


if __name__ == "__main__":
    main()
