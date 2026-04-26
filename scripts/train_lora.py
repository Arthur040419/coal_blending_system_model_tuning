#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_config(path: str | Path) -> dict:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else ROOT / p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter for coal blending explanation.")
    parser.add_argument("--config", default="configs/qwen_lora.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name_or_path"],
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if cfg.get("use_4bit"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.get("bf16") else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if cfg.get("bf16") else torch.float16,
    )
    if cfg.get("use_4bit"):
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=int(cfg.get("lora_r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.get("target_modules"),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(
        "json",
        data_files={
            "train": resolve_path(cfg["train_file"]),
            "validation": resolve_path(cfg["eval_file"]),
        },
    )

    training_args = TrainingArguments(
        output_dir=resolve_path(cfg["output_dir"]),
        num_train_epochs=float(cfg.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(cfg.get("logging_steps", 5)),
        eval_steps=int(cfg.get("eval_steps", 20)),
        save_steps=int(cfg.get("save_steps", 20)),
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        fp16=bool(cfg.get("fp16", True)),
        bf16=bool(cfg.get("bf16", False)),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=int(cfg.get("max_seq_length", 2048)),
        packing=False,
    )
    trainer.train()
    trainer.save_model(resolve_path(cfg["output_dir"]))
    tokenizer.save_pretrained(resolve_path(cfg["output_dir"]))


if __name__ == "__main__":
    main()

