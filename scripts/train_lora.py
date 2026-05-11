#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# trl 在 import 时会用 Path.read_text() 读 UTF-8 的 .jinja 模板，不指定 encoding。
# 在中文 Windows 上默认编码为 GBK，会触发 UnicodeDecodeError；必须在导入 trl 之前打补丁。
def _patch_path_read_text_default_utf8() -> None:
    import pathlib

    _orig = pathlib.Path.read_text

    def _read_text(self: pathlib.Path, *args: object, **kwargs: object) -> str:
        if args and args[0] is not None:
            return _orig(self, *args, **kwargs)
        if args:  # encoding passed as first positional and is None
            return _orig(self, "utf-8", *args[1:], **kwargs)
        if "encoding" in kwargs and kwargs["encoding"] is not None:
            return _orig(self, **kwargs)
        return _orig(self, **{**kwargs, "encoding": "utf-8"})

    pathlib.Path.read_text = _read_text  # type: ignore[assignment]


_patch_path_read_text_default_utf8()

try:
    import torch
    import yaml
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer
except ModuleNotFoundError as exc:
    missing = exc.name or str(exc)
    raise SystemExit(
        f"Missing Python dependency: {missing}\n"
        "Run this script from the project virtual environment:\n"
        "  source .venv/bin/activate\n"
        "  python scripts/train_lora.py --config configs/qwen_lora.yaml\n"
        "or run it directly with:\n"
        "  .venv/bin/python scripts/train_lora.py --config configs/qwen_lora.yaml\n"
        "If the virtual environment is not installed yet, run:\n"
        "  python3 -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  pip install -r requirements.txt"
    ) from exc

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


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _print_pytorch_gpu_diagnostics() -> None:
    """Help when a machine has an NVIDIA GPU but this process still uses CPU (wrong PyTorch wheel, driver, etc.)."""
    cver = torch.version.cuda
    print("PyTorch GPU diagnostics (if you have an NVIDIA GPU but see cpu above):")
    print(f"  torch.__version__         = {torch.__version__}")
    print(
        f"  torch.version.cuda         = {cver!r}"
        + (
            "  -> None usually means a CPU-only PyTorch install; reinstall torch with CUDA (see https://pytorch.org)."
            if cver is None
            else ""
        )
    )
    print(f"  torch.cuda.is_available()  = {torch.cuda.is_available()}")
    if cver is not None and not torch.cuda.is_available():
        print("  (CUDA build present but GPU not available: update NVIDIA driver, or check `nvidia-smi` in a terminal.)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter for coal blending.")
    parser.add_argument("--config", default="configs/qwen_lora.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = detect_device()
    print(f"Detected device: {device}")
    if device == "cpu":
        _print_pytorch_gpu_diagnostics()
    if device == "cpu" and not cfg.get("allow_cpu_training", False):
        raise SystemExit(
            "No CUDA/MPS device is available. Refusing to train on CPU by default because Qwen LoRA "
            "training will be extremely slow or may run out of memory. Use a CUDA/MPS environment, "
            "or set allow_cpu_training: true in the config for a tiny-model smoke test."
        )

    # ── Tokenizer ──────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name_or_path"],
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model loading (CUDA QLoRA / MPS fp16 / CPU fp32) ──────
    use_4bit = cfg.get("use_4bit", False) and device == "cuda"
    quantization_config = None
    torch_dtype = torch.float32

    if use_4bit:
        compute_dtype = torch.bfloat16 if cfg.get("bf16") else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        torch_dtype = compute_dtype
    elif device == "mps":
        torch_dtype = torch.float16
    elif device == "cuda":
        torch_dtype = torch.float16

    model_kwargs: dict = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    if use_4bit:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    elif device == "mps":
        model_kwargs["device_map"] = {"": "mps"}
    elif device == "cuda":
        model_kwargs["device_map"] = "auto"

    print(f"Loading model with dtype={torch_dtype}, 4bit={use_4bit}, device={device}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        **model_kwargs,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # ── LoRA ───────────────────────────────────────────────────
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

    # ── Dataset ────────────────────────────────────────────────
    dataset = load_dataset(
        "json",
        data_files={
            "train": resolve_path(cfg["train_file"]),
            "validation": resolve_path(cfg["eval_file"]),
        },
    )

    # ── Training args ──────────────────────────────────────────
    training_args = SFTConfig(
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
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        fp16=(device != "cpu" and not cfg.get("bf16") and not use_4bit),
        bf16=bool(cfg.get("bf16", False)),
        report_to="none",
        remove_unused_columns=False,
        max_length=int(cfg.get("max_seq_length", 2048)),
        packing=False,
        # Qwen chat templates differ across model releases. Keep this configurable because
        # assistant_only_loss=True requires a chat template with assistant token masks.
        assistant_only_loss=bool(cfg.get("assistant_only_loss", False)),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(resolve_path(cfg["output_dir"]))
    tokenizer.save_pretrained(resolve_path(cfg["output_dir"]))


if __name__ == "__main__":
    main()
