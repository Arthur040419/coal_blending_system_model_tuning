#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import os
import sys
from pathlib import Path
from typing import Any

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
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer
except ModuleNotFoundError as exc:
    missing = exc.name or str(exc)
    if os.name == "nt":
        commands = (
            "PowerShell:\n"
            "  py -m venv .venv\n"
            "  .\\.venv\\Scripts\\Activate.ps1\n"
            "  python -m pip install --upgrade pip\n"
            "  pip install -r requirements.txt\n"
            "  python scripts/train_lora.py --config configs/qwen_lora.yaml\n"
            "or run it directly with:\n"
            "  .\\.venv\\Scripts\\python.exe scripts\\train_lora.py --config configs\\qwen_lora.yaml"
        )
    else:
        commands = (
            "macOS/Linux:\n"
            "  python3 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  python -m pip install --upgrade pip\n"
            "  pip install -r requirements.txt\n"
            "  python scripts/train_lora.py --config configs/qwen_lora.yaml\n"
            "or run it directly with:\n"
            "  .venv/bin/python scripts/train_lora.py --config configs/qwen_lora.yaml"
        )
    raise SystemExit(f"Missing Python dependency: {missing}\n{commands}") from exc

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


def resolve_model_name_or_path(value: str | Path) -> str:
    raw = str(value)
    # Keep HuggingFace ids and Windows drive paths untouched.
    if "/" in raw and not raw.startswith(".") and not raw.startswith("/"):
        return raw
    if len(raw) >= 3 and raw[1:3] == ":\\":
        return raw
    p = Path(raw)
    if p.is_absolute():
        return raw
    local = ROOT / p
    return str(local) if local.exists() else raw


def _exception_chain_contains(exc: BaseException, text: str) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if text in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def _tokenizer_kwargs_for(model_name_or_path: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"trust_remote_code": True, "use_fast": True}
    cfg_path = Path(model_name_or_path) / "tokenizer_config.json"
    if not cfg_path.exists():
        return kwargs
    try:
        tokenizer_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return kwargs
    if isinstance(tokenizer_cfg.get("extra_special_tokens"), list):
        # Some Qwen3 exports contain the old list form, while recent
        # transformers expects a dict here. The tokens are already present in
        # tokenizer.json; overriding this metadata avoids a loader crash.
        kwargs["extra_special_tokens"] = {}
    return kwargs


def load_tokenizer(model_name_or_path: str):
    kwargs = _tokenizer_kwargs_for(model_name_or_path)
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
    except Exception as exc:
        if _exception_chain_contains(exc, "extra_special_tokens") or _exception_chain_contains(
            exc, "object has no attribute 'keys'"
        ):
            retry_kwargs = {**kwargs, "extra_special_tokens": {}}
            print("Retrying tokenizer load with extra_special_tokens={} for compatibility.")
            return AutoTokenizer.from_pretrained(model_name_or_path, **retry_kwargs)
        raise


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


def filter_dataset_by_task(dataset: Any, task: str | None) -> Any:
    if not task:
        return dataset
    print(f"Filtering dataset by task={task!r}")
    filtered = dataset.filter(lambda row: row.get("task") == task)
    for split, rows in filtered.items():
        if len(rows) == 0:
            raise SystemExit(f"No rows left in split {split!r} after filtering task={task!r}.")
    return filtered


def prepare_prompt_completion_dataset(dataset: Any) -> Any:
    """Convert local JSONL rows to TRL prompt/completion format.

    The dataset stores prompt/completion as chat messages for readability. For
    training, convert them to explicit Qwen chat text so we do not depend on the
    downloaded tokenizer directory containing a compatible chat_template.jinja.
    TRL will still use completion_mask so loss is applied only to the assistant
    answer.
    """

    def render_message(message: dict[str, Any]) -> str:
        role = message.get("role")
        content = message.get("content", "")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported chat role: {role!r}")
        return f"<|im_start|>{role}\n{content}<|im_end|>\n"

    def render_prompt(messages: list[dict[str, Any]]) -> str:
        return "".join(render_message(message) for message in messages) + "<|im_start|>assistant\n"

    def render_completion(messages: list[dict[str, Any]]) -> str:
        if len(messages) != 1 or messages[0].get("role") != "assistant":
            raise ValueError("Completion must contain exactly one assistant message.")
        return f"{messages[0].get('content', '')}<|im_end|>\n"

    def convert(row: dict[str, Any]) -> dict[str, Any]:
        if row.get("prompt") and row.get("completion"):
            prompt = row["prompt"]
            completion = row["completion"]
            if isinstance(prompt, list) and isinstance(completion, list):
                return {"prompt": render_prompt(prompt), "completion": render_completion(completion)}
            return {"prompt": str(prompt), "completion": str(completion)}
        messages = row.get("messages") or []
        if len(messages) < 2:
            raise ValueError(f"Row {row.get('id') or '<unknown>'} has no usable messages.")
        assistant = messages[-1]
        if assistant.get("role") != "assistant":
            raise ValueError(f"Row {row.get('id') or '<unknown>'} must end with an assistant message.")
        return {"prompt": render_prompt(messages[:-1]), "completion": render_completion([assistant])}

    converted = {}
    for split, rows in dataset.items():
        converted[split] = rows.map(
            convert,
            remove_columns=rows.column_names,
            desc=f"Converting {split} split to prompt/completion",
        )
    return dataset.__class__(converted)


def print_and_validate_length_report(dataset: Any, tokenizer: Any, max_length: int, fail_on_truncation: bool) -> None:
    print("Token length diagnostics:")
    bad_splits: list[str] = []
    for split, rows in dataset.items():
        totals: list[int] = []
        prompts: list[int] = []
        completions: list[int] = []
        truncating = 0
        prompt_too_long = 0
        for row in rows:
            prompt_ids = tokenizer(str(row["prompt"]), add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(str(row["prompt"]) + str(row["completion"]), add_special_tokens=False)["input_ids"]
            prompt_len = len(prompt_ids)
            total_len = len(full_ids)
            completion_len = max(0, total_len - prompt_len)
            prompts.append(prompt_len)
            completions.append(completion_len)
            totals.append(total_len)
            if prompt_len >= max_length:
                prompt_too_long += 1
            if total_len > max_length:
                truncating += 1
        if truncating or prompt_too_long:
            bad_splits.append(split)
        print(
            "  {split}: rows={rows}, avg_total={avg_total:.1f}, max_total={max_total}, "
            "avg_prompt={avg_prompt:.1f}, max_prompt={max_prompt}, "
            "avg_completion={avg_completion:.1f}, max_completion={max_completion}, "
            "truncating={truncating}, prompt_too_long={prompt_too_long}".format(
                split=split,
                rows=len(rows),
                avg_total=statistics.mean(totals) if totals else 0.0,
                max_total=max(totals) if totals else 0,
                avg_prompt=statistics.mean(prompts) if prompts else 0.0,
                max_prompt=max(prompts) if prompts else 0,
                avg_completion=statistics.mean(completions) if completions else 0.0,
                max_completion=max(completions) if completions else 0,
                truncating=truncating,
                prompt_too_long=prompt_too_long,
            )
        )
    if fail_on_truncation and bad_splits:
        raise SystemExit(
            "Some prompt/completion samples exceed max_seq_length and would truncate target JSON. "
            "Increase max_seq_length or rebuild a more compact dataset."
        )


def limit_dataset_size(dataset: Any, cfg: dict[str, Any]) -> Any:
    limits = {
        "train": int(cfg.get("max_train_samples", 0) or 0),
        "validation": int(cfg.get("max_eval_samples", 0) or 0),
    }
    for split, limit in limits.items():
        if limit > 0 and split in dataset and len(dataset[split]) > limit:
            print(f"Limiting {split} split to first {limit} samples for a faster run.")
            dataset[split] = dataset[split].select(range(limit))
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter for coal blending.")
    parser.add_argument("--config", default="configs/qwen_lora.yaml")
    parser.add_argument("--model-name-or-path", help="Override model_name_or_path from the config.")
    parser.add_argument("--resume-from-checkpoint", help="Resume trainer state from an existing checkpoint directory.")
    parser.add_argument(
        "--dry-run-data",
        action="store_true",
        help="Load tokenizer and dataset, print length diagnostics, then exit before loading model weights.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.model_name_or_path:
        cfg["model_name_or_path"] = args.model_name_or_path

    device = detect_device()
    print(f"Detected device: {device}")
    if device == "cpu":
        _print_pytorch_gpu_diagnostics()
    if device == "cpu" and not cfg.get("allow_cpu_training", False) and not args.dry_run_data:
        raise SystemExit(
            "No CUDA/MPS device is available. Refusing to train on CPU by default because Qwen LoRA "
            "training will be extremely slow or may run out of memory. Use a CUDA/MPS environment, "
            "or set allow_cpu_training: true in the config for a tiny-model smoke test."
        )

    # ── Tokenizer ──────────────────────────────────────────────
    model_name_or_path = resolve_model_name_or_path(cfg["model_name_or_path"])
    tokenizer = load_tokenizer(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ────────────────────────────────────────────────
    dataset = load_dataset(
        "json",
        data_files={
            "train": resolve_path(cfg["train_file"]),
            "validation": resolve_path(cfg["eval_file"]),
        },
        cache_dir=resolve_path(cfg.get("datasets_cache_dir", "outputs/cache/datasets")),
    )
    dataset = filter_dataset_by_task(dataset, cfg.get("filter_task"))
    sft_format = str(cfg.get("sft_format", "prompt_completion"))
    if sft_format == "prompt_completion":
        dataset = prepare_prompt_completion_dataset(dataset)
        dataset = limit_dataset_size(dataset, cfg)
        print_and_validate_length_report(
            dataset,
            tokenizer,
            max_length=int(cfg.get("max_seq_length", 2048)),
            fail_on_truncation=bool(cfg.get("fail_on_truncated_completion", True)),
        )
    elif sft_format != "text":
        raise SystemExit(f"Unsupported sft_format={sft_format!r}; use 'prompt_completion' or 'text'.")
    if args.dry_run_data:
        print("Dry run finished before model loading.")
        return

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
        model_name_or_path,
        **model_kwargs,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # ── LoRA ───────────────────────────────────────────────────
    init_adapter = cfg.get("init_adapter")
    if init_adapter:
        adapter_path = resolve_path(init_adapter)
        print(f"Loading existing LoRA adapter for continued training: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
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

    # ── Training args ──────────────────────────────────────────
    eval_strategy = str(cfg.get("eval_strategy", "steps"))
    save_strategy = str(cfg.get("save_strategy", "steps"))
    training_args = SFTConfig(
        output_dir=resolve_path(cfg["output_dir"]),
        num_train_epochs=float(cfg.get("num_train_epochs", 3)),
        max_steps=int(cfg.get("max_steps", -1)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(cfg.get("logging_steps", 5)),
        eval_steps=int(cfg.get("eval_steps", 20)),
        save_steps=int(cfg.get("save_steps", 20)),
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        do_eval=eval_strategy != "no",
        fp16=(device != "cpu" and not cfg.get("bf16") and not use_4bit),
        bf16=bool(cfg.get("bf16", False)),
        report_to="none",
        remove_unused_columns=False,
        max_length=int(cfg.get("max_seq_length", 2048)),
        packing=False,
        dataset_text_field=str(cfg.get("dataset_text_field", "text")),
        completion_only_loss=cfg.get(
            "completion_only_loss",
            True if sft_format == "prompt_completion" else None,
        ),
        # Qwen chat templates differ across model releases. Keep this configurable because
        # assistant_only_loss=True requires a chat template with assistant token masks.
        assistant_only_loss=bool(cfg.get("assistant_only_loss", False)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if eval_strategy != "no" else None,
        processing_class=tokenizer,
    )

    resume_from_checkpoint = args.resume_from_checkpoint or cfg.get("resume_from_checkpoint")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(resolve_path(cfg["output_dir"]))
    tokenizer.save_pretrained(resolve_path(cfg["output_dir"]))


if __name__ == "__main__":
    main()
