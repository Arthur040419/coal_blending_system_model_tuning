#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coal_tuning.dataset import CANDIDATE_OUTPUT_FIELDS, EXPLANATION_OUTPUT_FIELDS


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("{"):
        candidate = text
    else:
        match = re.search(r"\{.*\}", text, flags=re.S)
        candidate = match.group(0) if match else ""
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def score_outputs(rows: list[dict], outputs: list[str]) -> dict:
    totals: dict[str, dict] = {}
    for row, out in zip(rows, outputs):
        task = row.get("task") or "unknown"
        bucket = totals.setdefault(task, _empty_bucket(task))
        bucket["total"] += 1
        parsed = extract_json(out)
        if parsed is None:
            continue
        bucket["valid_json"] += 1
        if task == "candidate_generation":
            _score_candidate(parsed, bucket)
        else:
            _score_explanation(parsed, bucket)
    by_task = {task: _finalize_bucket(task, bucket) for task, bucket in totals.items()}
    overall_total = max(1, len(rows))
    return {
        "total": len(rows),
        "valid_json_rate": round(sum(b["valid_json"] for b in totals.values()) / overall_total, 4),
        "by_task": by_task,
    }


def _empty_bucket(task: str) -> dict:
    fields = CANDIDATE_OUTPUT_FIELDS if task == "candidate_generation" else EXPLANATION_OUTPUT_FIELDS
    return {
        "total": 0,
        "valid_json": 0,
        "complete": 0,
        "nonempty": 0,
        "field_hits": {field: 0 for field in fields},
        "candidate_plan_count": 0,
        "candidate_valid_plan_count": 0,
        "candidate_valid_ratio_count": 0,
        "candidate_valid_item_count": 0,
    }


def _score_explanation(parsed: dict, bucket: dict) -> None:
    fields = EXPLANATION_OUTPUT_FIELDS
    if all(field in parsed for field in fields):
        bucket["complete"] += 1
    if all(str(parsed.get(field, "")).strip() for field in fields):
        bucket["nonempty"] += 1
    for field in fields:
        if str(parsed.get(field, "")).strip():
            bucket["field_hits"][field] += 1


def _score_candidate(parsed: dict, bucket: dict) -> None:
    if "plans" in parsed:
        bucket["complete"] += 1
        bucket["field_hits"]["plans"] += 1
    plans = parsed.get("plans")
    if isinstance(plans, list) and plans:
        bucket["nonempty"] += 1
        bucket["candidate_plan_count"] += len(plans)
        for plan in plans:
            if not isinstance(plan, dict):
                continue
            items = plan.get("items")
            has_required = all(str(plan.get(k, "")).strip() for k in ["planName", "strategy", "risk"])
            if isinstance(items, list) and 2 <= len(items) <= 4:
                bucket["candidate_valid_item_count"] += 1
                ratio_sum = 0.0
                material_keys = set()
                item_ok = True
                for item in items:
                    if not isinstance(item, dict):
                        item_ok = False
                        continue
                    ratio_sum += _safe_float(item.get("ratio"))
                    key = str(item.get("productBatchNo") or item.get("coalId") or "")
                    if not key or key in material_keys:
                        item_ok = False
                    material_keys.add(key)
                if abs(ratio_sum - 1.0) <= 0.02:
                    bucket["candidate_valid_ratio_count"] += 1
                if has_required and item_ok and abs(ratio_sum - 1.0) <= 0.02:
                    bucket["candidate_valid_plan_count"] += 1


def _finalize_bucket(task: str, bucket: dict) -> dict:
    total = max(1, bucket["total"])
    out = {
        "total": bucket["total"],
        "valid_json_rate": round(bucket["valid_json"] / total, 4),
        "complete_field_rate": round(bucket["complete"] / total, 4),
        "nonempty_field_rate": round(bucket["nonempty"] / total, 4),
        "field_nonempty_rate": {
            k: round(v / total, 4) for k, v in bucket["field_hits"].items()
        },
    }
    if task == "candidate_generation":
        plan_total = max(1, bucket["candidate_plan_count"])
        out.update(
            {
                "candidate_plan_count": bucket["candidate_plan_count"],
                "valid_candidate_plan_rate": round(bucket["candidate_valid_plan_count"] / plan_total, 4),
                "valid_ratio_plan_rate": round(bucket["candidate_valid_ratio_count"] / plan_total, 4),
                "valid_item_count_plan_rate": round(bucket["candidate_valid_item_count"] / plan_total, 4),
            }
        )
    return out


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def generate_outputs(
    rows: list[dict],
    base_model: str,
    adapter: str | None,
    max_new_tokens: int,
) -> list[str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter or base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    outputs: list[str] = []
    for row in rows:
        messages = row["messages"][:-1]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = row["text"].split("<|assistant|>", 1)[0] + "<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JSON output quality on eval samples.")
    parser.add_argument("--eval-file", default="data/processed/eval.jsonl")
    parser.add_argument("--base-model", help="Base model path/name for generation.")
    parser.add_argument("--adapter", help="LoRA adapter path. If omitted, evaluates base model.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--predictions-file", default="outputs/reports/predictions.jsonl")
    parser.add_argument("--report-file", default="outputs/reports/eval_report.json")
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    if not eval_path.is_absolute():
        eval_path = ROOT / eval_path
    rows = read_jsonl(eval_path)[: args.limit]

    if args.base_model:
        outputs = generate_outputs(rows, args.base_model, args.adapter, args.max_new_tokens)
    else:
        outputs = [json.dumps(r["output"], ensure_ascii=False) for r in rows]

    report = score_outputs(rows, outputs)
    pred_path = ROOT / args.predictions_file
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as f:
        for row, output in zip(rows, outputs):
            f.write(json.dumps({"id": row["id"], "prediction": output}, ensure_ascii=False) + "\n")

    report_path = ROOT / args.report_file
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"predictions: {pred_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
