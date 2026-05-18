#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
            _score_candidate_business(row, parsed, bucket)
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
        "business_plan_count": 0,
        "business_sample_count": 0,
        "business_best_score_sum": 0.0,
        "business_metric_sums": {
            "quality_compliance_score": 0.0,
            "quality_margin_score": 0.0,
            "cost_advantage_score": 0.0,
            "inventory_feasibility_score": 0.0,
            "blend_balance_score": 0.0,
            "risk_control_score": 0.0,
            "business_effect_score": 0.0,
        },
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


def _score_candidate_business(row: dict, parsed: dict, bucket: dict) -> None:
    context = parse_candidate_context(row.get("input", ""))
    if not context["materials"]:
        return
    plans = parsed.get("plans")
    if not isinstance(plans, list) or not plans:
        return

    sample_scores = []
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        metrics = score_business_plan(plan, context)
        if not metrics:
            continue
        sample_scores.append(metrics["business_effect_score"])
        bucket["business_plan_count"] += 1
        for key, value in metrics.items():
            bucket["business_metric_sums"][key] += value

    if sample_scores:
        bucket["business_sample_count"] += 1
        bucket["business_best_score_sum"] += max(sample_scores)


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
        business_total = max(1, bucket["business_plan_count"])
        radar_metrics = {
            key: round(value / business_total, 4)
            for key, value in bucket["business_metric_sums"].items()
        }
        out.update(
            {
                "business_plan_count": bucket["business_plan_count"],
                "average_business_effect_score": radar_metrics["business_effect_score"],
                "best_plan_business_effect_score": round(
                    bucket["business_best_score_sum"] / max(1, bucket["business_sample_count"]),
                    4,
                ),
                "radar_metrics": {
                    "质量达标": radar_metrics["quality_compliance_score"],
                    "质量余量": radar_metrics["quality_margin_score"],
                    "成本优势": radar_metrics["cost_advantage_score"],
                    "库存可执行": radar_metrics["inventory_feasibility_score"],
                    "配比均衡": radar_metrics["blend_balance_score"],
                    "风险控制": radar_metrics["risk_control_score"],
                },
            }
        )
    return out


def parse_candidate_context(text: str) -> dict:
    return {
        "order": {
            "demand_quantity": _extract_first_number(text, [r"需求量[：:]\s*([0-9.]+)\s*吨"]),
            "target_ash": _extract_first_number(text, [r"灰分(?:要求|上限)[：:]\s*(?:≤)?\s*([0-9.]+)"]),
            "target_sulfur": _extract_first_number(text, [r"硫分(?:要求|上限)[：:]\s*(?:≤)?\s*([0-9.]+)"]),
            "target_moisture": _extract_first_number(text, [r"水分(?:要求|上限)[：:]\s*(?:≤)?\s*([0-9.]+)"]),
            "target_calorific": _extract_first_number(text, [r"发热量(?:要求|下限)[：:]\s*(?:≥)?\s*([0-9.]+)"]),
        },
        "materials": _parse_materials(text),
    }


def _parse_materials(text: str) -> dict[str, dict]:
    materials: dict[str, dict] = {}
    for line in text.splitlines():
        if "coalId=" not in line or "productBatchNo=" not in line:
            continue
        material = {
            "coalId": _extract_value(line, "coalId"),
            "productBatchNo": _extract_value(line, "productBatchNo"),
            "availableQuantity": _extract_number_after(line, "可用量"),
            "unitCost": _extract_number_after(line, "单价"),
            "ashContent": _extract_number_after(line, "灰分"),
            "sulfurContent": _extract_number_after(line, "硫分"),
            "moistureContent": _extract_number_after(line, "水分"),
            "volatileContent": _extract_number_after(line, "挥发分"),
            "calorificValue": _extract_number_after(line, "发热量"),
        }
        batch = str(material.get("productBatchNo") or "").strip()
        coal_id = str(material.get("coalId") or "").strip()
        if batch:
            materials[batch] = material
        if coal_id:
            materials.setdefault(f"coal:{coal_id}", material)
    return materials


def score_business_plan(plan: dict, context: dict) -> dict[str, float] | None:
    items = plan.get("items")
    if not isinstance(items, list) or not items:
        return None
    selected = []
    ratio_sum = 0.0
    for item in items:
        if not isinstance(item, dict):
            continue
        ratio = _safe_float(item.get("ratio"))
        ratio_sum += ratio
        material = _lookup_material(item, context["materials"])
        if material and ratio > 0:
            selected.append((material, ratio))
    if not selected:
        return None

    normalized = [(m, r / ratio_sum) for m, r in selected] if ratio_sum > 0 else selected
    predicted = _weighted_metrics(normalized)
    order = context["order"]
    quality_compliance = _quality_compliance_score(predicted, order)
    quality_margin = _quality_margin_score(predicted, order)
    cost_advantage = _cost_advantage_score(predicted["unitCost"], context["materials"].values())
    inventory_feasibility = _inventory_feasibility_score(selected, order.get("demand_quantity"))
    blend_balance = _blend_balance_score(items, ratio_sum)
    risk_control = _risk_control_score(plan, quality_compliance, quality_margin, inventory_feasibility)
    business_effect = (
        quality_compliance * 0.35
        + quality_margin * 0.20
        + cost_advantage * 0.15
        + inventory_feasibility * 0.15
        + blend_balance * 0.10
        + risk_control * 0.05
    )
    return {
        "quality_compliance_score": round(quality_compliance, 4),
        "quality_margin_score": round(quality_margin, 4),
        "cost_advantage_score": round(cost_advantage, 4),
        "inventory_feasibility_score": round(inventory_feasibility, 4),
        "blend_balance_score": round(blend_balance, 4),
        "risk_control_score": round(risk_control, 4),
        "business_effect_score": round(business_effect, 4),
    }


def _lookup_material(item: dict, materials: dict[str, dict]) -> dict | None:
    batch = str(item.get("productBatchNo") or "").strip()
    coal_id = str(item.get("coalId") or "").strip()
    return materials.get(batch) or materials.get(f"coal:{coal_id}")


def _weighted_metrics(selected: list[tuple[dict, float]]) -> dict[str, float | None]:
    fields = ["ashContent", "sulfurContent", "moistureContent", "volatileContent", "calorificValue", "unitCost"]
    out: dict[str, float | None] = {}
    for field in fields:
        values = [(m.get(field), ratio) for m, ratio in selected if m.get(field) is not None]
        out[field] = sum(_safe_float(value) * ratio for value, ratio in values) if values else None
    return out


def _quality_compliance_score(predicted: dict, order: dict) -> float:
    checks = [
        _upper_bound_pass(predicted.get("ashContent"), order.get("target_ash")),
        _upper_bound_pass(predicted.get("sulfurContent"), order.get("target_sulfur")),
        _upper_bound_pass(predicted.get("moistureContent"), order.get("target_moisture")),
        _lower_bound_pass(predicted.get("calorificValue"), order.get("target_calorific")),
    ]
    known = [x for x in checks if x is not None]
    return sum(known) / len(known) if known else 0.0


def _quality_margin_score(predicted: dict, order: dict) -> float:
    scores = [
        _upper_margin_score(predicted.get("ashContent"), order.get("target_ash")),
        _upper_margin_score(predicted.get("sulfurContent"), order.get("target_sulfur")),
        _upper_margin_score(predicted.get("moistureContent"), order.get("target_moisture")),
        _lower_margin_score(predicted.get("calorificValue"), order.get("target_calorific")),
    ]
    known = [x for x in scores if x is not None]
    return sum(known) / len(known) if known else 0.0


def _cost_advantage_score(plan_cost: float | None, materials: list[dict] | object) -> float:
    if plan_cost is None:
        return 0.0
    costs = [_safe_float(m.get("unitCost")) for m in materials if m.get("unitCost") is not None]
    costs = [c for c in costs if c > 0]
    if not costs:
        return 0.0
    lo, hi = min(costs), max(costs)
    if math.isclose(lo, hi):
        return 1.0
    return _clip01((hi - plan_cost) / (hi - lo))


def _inventory_feasibility_score(selected: list[tuple[dict, float]], demand_quantity: float | None) -> float:
    if not demand_quantity or demand_quantity <= 0:
        return 0.0
    scores = []
    for material, ratio in selected:
        need = demand_quantity * ratio
        available = _safe_float(material.get("availableQuantity"))
        if need <= 0:
            continue
        scores.append(_clip01(available / need))
    return sum(scores) / len(scores) if scores else 0.0


def _blend_balance_score(items: list[dict], ratio_sum: float) -> float:
    ratios = [_safe_float(item.get("ratio")) for item in items if _safe_float(item.get("ratio")) > 0]
    if not ratios:
        return 0.0
    ratio_close = _clip01(1 - abs(ratio_sum - 1.0) / 0.05)
    count_score = 1.0 if 2 <= len(ratios) <= 4 else 0.4
    if len(ratios) == 1:
        entropy_score = 0.0
    else:
        normalized = [r / sum(ratios) for r in ratios]
        entropy = -sum(r * math.log(r) for r in normalized)
        entropy_score = entropy / math.log(len(normalized))
    concentration_score = 1.0 if max(ratios) <= 0.75 else 0.5
    return ratio_close * (count_score * 0.35 + entropy_score * 0.45 + concentration_score * 0.20)


def _risk_control_score(
    plan: dict,
    quality_compliance: float,
    quality_margin: float,
    inventory_feasibility: float,
) -> float:
    text = f"{plan.get('risk', '')} {plan.get('strategy', '')}".strip()
    risky = quality_compliance < 1.0 or quality_margin < 0.35 or inventory_feasibility < 1.0
    risk_words = ["风险", "接近", "超", "不足", "复核", "波动", "硫分", "水分", "库存", "质量"]
    mentions_risk = any(word in text for word in risk_words)
    if risky:
        return 1.0 if mentions_risk else 0.3
    return 0.9 if mentions_risk else 0.75


def _upper_bound_pass(value: float | None, target: float | None) -> float | None:
    if value is None or not target:
        return None
    return 1.0 if value <= target else 0.0


def _lower_bound_pass(value: float | None, target: float | None) -> float | None:
    if value is None or not target:
        return None
    return 1.0 if value >= target else 0.0


def _upper_margin_score(value: float | None, target: float | None) -> float | None:
    if value is None or not target:
        return None
    if value > target:
        return 0.0
    return _clip01((target - value) / target / 0.20)


def _lower_margin_score(value: float | None, target: float | None) -> float | None:
    if value is None or not target:
        return None
    if value < target:
        return 0.0
    return _clip01((value - target) / target / 0.20)


def _extract_value(text: str, key: str) -> str:
    match = re.search(rf"{re.escape(key)}=([^，,]+)", text)
    value = match.group(1).strip() if match else ""
    return "" if value == "—" else value


def _extract_number_after(text: str, label: str) -> float | None:
    return _extract_first_number(text, [rf"{label}=([0-9.]+)", rf"{label}[：:]?\s*([0-9.]+)"])


def _extract_first_number(text: str, patterns: list[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _safe_float(match.group(1))
    return None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


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

    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    outputs: list[str] = []
    for row in rows:
        messages = row["messages"][:-1]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    parser.add_argument("--max-new-tokens", type=int, default=2048)
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
