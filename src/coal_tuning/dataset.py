from __future__ import annotations

import json
import random
import csv
from pathlib import Path
from typing import Any

from .prompts import (
    CANDIDATE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_candidate_user_prompt,
    build_user_prompt,
    to_chatml,
)

EXPLANATION_OUTPUT_FIELDS = ["ruleBasis", "caseReference", "recommendReason", "riskTip", "finalExplanation"]
CANDIDATE_OUTPUT_FIELDS = ["plans"]
OUTPUT_FIELDS = EXPLANATION_OUTPUT_FIELDS


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def safe_text(value: object | None, default: str = "当前知识库依据不足") -> str:
    text = "" if value is None else str(value).strip()
    return text if text else default


def build_training_records(
    tables: dict[str, list[dict[str, object | None]]],
    public_samples: list[dict[str, object | None]] | None = None,
    include_explanation: bool = True,
    include_candidate: bool = True,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if include_explanation:
        records.extend(build_explanation_records(tables))
    if include_candidate:
        records.extend(build_candidate_generation_records(tables))
        if public_samples:
            records.extend(build_public_candidate_generation_records(public_samples))
    return records


def build_explanation_records(tables: dict[str, list[dict[str, object | None]]]) -> list[dict[str, Any]]:
    orders = {str(r["id"]): r for r in tables.get("orders", [])}
    details_by_plan: dict[str, list[dict[str, object | None]]] = {}
    for d in tables.get("blend_plan_detail", []):
        details_by_plan.setdefault(str(d.get("plan_id")), []).append(d)

    rules = tables.get("rule_knowledge", [])
    cases = tables.get("case_sample", [])
    rag = tables.get("rag_knowledge", [])
    inventory = tables.get("inventory", [])
    coal_types = {str(r["id"]): r for r in tables.get("coal_type", [])}

    records: list[dict[str, Any]] = []
    for plan in tables.get("blend_plan", []):
        if not _is_usable_plan(plan):
            continue
        order = orders.get(str(plan.get("order_id")))
        if not order:
            continue
        details = details_by_plan.get(str(plan.get("id")), [])
        context = build_context(order, plan, details, rules, cases, rag, inventory, coal_types)
        output = {
            "ruleBasis": safe_text(plan.get("rule_basis"), fallback_rule_basis(plan, rules)),
            "caseReference": safe_text(plan.get("case_reference"), fallback_case_reference(cases)),
            "recommendReason": safe_text(plan.get("recommend_reason"), fallback_recommend_reason(plan)),
            "riskTip": safe_text(plan.get("risk_tip"), "当前方案未发现明确风险，执行前仍需复核库存与煤质实测数据。"),
            "finalExplanation": build_final_explanation(plan, details, coal_types),
        }
        user_prompt = build_user_prompt(context)
        assistant_output = json_dumps(output)
        records.append(
            {
                "id": f"plan-{plan.get('id')}",
                "task": "plan_explanation",
                "system": SYSTEM_PROMPT,
                "input": user_prompt,
                "output": output,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_output},
                ],
                "text": to_chatml(SYSTEM_PROMPT, user_prompt, assistant_output),
                "meta": {
                    "planId": plan.get("id"),
                    "planCode": plan.get("plan_code"),
                    "orderId": plan.get("order_id"),
                    "modelName": plan.get("ai_model_name"),
                    "aiGenerated": plan.get("ai_generate_flag"),
                },
            }
        )
    return records


def build_candidate_generation_records(tables: dict[str, list[dict[str, object | None]]]) -> list[dict[str, Any]]:
    orders = {str(r["id"]): r for r in tables.get("orders", [])}
    details_by_plan: dict[str, list[dict[str, object | None]]] = {}
    for d in tables.get("blend_plan_detail", []):
        details_by_plan.setdefault(str(d.get("plan_id")), []).append(d)

    plans_by_order: dict[str, list[dict[str, object | None]]] = {}
    for plan in tables.get("blend_plan", []):
        if not _is_usable_candidate_plan(plan, details_by_plan.get(str(plan.get("id")), [])):
            continue
        plans_by_order.setdefault(str(plan.get("order_id")), []).append(plan)

    rules = tables.get("rule_knowledge", [])
    cases = tables.get("case_sample", [])
    rag = tables.get("rag_knowledge", [])
    inventory = tables.get("inventory", [])
    product_batches = tables.get("product_batch", [])
    coal_types = {str(r["id"]): r for r in tables.get("coal_type", [])}

    records: list[dict[str, Any]] = []
    for order_id, order_plans in plans_by_order.items():
        order = orders.get(order_id)
        if not order:
            continue
        ranked = sorted(
            order_plans,
            key=lambda p: (
                str(p.get("feasible_flag") or "0") == "1",
                _num(p.get("overall_score")),
                -_num(p.get("total_cost")),
            ),
            reverse=True,
        )[:5]
        if len(ranked) < 1:
            continue
        detail_groups = {str(p.get("id")): details_by_plan.get(str(p.get("id")), []) for p in ranked}
        candidate_scope = "product_batch" if any(
            d.get("product_batch_no") for rows in detail_groups.values() for d in rows
        ) else "coal_type"
        materials = build_candidate_materials(
            candidate_scope, product_batches, inventory, coal_types, detail_groups
        )
        if len(materials) < 2:
            continue
        context = build_candidate_context(
            order, candidate_scope, materials, rules, cases, rag
        )
        output = build_candidate_output(ranked, detail_groups, coal_types)
        if not output["plans"]:
            continue
        user_prompt = build_candidate_user_prompt(context)
        assistant_output = json_dumps(output)
        records.append(
            {
                "id": f"candidate-order-{order_id}",
                "task": "candidate_generation",
                "system": CANDIDATE_SYSTEM_PROMPT,
                "input": user_prompt,
                "output": output,
                "messages": [
                    {"role": "system", "content": CANDIDATE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_output},
                ],
                "text": to_chatml(CANDIDATE_SYSTEM_PROMPT, user_prompt, assistant_output),
                "meta": {
                    "orderId": order_id,
                    "candidateScope": candidate_scope,
                    "source": "system_sql_ranked_plans",
                },
            }
        )
    return records


def read_public_coal_quality_csv(path: str | Path | None) -> list[dict[str, object | None]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def build_public_candidate_generation_records(samples: list[dict[str, object | None]]) -> list[dict[str, Any]]:
    cleaned = [s for s in samples if _has_public_quality(s)]
    records: list[dict[str, Any]] = []
    scenarios = _build_all_public_scenarios()
    for scenario in scenarios:
        materials = public_samples_to_materials(cleaned)
        plans = build_public_candidate_plans(scenario, materials)
        if not plans:
            continue
        context = build_public_candidate_context(scenario, materials)
        output = {"plans": plans}
        user_prompt = build_candidate_user_prompt(context)
        assistant_output = json_dumps(output)
        records.append(
            {
                "id": f"candidate-{scenario['id']}",
                "task": "candidate_generation",
                "system": CANDIDATE_SYSTEM_PROMPT,
                "input": user_prompt,
                "output": output,
                "messages": [
                    {"role": "system", "content": CANDIDATE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_output},
                ],
                "text": to_chatml(CANDIDATE_SYSTEM_PROMPT, user_prompt, assistant_output),
                "meta": {"source": "public_coal_quality_samples", "scenario": scenario["id"]},
            }
        )
    return records


def _build_all_public_scenarios() -> list[dict[str, object | None]]:
    """Return realistic coal blending scenarios based on actual Chinese power/steel industry cases.

    Each scenario maps to a real-world blending pattern documented in industry papers
    (see CSV source_url fields for references).
    """
    return [
        # ── Case 1: 低硫环保电煤 — 神木+大同经典搭配 ──────────
        {
            "id": "public-low-sulfur-power",
            "orderCode": "PUBLIC-LOW-S-001",
            "customerName": "沿海电厂低硫动力煤采购",
            "demandQuantity": 5000.0,
            "targetAsh": 15.0,
            "targetSulfur": 0.60,
            "targetMoisture": 13.0,
            "targetCalorific": 5600.0,
            "priorityLevel": 2,
            "strategy": "以低硫低灰的神木煤为主力（60-70%），搭配大同煤调节热值和成本；重点控制硫分≤0.6%。",
        },
        # ── Case 2: 高热值补偿 — 阳泉无烟煤提热 ─────────────────
        {
            "id": "public-high-heat",
            "orderCode": "PUBLIC-HIGH-CV-001",
            "customerName": "高炉喷吹用煤配煤",
            "demandQuantity": 3200.0,
            "targetAsh": 12.0,
            "targetSulfur": 0.80,
            "targetMoisture": 10.0,
            "targetCalorific": 6500.0,
            "priorityLevel": 3,
            "strategy": "以高热值阳泉无烟煤为主，搭配低硫东胜煤；注意挥发分差值限制（无烟煤V<10%，配煤后V≥20%）。",
        },
        # ── Case 3: 准东高钠煤掺烧 — 酒钢宏晟案例 ──────────────
        {
            "id": "public-zhundong-blend",
            "orderCode": "PUBLIC-ZD-001",
            "customerName": "新疆准东煤掺烧配煤",
            "demandQuantity": 6000.0,
            "targetAsh": 12.0,
            "targetSulfur": 0.60,
            "targetMoisture": 18.0,
            "targetCalorific": 4800.0,
            "priorityLevel": 2,
            "strategy": "准东煤（高钠易结渣）掺配比例控制在50%以下；用大同煤或东胜煤稀释钠含量降低结渣风险。",
        },
        # ── Case 4: 多煤种均衡 — 张家口热电四煤种 0:2:5:3 ────
        {
            "id": "public-balanced-multi",
            "orderCode": "PUBLIC-BAL-001",
            "customerName": "多煤种均衡掺配动力煤",
            "demandQuantity": 5000.0,
            "targetAsh": 16.0,
            "targetSulfur": 0.90,
            "targetMoisture": 14.0,
            "targetCalorific": 5300.0,
            "priorityLevel": 2,
            "strategy": "参考张家口热电四煤种方案；以中档煤为主力（50%），高/低档煤各搭配25%；平衡成本与质量。",
        },
        # ── Case 5: 低成本经济型 — 曲靖电厂高比例低质煤 ─────
        {
            "id": "public-cost-priority",
            "orderCode": "PUBLIC-COST-001",
            "customerName": "低负荷经济掺烧配煤",
            "demandQuantity": 7000.0,
            "targetAsh": 22.0,
            "targetSulfur": 1.50,
            "targetMoisture": 15.0,
            "targetCalorific": 4500.0,
            "priorityLevel": 1,
            "strategy": "约束宽松优先降成本；高灰高硫六盘水煤可占30-40%；用东胜煤补热值同时拉低硫分；准东煤作低成本填料。",
        },
        # ── Case 6: 高挥发分配煤挥发分差值限制 ──────────────
        {
            "id": "public-volatile-diff",
            "orderCode": "PUBLIC-VOL-001",
            "customerName": "高低挥发分煤种搭配配煤",
            "demandQuantity": 4000.0,
            "targetAsh": 14.0,
            "targetSulfur": 0.80,
            "targetMoisture": 12.0,
            "targetCalorific": 5600.0,
            "priorityLevel": 3,
            "strategy": "阳泉无烟煤（V≈9%）与高挥发分煤搭配；参照GB/T 25960-2010，高低挥发分差值≥15%时低挥发分煤配入量≤20%。",
        },
        # ── Case 7: 炼焦煤配煤 — 离柳焦煤高贵煤种 ─────────
        {
            "id": "public-coking-blend",
            "orderCode": "PUBLIC-COKE-001",
            "customerName": "炼焦配煤优质主焦煤方案",
            "demandQuantity": 2800.0,
            "targetAsh": 9.5,
            "targetSulfur": 1.20,
            "targetMoisture": 8.0,
            "targetCalorific": 7200.0,
            "priorityLevel": 5,
            "strategy": "以离柳焦煤（低灰强粘结，G=75-88）为主力煤种；搭配东胜煤或大同煤调节硫分至≤1.2%；严格控制灰分。",
        },
        # ── Case 8: 超低硫环保合规 — 参照GB/T 25960硫分≤2% ──
        {
            "id": "public-ultra-low-sulfur",
            "orderCode": "PUBLIC-ULS-001",
            "customerName": "长三角环保严控区低硫电煤",
            "demandQuantity": 4500.0,
            "targetAsh": 14.0,
            "targetSulfur": 0.35,
            "targetMoisture": 13.0,
            "targetCalorific": 5700.0,
            "priorityLevel": 4,
            "strategy": "硫分约束极严（≤0.35%）；全部物料硫分需≤0.4%；东胜煤和神木煤为主力（硫分均≤0.4%）；大同煤因硫分1.2%禁用。",
        },
        # ── Case 9: 大用量保供 — 兼顾多仓库存 ──────────────
        {
            "id": "public-high-quantity",
            "orderCode": "PUBLIC-QTY-001",
            "customerName": "冬季供暖保供大用量配煤",
            "demandQuantity": 9000.0,
            "targetAsh": 18.0,
            "targetSulfur": 1.00,
            "targetMoisture": 15.0,
            "targetCalorific": 5000.0,
            "priorityLevel": 2,
            "strategy": "冬季保供大需求；3-4种煤分摊库存压力；大同煤/平朔煤做主力（库存量大），神木煤/东胜煤补质量。",
        },
    ]


def split_and_write(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    eval_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.Random(seed).shuffle(records)
    eval_size = max(1, int(len(records) * eval_ratio)) if len(records) > 1 else 0
    eval_records = records[:eval_size]
    train_records = records[eval_size:]
    train_path = out / "train.jsonl"
    eval_path = out / "eval.jsonl"
    write_jsonl(train_path, train_records)
    write_jsonl(eval_path, eval_records)
    return train_path, eval_path


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _is_usable_plan(plan: dict[str, object | None]) -> bool:
    text_parts = [
        plan.get("explanation"),
        plan.get("rule_basis"),
        plan.get("recommend_reason"),
        plan.get("final_explanation"),
    ]
    text = "\n".join(str(x) for x in text_parts if x)
    if len(text.strip()) < 30:
        return False
    bad_fragments = ["构建输出内容", "检查格式", "是否可执行", "2. 规则依据：3."]
    return not any(bad in text for bad in bad_fragments)


def build_context(
    order: dict[str, object | None],
    plan: dict[str, object | None],
    details: list[dict[str, object | None]],
    rules: list[dict[str, object | None]],
    cases: list[dict[str, object | None]],
    rag: list[dict[str, object | None]],
    inventory: list[dict[str, object | None]],
    coal_types: dict[str, dict[str, object | None]],
) -> str:
    lines: list[str] = []
    lines.append("【订单信息】")
    lines.append(
        "订单编号：{order_code}\n客户名称：{customer_name}\n需求量：{demand_quantity} 吨\n"
        "灰分要求：≤{target_ash}%\n硫分要求：≤{target_sulfur}%\n水分要求：≤{target_moisture}%\n"
        "发热量要求：≥{target_calorific} kcal/kg\n优先级：{priority_level}\n交付日期：{delivery_date}".format(
            **_fmt_map(order)
        )
    )
    lines.append("\n【推荐方案】")
    lines.append(
        "方案名称：{plan_name}\n总成本：{total_cost} 元\n质量评分：{quality_score}\n"
        "成本评分：{cost_score}\n稳定性评分：{stability_score}\n综合评分：{overall_score}\n"
        "可行性：{feasible_flag}\n风险等级：{risk_level}\n约束摘要：{constraint_summary}\n评分明细：{score_detail}".format(
            **_fmt_map(plan)
        )
    )
    if details:
        lines.append("\n【方案明细】")
        for d in details[:5]:
            coal = coal_types.get(str(d.get("coal_id")), {})
            lines.append(
                "- {coal_name}：配比 {blend_ratio}，用量 {use_quantity} 吨，预测灰分 {predicted_ash}%、"
                "硫分 {predicted_sulfur}%、水分 {predicted_moisture}%、热值 {predicted_calorific}，"
                "单价 {unit_cost} 元/吨，批次 {product_batch_no}".format(
                    coal_name=coal.get("coal_name") or f"煤种{d.get('coal_id')}",
                    **_fmt_map(d),
                )
            )
    lines.append("\n【命中规则候选】")
    for r in rules[:8]:
        lines.append(
            "- {rule_code} {rule_name}（{rule_type}）：{rule_content}".format(**_fmt_map(r))
        )
    lines.append("\n【参考案例候选】")
    for c in cases[:5]:
        lines.append(
            "- {case_name}：{order_desc}；{blend_desc}；效果：{effectiveness_eval}".format(
                **_fmt_map(c)
            )
        )
    lines.append("\n【RAG知识库片段】")
    for k in rag[:8]:
        lines.append("- {title}（{knowledge_type}）：{content}".format(**_fmt_map(k)))
    lines.append("\n【库存信息】")
    for inv in inventory[:8]:
        coal = coal_types.get(str(inv.get("coal_id")), {})
        lines.append(
            "- {coal_name}：仓库 {warehouse_code}，可用 {available_quantity} 吨，阶段 {material_stage}，批次 {product_batch_no}{raw_batch_no}".format(
                coal_name=coal.get("coal_name") or f"煤种{inv.get('coal_id')}",
                **_fmt_map(inv),
            )
        )
    return "\n".join(lines)


def build_candidate_materials(
    candidate_scope: str,
    product_batches: list[dict[str, object | None]],
    inventory: list[dict[str, object | None]],
    coal_types: dict[str, dict[str, object | None]],
    detail_groups: dict[str, list[dict[str, object | None]]],
) -> list[dict[str, object | None]]:
    used_batch_nos = {
        str(d.get("product_batch_no"))
        for rows in detail_groups.values()
        for d in rows
        if d.get("product_batch_no")
    }
    used_coal_ids = {
        str(d.get("coal_id"))
        for rows in detail_groups.values()
        for d in rows
        if d.get("coal_id")
    }
    materials: list[dict[str, object | None]] = []
    if candidate_scope == "product_batch":
        ranked_batches = sorted(
            product_batches,
            key=lambda p: (str(p.get("product_batch_no")) in used_batch_nos, _num(p.get("available_quantity"))),
            reverse=True,
        )
        for p in ranked_batches:
            if len(materials) >= 10 and str(p.get("product_batch_no")) not in used_batch_nos:
                continue
            coal = coal_types.get(str(p.get("coal_id")), {})
            materials.append(
                {
                    "coalId": p.get("coal_id"),
                    "productBatchNo": p.get("product_batch_no"),
                    "name": p.get("product_name") or coal.get("coal_name"),
                    "availableQuantity": p.get("available_quantity") or p.get("quantity"),
                    "unitCost": coal.get("purchase_price"),
                    "ashContent": p.get("ash_content"),
                    "sulfurContent": p.get("sulfur_content"),
                    "moistureContent": p.get("moisture_content"),
                    "volatileContent": p.get("volatile_content"),
                    "calorificValue": p.get("calorific_value"),
                    "source": "product_batch",
                }
            )
    else:
        ranked_inventory = sorted(
            inventory,
            key=lambda inv: (str(inv.get("coal_id")) in used_coal_ids, _num(inv.get("available_quantity"))),
            reverse=True,
        )
        for inv in ranked_inventory:
            if len(materials) >= 10 and str(inv.get("coal_id")) not in used_coal_ids:
                continue
            coal = coal_types.get(str(inv.get("coal_id")), {})
            materials.append(
                {
                    "coalId": inv.get("coal_id"),
                    "productBatchNo": inv.get("product_batch_no"),
                    "name": coal.get("coal_name"),
                    "availableQuantity": inv.get("available_quantity"),
                    "unitCost": coal.get("purchase_price"),
                    "ashContent": None,
                    "sulfurContent": None,
                    "moistureContent": None,
                    "volatileContent": None,
                    "calorificValue": None,
                    "source": "coal_type_inventory",
                }
            )
    return _dedupe_materials(materials)


def build_candidate_context(
    order: dict[str, object | None],
    candidate_scope: str,
    materials: list[dict[str, object | None]],
    rules: list[dict[str, object | None]],
    cases: list[dict[str, object | None]],
    rag: list[dict[str, object | None]],
) -> str:
    lines = [
        "你是煤矿智能配煤系统中的候选方案生成助手。请基于订单约束、候选物料、规则和案例，生成候选配比建议。",
        "目标：优先生成质量达标、质量余量充足、成本可控且库存可执行的配煤方案。",
        f"\n【候选范围】{candidate_scope}",
        "\n【订单信息】",
        (
            "订单编号：{order_code}\n客户名称：{customer_name}\n需求量：{demand_quantity} 吨\n"
            "灰分上限：{target_ash}%\n硫分上限：{target_sulfur}%\n水分上限：{target_moisture}%\n"
            "挥发分参考：{target_volatile}%\n发热量下限：{target_calorific} kcal/kg\n优先级：{priority_level}"
        ).format(**_fmt_map(order)),
        "\n【候选物料】",
    ]
    for m in materials[:12]:
        lines.append(
            "- coalId={coalId}，productBatchNo={productBatchNo}，名称={name}，可用量={availableQuantity}吨，"
            "单价={unitCost}元/吨，灰分={ashContent}%，硫分={sulfurContent}%，水分={moistureContent}%，"
            "挥发分={volatileContent}%，发热量={calorificValue} kcal/kg，来源={source}".format(
                **_fmt_map(m)
            )
        )
    lines.append("\n【命中规则】")
    for r in rules[:8]:
        lines.append("- {rule_code} {rule_name}（{rule_type}）：{rule_content}".format(**_fmt_map(r)))
    lines.append("\n【参考案例】")
    for c in cases[:5]:
        lines.append("- {case_name}：{order_desc}；{blend_desc}；效果：{effectiveness_eval}".format(**_fmt_map(c)))
    lines.append("\n【RAG知识摘录】")
    for k in rag[:8]:
        lines.append("- {title}（{knowledge_type}）：{content}".format(**_fmt_map(k)))
    return "\n".join(lines)


def build_candidate_output(
    ranked_plans: list[dict[str, object | None]],
    detail_groups: dict[str, list[dict[str, object | None]]],
    coal_types: dict[str, dict[str, object | None]],
) -> dict[str, list[dict[str, Any]]]:
    plans: list[dict[str, Any]] = []
    for plan in ranked_plans[:5]:
        details = detail_groups.get(str(plan.get("id")), [])
        items = []
        for d in details[:4]:
            coal = coal_types.get(str(d.get("coal_id")), {})
            items.append(
                {
                    "coalId": _to_int(d.get("coal_id")),
                    "productBatchNo": str(d.get("product_batch_no") or ""),
                    "ratio": _round_ratio(d.get("blend_ratio")),
                    "reason": (
                        f"{coal.get('coal_name') or '该物料'}参与配煤；"
                        f"预测硫分{d.get('predicted_sulfur') or '—'}%，热值{d.get('predicted_calorific') or '—'}，"
                        f"兼顾质量约束与库存可执行性。"
                    ),
                }
            )
        if len(items) < 2:
            continue
        normalized = _normalize_items(items)
        if not normalized:
            continue
        plans.append(
            {
                "planName": str(plan.get("plan_name") or "候选方案"),
                "strategy": safe_text(
                    plan.get("ai_candidate_reason"),
                    f"综合评分{plan.get('overall_score') or '—'}，质量、成本和库存稳定性综合排序靠前。",
                ),
                "items": normalized,
                "risk": safe_text(plan.get("risk_tip"), "执行前复核煤质实测与库存余量。"),
            }
        )
    return {"plans": plans}


def public_samples_to_materials(samples: list[dict[str, object | None]]) -> list[dict[str, object | None]]:
    materials: list[dict[str, object | None]] = []
    for idx, s in enumerate(samples, start=1):
        materials.append(
            {
                "coalId": idx,
                "productBatchNo": f"PUBLIC-{idx:03d}",
                "name": s.get("sample_name"),
                "availableQuantity": s.get("available_quantity") or 6000,
                "unitCost": s.get("unit_cost") or _estimate_public_unit_cost(s),
                "ashContent": s.get("ash_content"),
                "sulfurContent": s.get("sulfur_content"),
                "moistureContent": s.get("moisture_content"),
                "volatileContent": s.get("volatile_content"),
                "calorificValue": s.get("calorific_value"),
                "source": s.get("source_title"),
                "sourceUrl": s.get("source_url"),
            }
        )
    return materials


def build_public_candidate_context(
    scenario: dict[str, object | None],
    materials: list[dict[str, object | None]],
) -> str:
    lines = [
        "你是煤矿智能配煤系统中的候选方案生成助手。以下候选物料来自公开煤质资料整理，适合用于模型调优基线实验。",
        "\n【候选范围】product_batch",
        "\n【订单信息】",
        (
            "订单编号：{orderCode}\n客户名称：{customerName}\n需求量：{demandQuantity} 吨\n"
            "灰分上限：{targetAsh}%\n硫分上限：{targetSulfur}%\n水分上限：{targetMoisture}%\n"
            "发热量下限：{targetCalorific} kcal/kg\n优先级：{priorityLevel}"
        ).format(**_fmt_map(scenario)),
        "\n【候选物料】",
    ]
    for m in materials:
        lines.append(
            "- coalId={coalId}，productBatchNo={productBatchNo}，名称={name}，可用量={availableQuantity}吨，"
            "单价={unitCost}元/吨，灰分={ashContent}%，硫分={sulfurContent}%，水分={moistureContent}%，"
            "挥发分={volatileContent}%，发热量={calorificValue} kcal/kg，公开来源={source}".format(
                **_fmt_map(m)
            )
        )
    lines.extend(
        _build_public_rules_context(scenario)
    )
    return "\n".join(lines)


def build_public_candidate_plans(
    scenario: dict[str, object | None],
    materials: list[dict[str, object | None]],
) -> list[dict[str, Any]]:
    by_batch = {str(m["productBatchNo"]): m for m in materials}

    # All 2-material and 3-material combinations (excluding BMK-SEAM-1 junk coal).
    premium_batches = [
        f"PUBLIC-{i:03d}" for i in range(1, len(by_batch) + 1)
    ]
    combos: list[list[tuple[str, float]]] = []

    # 2-coal combos: 60/40, 50/50, 40/60
    for i, a in enumerate(premium_batches):
        for b in premium_batches[i + 1 :]:
            for r1, r2 in [(0.6, 0.4), (0.5, 0.5), (0.4, 0.6)]:
                combos.append([(a, r1), (b, r2)])

    # 3-coal combos: 50/30/20, 40/40/20, 40/30/30, 50/25/25
    for i, a in enumerate(premium_batches):
        for j, b in enumerate(premium_batches[i + 1 :], i + 1):
            for c in premium_batches[j + 1 :]:
                for r1, r2, r3 in [(0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.4, 0.3, 0.3), (0.5, 0.25, 0.25)]:
                    combos.append([(a, r1), (b, r2), (c, r3)])

    plans: list[dict[str, Any]] = []
    used_signatures: set[str] = set()
    for combo in combos:
        selected = [(by_batch[b], r) for b, r in combo if b in by_batch]
        if len(selected) < 2:
            continue
        metrics = _weighted_public_metrics(selected)
        if not _public_plan_reasonable(scenario, metrics):
            continue
        sig = _combo_signature(selected)
        if sig in used_signatures:
            continue
        used_signatures.add(sig)
        items = [
            {
                "coalId": _to_int(m["coalId"]),
                "productBatchNo": str(m["productBatchNo"]),
                "ratio": round(ratio, 4),
                "reason": _public_item_reason(m, scenario),
            }
            for m, ratio in selected
        ]
        # Normalise ratios to sum exactly to 1.
        total_r = sum(float(i["ratio"]) for i in items)
        if total_r > 0:
            for i in items:
                i["ratio"] = round(float(i["ratio"]) / total_r, 4)
        plans.append(
            {
                "planName": f"公开煤质候选方案-{chr(64 + len(plans) + 1)}",
                "strategy": (
                    f"{scenario['strategy']} 加权预测灰分{metrics['ash']:.2f}%、"
                    f"硫分{metrics['sulfur']:.2f}%、水分{metrics['moisture']:.2f}%、"
                    f"热值{metrics['calorific']:.0f} kcal/kg。"
                ),
                "items": items,
                "risk": "公开样本来自文献或数据库摘要，实际执行前需用企业入厂煤质检测值复核。",
            }
        )
        if len(plans) >= 3:
            break
    return plans


def fallback_rule_basis(plan: dict[str, object | None], rules: list[dict[str, object | None]]) -> str:
    names = "、".join(str(r.get("rule_name")) for r in rules[:3] if r.get("rule_name"))
    if not names:
        return "当前知识库依据不足"
    return f"当前方案需要结合订单质量约束、库存可用性与成本控制进行判断，可参考规则：{names}。"


def fallback_case_reference(cases: list[dict[str, object | None]]) -> str:
    names = "、".join(str(c.get("case_name")) for c in cases[:2] if c.get("case_name"))
    return f"可参考历史案例：{names}。" if names else "当前知识库依据不足"


def fallback_recommend_reason(plan: dict[str, object | None]) -> str:
    return (
        f"该方案综合评分为 {plan.get('overall_score') or '未知'}，质量评分为 {plan.get('quality_score') or '未知'}，"
        f"成本评分为 {plan.get('cost_score') or '未知'}，稳定性评分为 {plan.get('stability_score') or '未知'}，"
        "系统根据质量、成本与库存稳定性进行综合推荐。"
    )


def build_final_explanation(
    plan: dict[str, object | None],
    details: list[dict[str, object | None]],
    coal_types: dict[str, dict[str, object | None]],
) -> str:
    if not details:
        return safe_text(plan.get("final_explanation"), safe_text(plan.get("explanation")))
    parts: list[str] = []
    for d in details:
        coal = coal_types.get(str(d.get("coal_id")), {})
        name = coal.get("coal_name") or f"煤种{d.get('coal_id')}"
        batch = str(d.get("product_batch_no") or "").strip()
        ratio = _num(d.get("blend_ratio")) * 100
        material = f"{name}（{batch}）" if batch else str(name)
        parts.append(f"{material}{ratio:.0f}%")
    return (
        "组合为 " + " + ".join(parts)
        + f"，预测灰分 {fmt_num(details[0].get('predicted_ash'))}%"
        + f"、硫分 {fmt_num(details[0].get('predicted_sulfur'))}%"
        + f"、水分 {fmt_num(details[0].get('predicted_moisture'))}%"
        + f"、发热量 {fmt_num(details[0].get('predicted_calorific'))} kcal/kg；"
        + f"质量分 {fmt_num(plan.get('quality_score'))}"
        + f"、成本分 {fmt_num(plan.get('cost_score'))}"
        + f"、库存稳定性分 {fmt_num(plan.get('stability_score'))}"
        + f"、综合分 {fmt_num(plan.get('overall_score'))}。"
    )


def fmt_num(value: object | None) -> str:
    if value is None or value == "":
        return "—"
    n = _num(value)
    if n == 0 and str(value).strip() not in {"0", "0.0", "0.00"}:
        return str(value)
    return f"{n:.2f}".rstrip("0").rstrip(".")


def _fmt_map(row: dict[str, object | None]) -> dict[str, str]:
    return {k: ("—" if v is None or v == "" else str(v)) for k, v in row.items()}


def _is_usable_candidate_plan(
    plan: dict[str, object | None],
    details: list[dict[str, object | None]],
) -> bool:
    if len(details) < 2 or len(details) > 4:
        return False
    ratios = [_num(d.get("blend_ratio")) for d in details]
    if any(r <= 0 for r in ratios):
        return False
    ratio_sum = sum(ratios)
    if ratio_sum < 0.95 or ratio_sum > 1.05:
        return False
    material_keys = set()
    coal_ids = set()
    for d in details:
        key = str(d.get("product_batch_no") or d.get("coal_id") or "")
        coal_id = str(d.get("coal_id") or "")
        if not key or key in material_keys or coal_id in coal_ids:
            return False
        material_keys.add(key)
        coal_ids.add(coal_id)
    return True


def _dedupe_materials(materials: list[dict[str, object | None]]) -> list[dict[str, object | None]]:
    out = []
    seen = set()
    for m in materials:
        key = str(m.get("productBatchNo") or m.get("coalId") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def _normalize_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ratio_sum = sum(float(i["ratio"]) for i in items if i.get("ratio") is not None)
    if ratio_sum <= 0:
        return []
    normalized = []
    running = 0.0
    for idx, item in enumerate(items):
        item = dict(item)
        if idx == len(items) - 1:
            item["ratio"] = round(1.0 - running, 4)
        else:
            item["ratio"] = round(float(item["ratio"]) / ratio_sum, 4)
            running += item["ratio"]
        normalized.append(item)
    if abs(sum(float(i["ratio"]) for i in normalized) - 1.0) > 0.01:
        return []
    return normalized


def _round_ratio(value: object | None) -> float:
    return round(_num(value), 4)


def _num(value: object | None) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(str(value).strip())
    except ValueError:
        return 0.0


def _to_int(value: object | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return None


def _has_public_quality(row: dict[str, object | None]) -> bool:
    required = ["sample_name", "ash_content", "sulfur_content", "moisture_content", "calorific_value"]
    return all(str(row.get(k) or "").strip() for k in required)


def _estimate_public_unit_cost(row: dict[str, object | None]) -> float:
    calorific = _num(row.get("calorific_value"))
    sulfur = _num(row.get("sulfur_content"))
    ash = _num(row.get("ash_content"))
    price = 260 + calorific / 20 - sulfur * 35 - ash * 4
    return round(max(180, min(price, 760)), 2)


def _combo_signature(selected: list[tuple[dict[str, object | None], float]]) -> str:
    parts = sorted(
        (str(m.get("productBatchNo") or m.get("coalId")), round(ratio, 4))
        for m, ratio in selected
    )
    return "|".join(f"{k}:{v}" for k, v in parts)


def _weighted_public_metrics(selected: list[tuple[dict[str, object | None], float]]) -> dict[str, float]:
    return {
        "ash": sum(_num(m.get("ashContent")) * r for m, r in selected),
        "sulfur": sum(_num(m.get("sulfurContent")) * r for m, r in selected),
        "moisture": sum(_num(m.get("moistureContent")) * r for m, r in selected),
        "calorific": sum(_num(m.get("calorificValue")) * r for m, r in selected),
    }


def _public_plan_reasonable(scenario: dict[str, object | None], metrics: dict[str, float]) -> bool:
    return (
        metrics["ash"] <= _num(scenario.get("targetAsh")) * 1.15
        and metrics["sulfur"] <= _num(scenario.get("targetSulfur")) * 1.20
        and metrics["moisture"] <= _num(scenario.get("targetMoisture")) * 1.35
        and metrics["calorific"] >= _num(scenario.get("targetCalorific")) * 0.90
    )


def _build_public_rules_context(scenario: dict[str, object | None]) -> list[str]:
    """Build realistic blending rule texts based on GB/T 25960-2010 and industry practice."""
    return [
        "\n【命中规则（GB/T 25960-2010 动力配煤规范）】",
        "- 挥发分差值限制规则：高挥发分煤与低挥发分煤Vdaf差值≥15%时，低挥发分煤配入量一般不大于20%，且须先做燃烧试验。",
        "- 硫分总量控制规则：配煤产品收到基硫分一般≤2.0%；高硫煤（St>2%）应优先洗选降硫，不宜直接配入。",
        "- 配煤煤种数量规则：一般以2-4种为宜，过多增加工艺复杂性和生产成本。",
        "- 发热量下限规则：大型电站锅炉用煤Qnet,ar≥23MJ/kg（≈5500kcal）；中小型锅炉≥18.82MJ/kg（≈4500kcal）。",
        "- 灰分对热效率影响规则：灰分每增加10%，锅炉热效率下降约3-4%；灰分≤30%时热效率可维持在76%以上。",
        "- 水分均匀性规则：水分过高影响混合均匀性和热效率；水分过低碳粉飞扬损失大；需控制各煤种水分差。",
        "\n【参考案例】",
        "- 酒钢宏晟准东煤掺烧案例：强结焦煤掺烧比从17%经分阶段优化提升至68%，通过一炉一策和二次风调整实现安全纯烧。",
        "- 张家口热电四煤种配比0:2:5:3案例：配煤价格较设计煤种低43元/吨，年节约约1800万元，锅炉效率仅下降0.5个百分点。",
        "- 曲靖电厂叠加配煤案例：4000kcal稳燃煤+3000kcal低质煤叠加配煤，年掺烧低质煤约98万吨，降本6048万元。",
        "- 大唐云冈300MW现货配煤案例：深度配烧模型使燃料成本下降3.47%，年节约6800万元，代价是厂用电率增高0.645%。",
        "\n【RAG知识摘录】",
        "- GB/T 25960-2010《动力配煤规范》2011年6月1日实施，含5条强制性条款，适用于电站锅炉、工业锅炉和工业窑炉。",
        "- 煤质指标中灰分、硫分、水分、挥发分、发热量均可按配比线性加权计算；灰熔点（ST/FT）具有非线性特征，需用BP神经网络等方法预测。",
        "- 不同煤田中：鄂尔多斯煤田煤质最优良（低灰低硫）；山西煤田煤种最全；准东煤高钠特性限制纯烧；西南煤田硫分偏高（部分>3%）。",
    ]


def _public_item_reason(material: dict[str, object | None], scenario: dict[str, object | None]) -> str:
    name = str(material.get("name") or "")
    reasons = []
    sulfur = _num(material.get("sulfurContent"))
    calorific = _num(material.get("calorificValue"))
    ash = _num(material.get("ashContent"))
    moisture = _num(material.get("moistureContent"))
    cost = _num(material.get("unitCost"))
    target_s = _num(scenario.get("targetSulfur"))
    target_cv = _num(scenario.get("targetCalorific"))
    target_ash = _num(scenario.get("targetAsh"))
    target_mt = _num(scenario.get("targetMoisture"))

    if sulfur <= target_s * 0.5:
        reasons.append(f"硫分({sulfur:.2f}%)远低于订单上限")
    elif sulfur <= target_s:
        reasons.append(f"硫分({sulfur:.2f}%)满足订单要求")
    elif sulfur <= target_s * 1.5:
        reasons.append(f"硫分({sulfur:.2f}%)略高，需搭配低硫煤")
    else:
        reasons.append(f"硫分({sulfur:.2f}%)偏高，控制掺配比例")

    if calorific >= target_cv * 1.1:
        reasons.append(f"热值({calorific:.0f}kcal)高，可作提热主力")
    elif calorific >= target_cv:
        reasons.append(f"热值({calorific:.0f}kcal)达标")
    elif calorific >= target_cv * 0.85:
        reasons.append(f"热值({calorific:.0f}kcal)偏低，需高热值煤补充")
    else:
        reasons.append(f"热值({calorific:.0f}kcal)低，少量用于降本")

    if ash <= target_ash * 0.6:
        reasons.append(f"灰分({ash:.1f}%)极低，有利拉低配煤灰分")
    elif ash <= target_ash:
        reasons.append(f"灰分({ash:.1f}%)在目标范围内")
    if moisture > target_mt and target_mt > 0:
        reasons.append(f"水分({moisture:.1f}%)超限，限制配比")
    if cost <= 300:
        reasons.append(f"成本低({cost:.0f}元/吨)，降本效果好")
    elif cost >= 700:
        reasons.append(f"单价较高({cost:.0f}元/吨)，适量使用控制总成本")

    return "；".join(reasons[:4])
