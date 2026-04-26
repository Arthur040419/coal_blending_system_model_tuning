#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_radar_metrics(path: str | Path) -> dict[str, float]:
    report_path = Path(path)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report.get("by_task", {}).get("candidate_generation", {}).get("radar_metrics", {})
    if not metrics:
        raise ValueError(f"No candidate_generation.radar_metrics found in {report_path}")
    return {str(k): float(v) for k, v in metrics.items()}


def polygon_points(metrics: dict[str, float], labels: list[str], cx: float, cy: float, radius: float) -> str:
    points = []
    for idx, label in enumerate(labels):
        angle = -math.pi / 2 + idx * 2 * math.pi / len(labels)
        value = max(0.0, min(1.0, metrics.get(label, 0.0)))
        x = cx + math.cos(angle) * radius * value
        y = cy + math.sin(angle) * radius * value
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def axis_point(idx: int, labels: list[str], cx: float, cy: float, radius: float) -> tuple[float, float]:
    angle = -math.pi / 2 + idx * 2 * math.pi / len(labels)
    return cx + math.cos(angle) * radius, cy + math.sin(angle) * radius


def build_svg(base: dict[str, float], tuned: dict[str, float], output: Path) -> None:
    labels = list(tuned.keys() or base.keys())
    for label in base:
        if label not in labels:
            labels.append(label)
    width, height = 760, 620
    cx, cy, radius = width / 2, 315, 210
    grid = []
    for scale in [0.2, 0.4, 0.6, 0.8, 1.0]:
        pts = " ".join(
            f"{axis_point(i, labels, cx, cy, radius * scale)[0]:.2f},{axis_point(i, labels, cx, cy, radius * scale)[1]:.2f}"
            for i in range(len(labels))
        )
        grid.append(f'<polygon points="{pts}" fill="none" stroke="#d9dee7" stroke-width="1"/>')
    axes = []
    label_nodes = []
    for idx, label in enumerate(labels):
        x, y = axis_point(idx, labels, cx, cy, radius)
        lx, ly = axis_point(idx, labels, cx, cy, radius + 44)
        anchor = "middle"
        if lx < cx - 20:
            anchor = "end"
        elif lx > cx + 20:
            anchor = "start"
        axes.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.2f}" y2="{y:.2f}" stroke="#c7ceda" stroke-width="1"/>')
        label_nodes.append(
            f'<text x="{lx:.2f}" y="{ly:.2f}" text-anchor="{anchor}" font-size="15" fill="#263244">{label}</text>'
        )
    base_pts = polygon_points(base, labels, cx, cy, radius)
    tuned_pts = polygon_points(tuned, labels, cx, cy, radius)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{cx}" y="44" text-anchor="middle" font-size="24" font-weight="700" fill="#172033">模型生成配煤方案业务质量对比</text>
  <text x="{cx}" y="76" text-anchor="middle" font-size="14" fill="#5a6678">指标越接近外圈表示业务效果越好</text>
  {"".join(grid)}
  {"".join(axes)}
  <polygon points="{base_pts}" fill="#7b8fae" fill-opacity="0.22" stroke="#53657f" stroke-width="2"/>
  <polygon points="{tuned_pts}" fill="#2f9e75" fill-opacity="0.28" stroke="#1f7a59" stroke-width="2"/>
  {"".join(label_nodes)}
  <rect x="246" y="548" width="18" height="10" fill="#7b8fae" fill-opacity="0.35" stroke="#53657f"/>
  <text x="274" y="558" font-size="14" fill="#263244">优化前模型</text>
  <rect x="392" y="548" width="18" height="10" fill="#2f9e75" fill-opacity="0.45" stroke="#1f7a59"/>
  <text x="420" y="558" font-size="14" fill="#263244">优化后模型</text>
</svg>
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot business-quality radar chart from eval reports.")
    parser.add_argument("--base-report", required=True, help="Evaluation report of the base model.")
    parser.add_argument("--tuned-report", required=True, help="Evaluation report of the tuned model.")
    parser.add_argument("--output", default="outputs/reports/business_quality_radar.svg")
    args = parser.parse_args()

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    build_svg(load_radar_metrics(args.base_report), load_radar_metrics(args.tuned_report), output)
    print(f"radar chart: {output}")


if __name__ == "__main__":
    main()
