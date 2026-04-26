#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coal_tuning.dataset import build_training_records, read_public_coal_quality_csv, split_and_write
from coal_tuning.sql_dump import parse_mysql_dump


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description="Build coal blending instruction tuning JSONL data from a MySQL dump."
    ).parse_args()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sql",
        default="../coal_blending_system/db/coal_blending_system_2026-04-26.sql",
        help="Path to the coal_blending_system MySQL dump.",
    )
    parser.add_argument(
        "--public-samples",
        default="data/raw/public_coal_quality_samples.csv",
        help="Optional public coal quality CSV used to augment candidate-generation samples.",
    )
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", type=int, default=3)
    parser.add_argument(
        "--tasks",
        choices=["all", "candidate", "explanation"],
        default="all",
        help="Which task records to build.",
    )
    args = parser.parse_args()

    sql_path = (ROOT / args.sql).resolve() if not Path(args.sql).is_absolute() else Path(args.sql)
    tables = parse_mysql_dump(sql_path)
    public_path = (
        (ROOT / args.public_samples).resolve()
        if args.public_samples and not Path(args.public_samples).is_absolute()
        else Path(args.public_samples) if args.public_samples else None
    )
    public_samples = read_public_coal_quality_csv(public_path)
    records = build_training_records(
        tables,
        public_samples=public_samples,
        include_explanation=args.tasks in ("all", "explanation"),
        include_candidate=args.tasks in ("all", "candidate"),
    )
    if not records:
        raise SystemExit("No usable training records were generated. Check SQL dump content.")

    train_path, eval_path = split_and_write(
        records,
        ROOT / args.output_dir,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    preview_path = ROOT / args.output_dir / "preview.json"
    preview_path.write_text(
        json.dumps(records[: args.preview], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"tables parsed: {len(tables)}")
    print(f"public samples loaded: {len(public_samples)}")
    print(f"records generated: {len(records)}")
    print(f"train file: {train_path}")
    print(f"eval file: {eval_path}")
    print(f"preview file: {preview_path}")


if __name__ == "__main__":
    main()
