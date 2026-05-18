#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an Ollama Modelfile template for a merged or converted coal model."
    )
    parser.add_argument("--base", default="qwen3:4b")
    parser.add_argument("--output", default="outputs/adapters/Modelfile")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""FROM {args.base}

SYSTEM \"\"\"你是煤矿智能配煤系统中的候选方案生成助手。你必须依据输入中的订单、候选物料、规则、案例和知识库内容输出合法 JSON，不编造事实。\"\"\"

PARAMETER temperature 0.2
PARAMETER top_p 0.8
""",
        encoding="utf-8",
    )
    print(f"created: {output}")


if __name__ == "__main__":
    main()
