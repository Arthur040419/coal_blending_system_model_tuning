# coal_blending_system_model_tuning

本科毕业设计“基于大模型的煤矿智能配煤系统设计与实现”的模型调优实验项目。

本项目的目标不是从零训练大模型，而是完成一套可复现、可答辩的轻量调优流程：

1. 从现有煤矿智能配煤系统 SQL 快照抽取订单、方案、候选配比、规则、案例、RAG 知识等数据。
2. 补充公开煤质资料样本，构造更真实的候选物料场景。
3. 构造双任务指令微调样本：候选方案生成 + 方案解释生成。
4. 使用开源大模型做 LoRA/QLoRA 领域适配。
5. 对比调优前后候选 JSON 合法率、配比合法率、后端可接收率、解释字段完整率等指标。
6. 将调优后的模型通过 Ollama 或 OpenAI Chat Completions 兼容服务接入主系统。

## 目录结构

```text
configs/                 LoRA 训练参数
data/raw/                公开煤质样本等原始辅助数据
data/processed/          生成后的 train/eval JSONL 数据
docs/                    实验说明和论文写作材料
outputs/reports/         评估结果
scripts/                 数据构建、训练、评估脚本
src/coal_tuning/         可复用的数据和提示词代码
```

## 环境准备

建议使用独立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果使用 QLoRA 且机器支持 CUDA，可额外安装：

```bash
pip install bitsandbytes
```

## 生成调优数据

默认读取兄弟目录中的后端 SQL 快照，并补充 `data/raw/public_coal_quality_samples.csv`：

```bash
python3 scripts/build_dataset.py
```

只生成候选方案生成任务：

```bash
python3 scripts/build_dataset.py --tasks candidate
```

只生成方案解释任务：

```bash
python3 scripts/build_dataset.py --tasks explanation
```

当前已生成：

- `data/processed/train.jsonl`
- `data/processed/eval.jsonl`
- `data/processed/preview.json`

数据格式包括：

- `messages`：适合 Chat 模型训练的 system/user/assistant 三段消息；
- `text`：兼容 SFTTrainer 的纯文本样本；
- `task=candidate_generation`：目标 JSON 字段与后端 `AiBlendCandidateServiceImpl` 保持一致，即 `{"plans":[...]}`。
- `task=plan_explanation`：目标 JSON 字段与后端 `ModelInferenceServiceImpl` 保持一致。

公开煤质数据来源说明见 [docs/公开煤质数据来源.md](docs/公开煤质数据来源.md)。

## 训练 LoRA

先根据机器配置修改 `configs/qwen_lora.yaml` 中的 `model_name_or_path`。

轻量实验建议从小模型开始，例如：

- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`

训练命令：

```bash
python3 scripts/train_lora.py --config configs/qwen_lora.yaml
```

训练产物默认输出到：

```text
outputs/adapters/qwen-coal-lora/
```

该目录可能很大，已在 `.gitignore` 中忽略。

## 评估

不加载模型时，脚本会检查评测集标准答案的 JSON 格式：

```bash
python3 scripts/evaluate_json_outputs.py --limit 20
```

加载基座模型评估：

```bash
python3 scripts/evaluate_json_outputs.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --limit 20 \
  --report-file outputs/reports/base_eval_report.json
```

加载 LoRA adapter 评估：

```bash
python3 scripts/evaluate_json_outputs.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter outputs/adapters/qwen-coal-lora \
  --limit 20 \
  --report-file outputs/reports/lora_eval_report.json
```

重点记录以下指标：

- 通用：`valid_json_rate`、`complete_field_rate`、`nonempty_field_rate`；
- 候选生成：`valid_candidate_plan_rate`、`valid_ratio_plan_rate`、`valid_item_count_plan_rate`；
- 解释生成：五个解释字段的非空率。

## 接入主系统

推荐两种方式：

1. 将 LoRA 合并到基座模型，再转换为 Ollama 可运行格式。
2. 使用兼容 OpenAI Chat Completions 的推理服务加载基座模型和 LoRA adapter。

主系统只需要在 `model_config` 表中新增或启用一条配置：

```text
model_name: qwen-coal-lora
model_type: LOCAL_OLLAMA 或 LLM
api_url: http://127.0.0.1:11434/v1/chat/completions
status: 1
```

后端无需大改，因为当前系统已经按 OpenAI Chat Completions 兼容格式调用模型。
