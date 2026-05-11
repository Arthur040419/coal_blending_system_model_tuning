# coal_blending_system_model_tuning

本科毕业设计“基于大模型的煤矿智能配煤系统设计与实现”的模型调优实验项目。

本项目的目标不是从零训练大模型，而是完成一套可复现、可答辩的轻量调优流程：

1. 从现有煤矿智能配煤系统 SQL 快照抽取订单、方案、候选配比、规则、案例、RAG 知识等数据。
2. 补充公开煤质资料样本，构造更真实的候选物料场景。
3. 构造双任务指令微调样本：候选方案生成 + 方案解释生成。
4. 使用开源大模型做 LoRA/QLoRA 领域适配。
5. 在主系统后端对调优前后模型生成的配煤方案进行统一评分，并记录质量、成本、库存和综合评分。
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

macOS/Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果 PowerShell 禁止激活脚本，可在当前窗口临时放开策略：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
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

默认基座模型已经配置为 `Qwen/Qwen2.5-1.5B-Instruct`，对应 Ollama 侧常用 tag `qwen2.5:1.5b`。注意：`qwen2.5:1.5b` 是 Ollama tag，不能直接作为 transformers 训练脚本的 `model_name_or_path`。

训练命令：

```bash
source .venv/bin/activate
python3 scripts/train_lora.py --config configs/qwen_lora.yaml
```

如果不想激活虚拟环境，也可以直接执行：

```bash
.venv/bin/python scripts/train_lora.py --config configs/qwen_lora.yaml
```

### HuggingFace 下载超时

训练首次运行会从 HuggingFace 下载 `Qwen/Qwen2.5-1.5B-Instruct`。如果 Windows 上出现 `WinError 10060`、`Read timed out` 或一直重试，说明网络无法稳定访问 HuggingFace。可选择以下任一方式：

方式一：在 PowerShell 中为当前窗口设置代理后重新训练：

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7890"
$env:HTTPS_PROXY="http://127.0.0.1:7890"
python scripts/train_lora.py --config configs/qwen_lora.yaml
```

其中端口需要改成你本机代理软件实际提供的 HTTP 端口。

方式二：先把模型下载到本地，再把 `configs/qwen_lora.yaml` 中的 `model_name_or_path` 改成本地目录：

```powershell
hf download Qwen/Qwen2.5-1.5B-Instruct `
  --local-dir D:\models\Qwen2.5-1.5B-Instruct
```

然后修改：

```yaml
model_name_or_path: D:\models\Qwen2.5-1.5B-Instruct
```

训练产物默认输出到：

```text
outputs/adapters/qwen2.5-1.5b-coal-lora/
```

该目录可能很大，已在 `.gitignore` 中忽略。

## 评估

模型优化前后对比建议以主系统后端为准：切换 `model_config` 中启用的模型，分别使用优化前模型和优化后模型生成同一订单的配煤方案。后端会将评分结果写入 `experiment_record` 表，并通过以下接口提供评分表和雷达图数据：

生成方案时可以传入相同的 `experimentCode`，便于把优化前后结果归为同一组实验。

```text
GET /experimentRecord/page
GET /experimentRecord/byOrder/{orderId}
GET /experimentRecord/radar?orderId=1
```

雷达图核心维度为：

- `质量匹配`：质量评分；
- `成本优势`：成本评分；
- `库存合理`：库存合理性评分；
- `综合效果`：综合评分。

下面的离线脚本主要用于训练前后的结构诊断和辅助验证。

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
  --adapter outputs/adapters/qwen2.5-1.5b-coal-lora \
  --limit 20 \
  --report-file outputs/reports/lora_eval_report.json
```

重点记录以下指标：

- 业务质量：`average_business_effect_score`、`best_plan_business_effect_score`、`radar_metrics`；
- 候选生成：`质量达标`、`质量余量`、`成本优势`、`库存可执行`、`配比均衡`、`风险控制`；
- 基础诊断：`valid_json_rate`、`complete_field_rate`、`valid_ratio_plan_rate`、`valid_item_count_plan_rate`；
- 解释生成：五个解释字段的非空率。

生成优化前后业务质量雷达图：

```bash
python3 scripts/plot_quality_radar.py \
  --base-report outputs/reports/base_eval_report.json \
  --tuned-report outputs/reports/lora_eval_report.json \
  --output outputs/reports/business_quality_radar.svg
```

论文中建议将格式合法率作为基础诊断，把业务质量雷达图和指标表作为主要对比结果。

## 接入主系统

推荐两种方式：

1. 将 LoRA 合并到基座模型，再转换为 Ollama 可运行格式。
2. 使用兼容 OpenAI Chat Completions 的推理服务加载基座模型和 LoRA adapter。

合并 LoRA：

```bash
python3 merge_lora.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter outputs/adapters/qwen2.5-1.5b-coal-lora \
  --output outputs/merged/qwen2.5-1.5b-coal-merged
```

生成 Ollama Modelfile 模板时，默认基座为 `qwen2.5:1.5b`：

```bash
python3 scripts/make_ollama_modelfile.py --output outputs/merged/Modelfile
```

主系统只需要在 `model_config` 表中新增或启用一条配置：

```text
model_name: qwen2.5-1.5b-coal-lora
model_type: LOCAL_OLLAMA 或 LLM
api_url: http://127.0.0.1:11434/v1/chat/completions
status: 1
```

后端无需大改，因为当前系统已经按 OpenAI Chat Completions 兼容格式调用模型。
