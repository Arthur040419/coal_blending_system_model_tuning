# 煤矿智能配煤管理系统 · 模型调优项目

本科毕业设计“基于大模型的煤矿智能配煤系统设计与实现”的模型调优实验项目。本项目不从零训练大模型，而是围绕已有后端业务数据构建一套可复现、可答辩的轻量领域适配流程：

1. 从后端 MySQL SQL 快照抽取订单、配煤方案、方案明细、规则、案例和 RAG 知识。
2. 补充公开煤质样本，构造更丰富的候选物料场景。
3. 生成指令微调数据，覆盖候选方案生成和方案解释生成两个任务。
4. 使用 Qwen 等开源大模型进行 LoRA/QLoRA 微调。
5. 将基座模型和调优模型接入主系统，用同一批订单做业务评分对比。
6. 将可用模型通过 Ollama 或 OpenAI Chat Completions 兼容服务接入后端。

## 项目关系

建议与以下两个兄弟项目配合使用：

- `../coal_blending_system`：主系统后端，提供数据快照、评分逻辑、模型配置和实验记录。
- `../coal_blending_system_frontend`：主系统前端，用于生成方案、查看模型效果和演示系统流程。

调优项目只负责实验流程；最终效果是否可用，以主系统后端评分和前端“模型效果”页面为准。

## 目录结构

```text
configs/                 LoRA/QLoRA 训练配置
data/raw/                公开煤质样本等原始辅助数据
data/processed*/         生成后的 train/eval JSONL 数据
docs/                    数据来源、实验说明和论文材料
outputs/adapters/        LoRA adapter、checkpoint 和 Ollama Modelfile
outputs/cache/           datasets 等缓存
outputs/reports/         评估报告和图表
scripts/                 数据构建、训练、评估、Modelfile 生成脚本
src/coal_tuning/         数据抽取、样本构造和 Prompt 模板代码
```

## 环境要求

- Python 3.9+
- 推荐 NVIDIA GPU + CUDA，用于 QLoRA 训练
- macOS MPS 可做小规模实验，速度和显存能力有限
- CPU 默认不训练大模型，只建议做数据构建和 dry-run 检查

创建虚拟环境：

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

如果 PowerShell 禁止激活脚本：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

CUDA 机器使用 4-bit QLoRA 时，可额外安装：

```bash
pip install bitsandbytes
```

## 数据来源

默认读取后端 SQL 快照：

```text
../coal_blending_system/db/coal_blending_system_2026-05-11.sql
```

公开煤质辅助样本：

```text
data/raw/public_coal_quality_samples.csv
```

公开样本来源说明见：

```text
docs/公开煤质数据来源.md
```

## 生成训练数据

生成完整双任务数据：

```bash
python3 scripts/build_dataset.py
```

默认输出：

```text
data/processed/train.jsonl
data/processed/eval.jsonl
data/processed/preview.json
```

只生成候选方案生成任务：

```bash
python3 scripts/build_dataset.py --tasks candidate
```

只生成方案解释任务：

```bash
python3 scripts/build_dataset.py --tasks explanation
```

推荐的候选方案专项数据：

```bash
python3 scripts/build_dataset.py \
  --tasks candidate \
  --compact-candidate-context \
  --candidate-prompt-style backend \
  --output-dir data/processed_candidate
```

当前更推荐使用“安全 LoRA”数据配方，增加公开场景配比扰动、方案名称池和风险文本池，降低模板记忆：

```bash
python3 scripts/build_dataset.py \
  --tasks candidate \
  --compact-candidate-context \
  --diversify-public \
  --diversify-variants 3 \
  --candidate-prompt-style backend \
  --output-dir data/processed_candidate_safe
```

用于对照实验的 real-only 数据，只保留真实 SQL 业务方案，舍弃合成公开场景：

```bash
python3 scripts/build_dataset.py \
  --tasks candidate \
  --compact-candidate-context \
  --real-only \
  --candidate-prompt-style backend \
  --output-dir data/processed_candidate_real_only
```

JSONL 每行包含：

- `messages`：system/user/assistant 三段对话。
- `prompt` / `completion`：适合 TRL completion-only SFT。
- `text`：兼容 SFTTrainer 的纯文本字段。
- `task=candidate_generation`：输出结构为 `{"plans":[...]}`，与后端候选生成服务对齐。
- `task=plan_explanation`：输出结构与后端解释生成服务对齐。

## 训练 LoRA

训练前建议先检查样本长度，确认 completion 不会被截断：

```bash
python3 scripts/train_lora.py \
  --config configs/qwen_lora_candidate_safe.yaml \
  --dry-run-data
```

推荐主路线：安全 LoRA 候选方案专项训练。

```bash
python3 scripts/train_lora.py \
  --config configs/qwen_lora_candidate_safe.yaml
```

对照实验：只使用真实 SQL 计划样本。

```bash
python3 scripts/train_lora.py \
  --config configs/qwen_lora_candidate_real_only.yaml
```

旧版候选方案专项训练入口仍保留：

```bash
python3 scripts/train_lora.py \
  --config configs/qwen_lora_candidate.yaml
```

完整双任务训练入口：

```bash
python3 scripts/train_lora.py \
  --config configs/qwen_lora.yaml
```

不想激活虚拟环境时，可直接使用虚拟环境解释器：

```bash
.venv/bin/python scripts/train_lora.py --config configs/qwen_lora_candidate_safe.yaml
```

训练脚本会自动检测设备：

- CUDA：支持 4-bit QLoRA。
- MPS：使用 fp16，适合小规模实验。
- CPU：默认拒绝正式训练，避免耗时过长或内存不足；如需烟雾测试，请使用专门的小模型配置并设置 `allow_cpu_training: true`。

## 配置文件说明

常用配置：

| 配置 | 用途 |
| --- | --- |
| `configs/qwen_lora_candidate_safe.yaml` | 推荐主路线，候选生成专项，低学习率、轻量 LoRA、best checkpoint |
| `configs/qwen_lora_candidate_real_only.yaml` | 真实 SQL 样本对照实验 |
| `configs/qwen_lora_candidate.yaml` | 旧版候选生成专项配置 |
| `configs/qwen_lora.yaml` | 候选生成 + 方案解释双任务训练 |
| `configs/qwen_lora_cpu_smoke.yaml` | CPU 小规模烟雾测试 |
| `configs/qwen2_5_3b_lora_candidate_safe.yaml` | Qwen2.5 3B 候选生成安全配置 |

配置中的重要字段：

- `model_name_or_path`：HuggingFace 模型 ID 或本地模型目录。
- `train_file` / `eval_file`：训练和验证 JSONL。
- `output_dir`：LoRA adapter 输出目录。
- `max_seq_length`：最大序列长度。
- `target_modules`：LoRA 注入模块。
- `learning_rate`、`num_train_epochs`、`lora_r`、`lora_alpha`：主要训练超参。
- `eval_strategy`、`load_best_model_at_end`、`metric_for_best_model`：训练中评估和最佳 checkpoint 选择。

注意：`qwen3:4b` 是 Ollama tag，不能直接作为 transformers 训练脚本的 `model_name_or_path`。训练脚本应使用 HuggingFace ID，例如 `Qwen/Qwen3-4B-Instruct-2507`，或本地模型目录。

## HuggingFace 下载问题

首次训练会下载基座模型。如果 Windows 出现 `WinError 10060`、`Read timed out` 或一直重试，说明网络无法稳定访问 HuggingFace。

方式一：设置代理后重新训练：

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7890"
$env:HTTPS_PROXY="http://127.0.0.1:7890"
python scripts/train_lora.py --config configs/qwen_lora_candidate_safe.yaml
```

方式二：先下载到本地，再把配置中的 `model_name_or_path` 改成本地目录：

```powershell
hf download Qwen/Qwen3-4B-Instruct-2507 `
  --local-dir D:\models\Qwen3-4B-Instruct-2507
```

配置示例：

```yaml
model_name_or_path: D:\models\Qwen3-4B-Instruct-2507
```

## 评估

本项目提供 JSON 格式和候选业务指标的离线评估脚本：

```bash
python3 scripts/evaluate_json_outputs.py
```

更重要的是主系统业务评估。推荐流程：

1. 在主系统后端 `model_config` 中启用基座模型。
2. 使用同一订单生成配煤方案，记录 `experimentCode`。
3. 切换为调优后模型，使用相同订单和相同 `experimentCode` 再次生成方案。
4. 在前端“模型效果”页面查看雷达图和综合效果。
5. 后端也可直接访问实验接口：

```text
GET /experimentRecord/page
GET /experimentRecord/byOrder/{orderId}
GET /experimentRecord/radar?orderId=1
GET /experimentRecord/model-effect
```

核心判断标准不是 `valid_json_rate`，而是主系统记录的业务指标：

- 质量达标
- 质量余量
- 成本优势
- 库存可执行
- 配比均衡
- 风险控制
- `model_effect_score` 或雷达图综合效果

只有调优模型在同一批订单上稳定高于基座模型，才建议接入主系统演示。

## 导出与接入 Ollama

本项目可生成 Ollama `Modelfile` 模板：

```bash
python3 scripts/make_ollama_modelfile.py \
  --base qwen3:4b \
  --output outputs/adapters/Modelfile
```

脚本会写入适合候选方案生成的 system prompt 和低温度参数。根据实际模型导出方式，你可以把 LoRA 合并后的模型或转换后的 GGUF 模型写入 `FROM`。

接入主系统时，在后端或前端“模型配置”中填写 OpenAI 兼容接口：

```text
模型名称：调优后的模型名称
接口地址：http://127.0.0.1:11434/v1/chat/completions
状态：启用
```

如果调优模型用于候选方案生成，必须稳定输出：

```json
{
  "plans": [
    {
      "planName": "...",
      "strategy": "...",
      "risk": "...",
      "items": [
        {
          "coalId": 1,
          "productBatchNo": "...",
          "ratio": 0.6
        }
      ]
    }
  ]
}
```

比例合计应接近 `1.0`，物料必须来自输入候选范围，不能编造不存在的批次。

## 调优失败诊断

早期 `qwen3-4b-coal-candidate-lora` 实验中，LoRA 接入主系统后业务评分低于 `Qwen3-4B-Instruct-2507` 基座。主要原因：

1. 训练集太小且分布失衡。
2. 合成标签模板化严重，方案名称、配比和风险文本重复。
3. 学习率和 LoRA 注入范围偏激进，容易破坏基座推理能力。
4. 没有训练中评估和 best checkpoint 选择。
5. 合成 gold label 本身业务评分上限不高。

因此当前推荐 `qwen_lora_candidate_safe.yaml`：

- 使用 `--diversify-public` 打散公开场景。
- 低学习率 `2e-5`。
- 只注入 `q_proj` 和 `v_proj`。
- `lora_alpha == lora_r`，降低额外缩放。
- 启用 eval、early stopping 和 `load_best_model_at_end`。
- 不从旧 adapter 继续训练。

`qwen_lora_candidate_real_only.yaml` 用于对照，样本少但更贴近真实业务计划。

## 常见问题

**训练脚本提示没有 CUDA/MPS**

当前机器不适合正式训练 Qwen LoRA。可改用 GPU 环境，或只运行 `--dry-run-data` 检查数据。

**训练时 completion 被截断**

增大配置中的 `max_seq_length`，或重新生成更紧凑的数据，例如增加 `--compact-candidate-context`。

**调优后 JSON 合法率高但业务效果差**

说明模型学会了格式，但没有生成更好的配煤方案。应以主系统评分为准，检查成本、库存、质量余量和风险控制，而不是只看 JSON 格式指标。

**模型输出编造批次**

候选生成 Prompt 和训练数据必须使用后端 `backend` 风格，并确保输出中的 `productBatchNo` 来自输入候选物料。必要时在后端继续保留候选合法性校验和过滤。

**Windows 训练中文模板报编码错误**

`scripts/train_lora.py` 已在导入 TRL 前修补 `Path.read_text()` 默认 UTF-8，一般无需额外处理。仍异常时请确认终端和源码文件均为 UTF-8。
