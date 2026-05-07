from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = r"D:\code\graduation_project\coal_blending_system_model_tuning\outputs\merged\qwen3-coal-merged"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

model.eval()

prompt = """
请根据以下订单生成配煤方案：
需求量：5000吨
目标灰分：不高于18%
目标硫分：不高于0.8%
目标发热量：不低于5000 kcal/kg

可用煤种：
1. 山西低硫贫煤：灰分12.5%，硫分0.35%，发热量5100 kcal/kg，价格520元/吨，库存3000吨
2. 内蒙古长焰煤：灰分20.0%，硫分0.60%，发热量4600 kcal/kg，价格390元/吨，库存6000吨
3. 陕西高热值烟煤：灰分15.0%，硫分0.75%，发热量5600 kcal/kg，价格610元/吨，库存2000吨
4. 高硫经济煤：灰分16.5%，硫分1.50%，发热量5200 kcal/kg，价格330元/吨，库存5000吨

请输出推荐配比、使用量、预测质量、成本估算和方案解释。
"""

messages = [
    {"role": "system", "content": "你是煤矿智能配煤系统助手。"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1000,
        temperature=0.3,
        top_p=0.9,
        do_sample=True
    )

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
)

print(response)