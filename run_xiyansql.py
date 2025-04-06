# -*- coding: utf-8 -*-
"""
run_xiyansql.py (Refactored)

Loads a Text-to-SQL model based on .env configuration and generates SQL.
Handles device detection (CUDA, MPS, CPU) and reads settings from .env.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os
from dotenv import load_dotenv

# --- Load Configuration from .env file ---
load_dotenv()

MODEL_ID = os.getenv("MODEL_ID", "XGenerationLab/XiYanSQL-QwenCoder-3B-2502")
MODEL_DTYPE_STR = os.getenv("MODEL_DTYPE", "float16").lower()
DEVICE_PREFERENCE = os.getenv("DEVICE_PREFERENCE", "auto").lower()
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 300))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
TOP_P = float(os.getenv("TOP_P", 0.9))

print("--- Configuration ---")
print(f"MODEL_ID: {MODEL_ID}")
print(f"MODEL_DTYPE: {MODEL_DTYPE_STR}")
print(f"DEVICE_PREFERENCE: {DEVICE_PREFERENCE}")
print(f"MAX_NEW_TOKENS: {MAX_NEW_TOKENS}")
print(f"TEMPERATURE: {TEMPERATURE}")
print(f"TOP_P: {TOP_P}")
print("--------------------")

# --- Determine Model Data Type ---
if MODEL_DTYPE_STR == "float16":
    MODEL_DTYPE = torch.float16
elif MODEL_DTYPE_STR == "bfloat16":
    MODEL_DTYPE = torch.bfloat16
elif MODEL_DTYPE_STR == "float32":
    MODEL_DTYPE = torch.float32
else:
    print(f"警告: 无效的 MODEL_DTYPE '{MODEL_DTYPE_STR}' 在 .env 文件中。将使用 float16。")
    MODEL_DTYPE = torch.float16

# --- Determine Device ---
DEVICE = None
if DEVICE_PREFERENCE == "cuda":
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        print("警告: .env 请求使用 CUDA, 但 CUDA 不可用。")
elif DEVICE_PREFERENCE == "mps":
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        print("警告: .env 请求使用 MPS, 但 MPS 不可用。")
elif DEVICE_PREFERENCE == "cpu":
    DEVICE = "cpu"

# 如果是 'auto' 或之前的选择失败，则自动检测
if DEVICE is None:
    if torch.cuda.is_available():
        DEVICE = "cuda"
        print("自动检测到 CUDA 设备。")
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
        print("自动检测到 MPS 设备。")
    else:
        DEVICE = "cpu"
        print("未检测到 CUDA 或 MPS，将使用 CPU。")

print(f"最终使用的设备: {DEVICE.upper()}")

# --- 加载模型和分词器 ---
print(f"\n正在加载模型和分词器: {MODEL_ID}...")
print(f"使用数据类型: {MODEL_DTYPE}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 使用 accelerate 的 device_map="auto" 通常是管理大型模型跨设备(包括CPU offload)的最佳方式
    # 它会尝试利用 .env 中确定的 DEVICE，但也可能根据内存将部分层放到 CPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=MODEL_DTYPE,
        device_map="auto",  # 让 accelerate 自动处理设备放置
        trust_remote_code=True
    ).eval()
    print(f"模型已加载，通过 device_map='auto' 分配。主设备可能为: {model.device}")

except Exception as e:
    print(f"使用 {MODEL_DTYPE} 和 device_map='auto' 加载模型时出错: {e}")
    print("尝试不指定 dtype 加载 (使用默认 float32)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        print(f"模型已以默认精度加载，通过 device_map='auto' 分配。主设备可能为: {model.device}")
    except Exception as e_fallback:
        print(f"加载模型仍然失败: {e_fallback}")
        exit()

print("模型和分词器加载完成.")

# --- 定义 SQL 生成函数 ---
def generate_sql(schema, question, model, tokenizer, device, max_new_toks, temp, top_p_val):
    """
    根据数据库 Schema 和自然语言问题生成 SQL 查询。
    模型和分词器作为参数传入。
    """
    prompt = f"""给定以下 SQL 表结构，请将自然语言问题转换为 SQL 查询语句。

### 数据库 Schema:
{schema}

### 问题:
{question}

### SQL 查询:
"""
    print("\n--- 输入 Prompt (用于模型生成) ---")
    print(prompt)
    print("----------------------------------\n")

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # --- 将输入移动到模型主要所在的设备 ---
    try:
        # 当使用 device_map 时，模型可能跨设备，我们通常将输入移动到 model.device
        # 它代表了模型计算开始或主要的设备
        target_device = model.device
        inputs = inputs.to(target_device)
        print(f"输入数据已移动到设备: {target_device}")
    except Exception as e:
        print(f"警告: 无法将输入移动到 model.device ({getattr(model, 'device', 'N/A')}). 尝试使用检测到的设备 {device}. Error: {e}")
        try:
            inputs = inputs.to(device) # Fallback to the globally detected device
            print(f"输入数据已移动到设备: {device}")
        except Exception as e_fallback:
             print(f"错误: 无法将输入移动到任何设备: {e_fallback}")
             return "Error moving inputs to device." # Return error if input cannot be moved


    print("正在生成 SQL...")
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs, # 解包 inputs 字典 (input_ids, attention_mask)
            max_new_tokens=max_new_toks,
            do_sample=True if temp > 0 else False,
            temperature=temp if temp > 0 else 1.0,
            top_p=top_p_val,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    end_time = time.time()
    generation_time = end_time - start_time
    print(f"生成耗时: {generation_time:.2f} 秒")

    output_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    num_generated_tokens = len(outputs[0]) - inputs['input_ids'].shape[1]
    if generation_time > 0:
        tokens_per_sec = num_generated_tokens / generation_time
        print(f"生成速度: {tokens_per_sec:.2f} tokens/sec")

    return output_text.strip()

# --- 主程序逻辑 ---
if __name__ == "__main__":
    # 使用你的数据库 Schema
    db_schema = """
    CREATE TABLE statistical_data (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        conversation_id VARCHAR(50) COMMENT '会话ID',
                        year VARCHAR(10) COMMENT '年份',
                        month VARCHAR(10) COMMENT '月份',
                        market_level VARCHAR(50) COMMENT '市场等级',
                        company_code VARCHAR(50) COMMENT '单位编号',
                        company_name VARCHAR(100) COMMENT '单位名称',
                        factor_code VARCHAR(50) COMMENT '因子编号',
                        factor_name VARCHAR(100) COMMENT '因子名称',
                        factor_val DECIMAL(20, 6) COMMENT '因子值',
                        factor_huanbi DECIMAL(10, 4) COMMENT '因子环比',
                        factor_tongbi DECIMAL(10, 4) COMMENT '因子同比',
                        sys_origin VARCHAR(100) COMMENT '数据来源',
                        data_version VARCHAR(50) COMMENT '数据版本',
                        factor_unit VARCHAR(20) COMMENT '计算单位',
                        calc_time VARCHAR(20) COMMENT '计算时间',
                        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '数据采集时间'
                    );
    """

    # 示例问题
    natural_language_question = "帮我查下A公司2025年2月员工人数"

    # 调用生成函数
    generated_sql = generate_sql(
        schema=db_schema,
        question=natural_language_question,
        model=model,
        tokenizer=tokenizer,
        device=DEVICE, # Pass the determined device
        max_new_toks=MAX_NEW_TOKENS,
        temp=TEMPERATURE,
        top_p_val=TOP_P
    )

    print("\n\n--- 自然语言问题 ---")
    print(natural_language_question)
    print("--------------------")

    print("\n--- 模型生成的 SQL ---")
    print(generated_sql)
    print("----------------------\n")