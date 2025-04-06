# -*- coding: utf-8 -*-
"""
api_server.py (v1.4.5 - Confirmed fallback to prompt_config.yaml)

使用 FastAPI 构建的 API 服务器，通过 OpenAI 兼容的 /v1/chat/completions 端点暴露本地 XiYanSQL 模型功能。
- 从 .env 加载配置。
- 从 prompt_config.yaml 加载默认 Schema 和默认 Prompt 模板。
- API 请求中的 db_schema 和 system message 优先，若缺失则使用配置文件中的默认值。
- 支持流式和非流式响应。
- 强化后处理，目标是仅返回提取出的 SQL 语句。
- 包含详细中文注释。
"""

# --- 基础 Python 库 ---
import time, os, re, yaml, json, uuid, asyncio
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from typing import List, Optional, Union, Dict, Any, AsyncGenerator
from threading import Thread

# --- 机器学习相关库 ---
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# --- Web 框架与工具 ---
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from starlette.status import HTTP_403_FORBIDDEN, HTTP_503_SERVICE_UNAVAILABLE, HTTP_400_BAD_REQUEST
from starlette.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
try:
    # 尝试导入 Python 3.9+ 的 to_thread
    from asyncio import to_thread
except ImportError:
    # 为旧版本 Python 提供后备方案 (虽然 Starlette 的 run_in_threadpool 更常用)
    async def to_thread(func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# --- 1. 加载 .env 配置 ---
# 从项目根目录下的 .env 文件加载环境变量
load_dotenv()
# 模型 ID，默认为 XiYanSQL
MODEL_ID = os.getenv("MODEL_ID", "XGenerationLab/XiYanSQL-QwenCoder-3B-2502")
# 模型加载时使用的数据类型 (float16, bfloat16, float32)，默认为 float16
MODEL_DTYPE_STR = os.getenv("MODEL_DTYPE", "float16").lower()
# 设备偏好 (auto, cuda, mps, cpu)，默认为 auto
DEVICE_PREFERENCE = os.getenv("DEVICE_PREFERENCE", "auto").lower()
# 模型生成时默认的最大新 token 数量
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", 300))
# 模型生成时默认的 temperature (控制随机性)
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", 0.1))
# 模型生成时默认的 top_p (核心采样)
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", 0.9))
# API 服务器监听的主机地址
API_HOST = os.getenv("API_HOST", "127.0.0.1")
# API 服务器监听的端口
API_PORT = int(os.getenv("API_PORT", 8000))
# 允许访问 API 的密钥列表 (逗号分隔)，为空则不启用认证
API_KEYS_STR = os.getenv("API_KEYS", "")
# 将密钥字符串处理成集合，方便快速查找
ALLOWED_API_KEYS = set(key.strip() for key in API_KEYS_STR.split(',') if key.strip())

# --- 2. 加载 prompt_config.yaml ---
# 定义配置文件的路径
PROMPT_CONFIG_PATH = "prompt_config.yaml"
# 初始化存储配置的字典
prompt_config = {}
# 初始化默认值，以防文件加载失败
DEFAULT_DB_SCHEMA = ""
DEFAULT_SYSTEM_MESSAGE_TEMPLATE = "你是一个 Text-to-SQL 助手。"
PROMPT_TEMPLATE = "{system_message}\n\n### Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:"
try:
    # 尝试以 UTF-8 编码打开并读取 YAML 文件
    with open(PROMPT_CONFIG_PATH, 'r', encoding='utf-8') as f:
        # 使用 PyYAML 安全加载 YAML 内容
        prompt_config = yaml.safe_load(f)
        # 如果文件为空，确保 prompt_config 是一个空字典
        if prompt_config is None:
            prompt_config = {}
            print(f"警告: 配置文件 {PROMPT_CONFIG_PATH} 为空。")
    print(f"成功加载 Prompt 配置文件: {PROMPT_CONFIG_PATH}")
    # 从加载的配置中获取值，如果键不存在，则使用之前的初始默认值
    DEFAULT_DB_SCHEMA = prompt_config.get("database_schema", DEFAULT_DB_SCHEMA)
    DEFAULT_SYSTEM_MESSAGE_TEMPLATE = prompt_config.get("default_system_message_template", DEFAULT_SYSTEM_MESSAGE_TEMPLATE)
    PROMPT_TEMPLATE = prompt_config.get("prompt_template", PROMPT_TEMPLATE)
# 处理文件未找到的异常
except FileNotFoundError:
    print(f"警告: 配置文件 {PROMPT_CONFIG_PATH} 未找到。将使用内置的默认值。")
# 处理 YAML 解析错误的异常
except yaml.YAMLError as e:
    print(f"错误: 解析配置文件 {PROMPT_CONFIG_PATH} 失败。 Error: {e}")
    exit(1) # 严重错误，退出程序
# 处理其他可能的加载错误
except Exception as e:
    print(f"加载配置文件 {PROMPT_CONFIG_PATH} 时发生未知错误: {e}")
    exit(1) # 严重错误，退出程序

# --- 3. 打印配置 ---
# 打印最终生效的关键配置信息
print("--- API 服务器配置 ---")
print(f"模型 ID (MODEL_ID): {MODEL_ID}")
# 报告是否成功从配置文件加载了默认 Schema
print(f"默认 DB Schema 已加载: {'是' if DEFAULT_DB_SCHEMA else '否 (或文件为空)'}")
# 报告是否成功从配置文件加载了默认系统消息
print(f"默认 System Message 已加载: {'是' if DEFAULT_SYSTEM_MESSAGE_TEMPLATE != '你是一个 Text-to-SQL 助手。' else '否 (或使用内置默认)'}")
# 报告是否成功从配置文件加载了 Prompt 模板
print(f"Prompt 模板已加载: {'是' if PROMPT_TEMPLATE != '{system_message}\n\n### Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:' else '否 (或使用内置默认)'}")
print("--------------------------")
# 如果未设置 API 密钥，发出警告
if not ALLOWED_API_KEYS:
    print("\n*** 警告: API_KEYS 未设置或为空! API 将处于不安全模式! ***\n")

# --- 4. 全局变量 ---
# 初始化模型、分词器、设备和加载状态的全局变量
model, tokenizer, DEVICE, MODEL_DTYPE, model_ready = None, None, None, None, False

# --- 5. 模型数据类型映射 ---
# 将字符串形式的数据类型映射到 torch 的实际数据类型
dtype_map = { "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32 }
# 获取实际的 torch 数据类型，默认为 float16
MODEL_DTYPE = dtype_map.get(MODEL_DTYPE_STR, torch.float16)
# 如果配置的类型无效，发出警告
if MODEL_DTYPE_STR not in dtype_map:
    print(f"警告: 无效的 MODEL_DTYPE '{MODEL_DTYPE_STR}'。将使用 float16。")

# --- 6. 设备检测逻辑 ---
# 函数用于确定运行模型的最佳设备 (CPU, CUDA, MPS)
def get_device() -> str:
    global DEVICE # 允许修改全局 DEVICE 变量
    # 如果设备已确定，直接返回
    if DEVICE: return DEVICE
    _device = None # 临时设备变量
    pref = DEVICE_PREFERENCE # 获取用户偏好
    print("\n--- 设备检测 ---")
    # 根据用户偏好尝试选择设备
    if pref == "cuda":
        if torch.cuda.is_available(): _device = "cuda"; print("检测到 CUDA 可用...")
        else: print("警告: 偏好 CUDA，但不可用。")
    elif pref == "mps": # MPS 主要用于 Apple Silicon Mac
        if torch.backends.mps.is_available(): _device = "mps"; print("检测到 MPS 可用...")
        else: print("警告: 偏好 MPS，但不可用。")
    elif pref == "cpu": _device = "cpu"; print("根据偏好选择 CPU。")
    # 如果偏好无效或不满足，则自动检测
    if _device is None:
        print(f"设备偏好为 '{pref}' 或失败。自动检测...")
        if torch.cuda.is_available(): _device = "cuda"; print("自动检测到 CUDA。")
        elif torch.backends.mps.is_available(): _device = "mps"; print("自动检测到 MPS。")
        else: _device = "cpu"; print("将使用 CPU。") # 最后备选 CPU
    # 设置全局 DEVICE 变量并打印最终选择
    DEVICE = _device; print(f"最终使用设备: {DEVICE.upper()}"); print("------------------")
    return DEVICE

# --- 7. 模型加载逻辑 ---
# 函数用于加载指定 ID 的模型和对应的分词器
def load_model_and_tokenizer():
    global model, tokenizer, model_ready, DEVICE, MODEL_DTYPE # 允许修改全局变量
    # 如果模型已加载，则无需重复加载
    if model_ready: return
    # 确定运行设备
    DEVICE = get_device()
    print(f"\n开始加载模型 '{MODEL_ID}' 到设备 '{DEVICE}' (类型 {MODEL_DTYPE})...")
    start_load_time = time.time() # 记录加载开始时间
    try:
        # 加载分词器，trust_remote_code=True 对于某些模型是必需的
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=MODEL_DTYPE, # 指定数据类型
            device_map="auto", # 让 transformers 自动分配模型层到可用设备 (如多 GPU)
            trust_remote_code=True # 允许执行模型仓库中的自定义代码
        ).eval() # 设置为评估模式 (禁用 dropout 等)
        # 报告模型实际加载到的设备
        print(f"模型已通过 device_map='auto' 加载。主设备: {getattr(model, 'device', 'N/A')}")
        model_ready = True # 标记模型已准备就绪
        load_time = time.time() - start_load_time # 计算加载耗时
        print(f"模型加载成功。耗时: {load_time:.2f} 秒。")
    except Exception as e:
        # 如果加载过程中发生任何错误，打印错误信息并标记模型未就绪
        print(f"!!! 错误: 加载模型失败! Error: {e}")
        model_ready = False

# --- 8. FastAPI 应用生命周期管理 ---
# 使用 asynccontextmanager 定义 FastAPI 应用的启动和关闭逻辑
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 应用启动时执行
    print("服务器启动中...")
    # 调用模型加载函数
    load_model_and_tokenizer()
    # yield 将控制权交给 FastAPI 应用本身，直到应用准备关闭
    yield
    # 应用关闭时执行
    print("服务器关闭中...")
    # 清理全局变量，释放资源
    global model, tokenizer, model_ready, DEVICE
    model, tokenizer, model_ready, DEVICE = None, None, False, None
    # 如果使用了 MPS 设备，尝试清空缓存
    if torch.backends.mps.is_available():
         try: torch.mps.empty_cache()
         except Exception as e_mps: print(f"清理 MPS 缓存时出错: {e_mps}")
    print("资源清理完成。")

# --- 9. FastAPI 应用初始化 ---
# 创建 FastAPI 应用实例，设置标题、版本号和生命周期管理器
app = FastAPI(
    title="XiYanSQL 本地 API 服务器 (SQL Only Output)",
    version="1.4.5", # 更新版本号以反映确认了 fallback 逻辑
    lifespan=lifespan
)

# --- 10. API 密钥认证 ---
# 定义 API 密钥在请求头中的名称
api_key_header_auth = APIKeyHeader(name="Authorization", auto_error=False)

# 依赖函数，用于验证 API 密钥
async def verify_api_key(api_key_header: Optional[str] = Security(api_key_header_auth)) -> str:
    # 如果环境变量中未设置任何 API 密钥，则跳过验证 (不安全模式)
    if not ALLOWED_API_KEYS:
        return "insecure_mode_key" # 返回一个占位符表示不安全模式
    # 如果请求头中没有 Authorization
    if api_key_header is None:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="请求头缺少 'Authorization'")
    # 检查是否是 Bearer Token 格式
    parts = api_key_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
         raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Authorization 头格式应为 'Bearer <YOUR_API_KEY>'")
    key = parts[1] # 提取密钥部分
    # 检查提取的密钥是否在允许的集合中
    if key in ALLOWED_API_KEYS:
        return key # 验证通过，返回密钥
    # 密钥无效
    raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="无效的 API 密钥")

# --- 11. Pydantic 模型定义 ---
# 定义 API 请求和响应的数据结构，用于数据验证和序列化

# 单条聊天消息结构
class ChatMessage(BaseModel):
    role: str # 角色 (system, user, assistant)
    content: str # 消息内容

# 流式响应中的消息增量部分
class DeltaMessage(BaseModel):
    role: Optional[str] = None # 可能包含角色 (通常只在第一个块)
    content: Optional[str] = None # 消息内容的增量

# 流式响应中单个选择的结构
class ChatCompletionChunkChoice(BaseModel):
    index: int # 通常为 0
    delta: DeltaMessage # 消息增量
    finish_reason: Optional[str] = None # 结束原因 (stop, length, error)

# 单个流式响应块 (SSE 事件)
class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}") # 随机生成 ID
    object: str = "chat.completion.chunk" # 固定值
    created: int = Field(default_factory=lambda: int(time.time())) # 创建时间戳
    model: str # 使用的模型 ID
    choices: List[ChatCompletionChunkChoice] # 包含增量信息

# API 请求体结构
class ChatCompletionRequest(BaseModel):
    model: str # 请求使用的模型 ID (主要用于客户端标识，服务器端使用 MODEL_ID)
    messages: List[ChatMessage] # 聊天消息列表
    temperature: Optional[float] = None # 可选的 temperature 参数
    max_tokens: Optional[int] = None # 可选的 max_tokens 参数 (对应 max_new_tokens)
    top_p: Optional[float] = None # 可选的 top_p 参数
    stream: Optional[bool] = False # 是否启用流式响应
    db_schema: Optional[str] = None # 可选的数据库 Schema (如果提供，则覆盖默认)

# 非流式响应中单个选择的结构
class Choice(BaseModel):
    index: int # 通常为 0
    message: ChatMessage # 完整的助手回复消息
    finish_reason: str # 结束原因

# Token 使用情况统计
class Usage(BaseModel):
    prompt_tokens: int # 输入提示的 token 数
    completion_tokens: int # 生成内容的 token 数 (基于模型原始输出)
    total_tokens: int # 总 token 数

# 完整的非流式响应结构
class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}") # 随机生成 ID
    object: str = "chat.completion" # 固定值
    created: int = Field(default_factory=lambda: int(time.time())) # 创建时间戳
    model: str # 使用的模型 ID
    choices: List[Choice] # 包含完整回复
    usage: Usage # Token 使用情况

# --- 12. 后处理函数: 提取 SQL ---
# (函数内容与之前版本一致，保持了 SQL 提取逻辑)
def extract_sql(raw_output: str, stop_markers: List[str] = ["###", "\n#", "\n\n"]) -> str:
    """
    从模型原始输出中更稳健地提取 SQL 语句。
    """
    if not raw_output:
        return ""
    # 1. 初步清理: 去除首尾空白和常见的代码块标记
    cleaned_output = raw_output.strip()
    if cleaned_output.startswith("```sql"):
        cleaned_output = cleaned_output[len("```sql"):].strip()
    elif cleaned_output.startswith("```"):
         cleaned_output = cleaned_output[len("```"):].strip()
    if cleaned_output.endswith("```"):
        cleaned_output = cleaned_output[:-len("```")].strip()

    # 2. 查找 SQL 起点: 寻找第一个出现的 SQL DML/DDL 关键字
    sql_keywords = ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"] # 增加了 DDL
    start_index = -1
    normalized_output_upper = cleaned_output.upper() # 转换为大写以便不区分大小写查找
    for keyword in sql_keywords:
        idx = normalized_output_upper.find(keyword)
        if idx != -1: # 如果找到了关键字
            # 如果是第一个找到的，或者比之前找到的更靠前，则更新起点索引
            if start_index == -1 or idx < start_index:
                start_index = idx

    # 3. 处理未找到 SQL 起点的情况
    if start_index == -1:
        print("警告: 未在模型输出中找到明确的 SQL 关键字起点。")
        # 尝试查找第一个分号作为备选方案
        first_semicolon = cleaned_output.find(';')
        if first_semicolon != -1:
             # 如果找到分号，认为到分号为止是 SQL
             maybe_sql = cleaned_output[:first_semicolon + 1]
             print("警告: 使用第一个分号作为 SQL 结束标记。")
             return maybe_sql.strip()
        else:
             # 如果连分号都没有，无法可靠判断，返回初步清理后的原始内容
             print("警告: 无法可靠提取 SQL，返回初步清理后的原始输出。")
             return cleaned_output

    # 4. 查找 SQL 结束点: 从 SQL 起点开始，查找第一个出现的停止标记
    sql_part_candidate = cleaned_output[start_index:] # 从起点开始的部分
    end_index = -1 # 结束点在 sql_part_candidate 中的索引

    # 遍历所有定义的停止标记
    for marker in stop_markers:
        idx = sql_part_candidate.find(marker)
        if idx != -1: # 如果找到标记
             # 如果是第一个找到的，或者比之前找到的更靠前，则更新结束点索引
            if end_index == -1 or idx < end_index:
                end_index = idx

    # 5. 截取 SQL
    if end_index != -1:
        # 如果找到了停止标记，截取从起点到标记之前的部分
        final_sql = sql_part_candidate[:end_index]
    else:
        # 如果未找到停止标记，认为从起点开始的所有内容都是 SQL
        final_sql = sql_part_candidate

    # 6. 返回最终清理后的 SQL
    return final_sql.strip()

# --- 13. 核心 SQL 生成逻辑 ---

# --- 非流式生成函数 (应用后处理) ---
def generate_sql_sync(system_message: str, schema: str, question: str, temp: float, top_p_val: float, max_new_toks: int) -> tuple[str, int, int]:
    """同步执行模型生成，并对结果进行后处理以提取 SQL。"""
    global model, tokenizer, DEVICE, PROMPT_TEMPLATE # 引用全局变量和从配置加载的模板
    if not model_ready:
        raise RuntimeError("模型未准备好")

    # 使用从配置加载或内置默认的 Prompt 模板构建最终输入
    try:
        prompt = PROMPT_TEMPLATE.format(system_message=system_message, schema=schema, question=question)
    except KeyError as e:
        # 处理模板占位符不匹配的理论情况
        print(f"错误: Prompt 模板占位符 {e} 缺失。使用备用格式。")
        prompt = f"{system_message}\n\n### 数据库 Schema:\n{schema}\n\n### 问题:\n{question}\n\n### SQL 查询:"

    # 使用分词器将 Prompt 文本转换为模型输入 ID
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    # 计算 Prompt 部分的 token 数量
    prompt_tokens = inputs['input_ids'].shape[1]

    # 尝试将输入数据移动到模型所在的设备
    try:
        target_device = getattr(model, 'device', DEVICE) # 获取模型实际设备
        inputs = inputs.to(target_device)
    except Exception as e_move:
        # 如果移动失败，尝试移动到全局 DEVICE
        print(f"警告: 移动输入到 {target_device} 失败, 尝试 {DEVICE}. Err: {e_move}")
        try:
            inputs = inputs.to(DEVICE)
        except Exception as e_inner:
            # 如果再次失败，则抛出运行时错误
            raise RuntimeError(f"无法移动输入到设备 {DEVICE}") from e_inner

    raw_output_text = "" # 初始化原始输出文本
    completion_tokens = 0 # 初始化完成 token 计数

    # 执行模型生成
    try:
        # 关闭梯度计算以节省内存和加速
        with torch.no_grad():
            outputs = model.generate(
                **inputs, # 输入 ID 和 attention mask
                max_new_tokens=max_new_toks, # 最大生成 token 数
                do_sample=temp > 0, # 是否进行采样 (temperature>0 时启用)
                temperature=temp if temp > 0 else 1.0, # 温度参数 (如果为0则设为1避免错误)
                top_p=top_p_val, # Top-P 采样参数
                pad_token_id=tokenizer.eos_token_id, # 使用 EOS token 作为 pad token
                eos_token_id=tokenizer.eos_token_id # 使用 EOS token 作为生成停止符
            )
        # 从输出中提取新生成的 token ID (排除输入的 prompt 部分)
        output_ids = outputs[0][inputs['input_ids'].shape[1]:]
        # 计算生成部分的 token 数
        completion_tokens = len(output_ids)
        # 将生成的 token ID 解码回文本，跳过特殊 token 并去除首尾空白
        raw_output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        print(f"模型原始输出 (前 100 字符): {raw_output_text[:100]}...")
    except Exception as e_gen:
        # 处理生成过程中可能发生的错误
        print(f"!!! 模型生成错误: {e_gen} !!!")
        raw_output_text = f"生成 SQL 出错: {e_gen}" # 返回错误信息
        completion_tokens = 0 # token 数设为 0

    # --- 应用后处理提取 SQL ---
    extracted_sql = extract_sql(raw_output_text)
    print(f"提取后的 SQL (前 100 字符): {extracted_sql[:100]}...")

    # 返回提取出的 SQL，以及 prompt 和 completion 的 token 数
    return extracted_sql, prompt_tokens, completion_tokens

# --- 流式生成函数 (应用后处理，只流式传输提取出的 SQL 部分) ---
async def generate_sql_stream(
    system_message: str, schema: str, question: str,
    temp: float, top_p_val: float, max_new_toks: int
) -> AsyncGenerator[str, None]:
    """异步生成器，用于流式生成 SQL 并进行实时后处理，仅流式传输提取到的 SQL 部分。"""
    global model, tokenizer, DEVICE, PROMPT_TEMPLATE, MODEL_ID # 引用全局变量
    # 检查模型是否就绪
    if not model_ready:
        error_detail = "模型未准备好或加载失败。"
        print(f"错误: {error_detail}")
        # 构建并发送错误信息块
        error_chunk = ChatCompletionChunk(model=MODEL_ID, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(role="assistant", content=f"错误: {error_detail}"), finish_reason="error")])
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield f"data: [DONE]\n\n" # 发送 SSE 结束标志
        return

    # --- 构建 Prompt ---
    try:
        prompt = PROMPT_TEMPLATE.format(system_message=system_message, schema=schema, question=question)
    except KeyError as e:
        print(f"错误: Prompt 模板占位符 {e} 缺失。使用备用格式。")
        prompt = f"{system_message}\n\n### 数据库 Schema:\n{schema}\n\n### 问题:\n{question}\n\n### SQL 查询:"

    # --- 输入处理、Streamer 设置、启动后台生成线程 ---
    try:
        # 准备模型输入
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        prompt_tokens = inputs['input_ids'].shape[1] # 记录 prompt token 数
        target_device = getattr(model, 'device', DEVICE)
        try: inputs = inputs.to(target_device) # 移动输入到设备
        except Exception: inputs = inputs.to(DEVICE) # 备用设备

        # 创建 TextIteratorStreamer 用于流式接收生成结果
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # 准备传递给 model.generate 的参数
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_toks,
            temperature=temp if temp > 0 else 1.0,
            top_p=top_p_val,
            do_sample=temp > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer # 传入 streamer 对象
        )
        # 在后台线程中运行模型生成，避免阻塞 FastAPI 事件循环
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        print("流式生成线程已启动...")

        # --- 迭代 Streamer 并应用后处理，只发送 SQL ---
        response_id = f"chatcmpl-{uuid.uuid4()}" # 为此流式响应生成唯一 ID
        start_time = time.time() # 记录开始时间
        full_raw_output = ""        # 用于累积模型返回的完整原始文本
        processed_sql_yielded = ""  # 记录到目前为止已提取并发送给客户端的 SQL 文本
        stop_yielding = False       # 标记是否应停止向客户端发送新的内容
        finish_reason = "stop"      # 默认的结束原因

        # 发送第一个数据块，包含角色信息
        first_chunk = ChatCompletionChunk(id=response_id, model=MODEL_ID, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(role="assistant"), finish_reason=None)])
        yield f"data: {first_chunk.model_dump_json()}\n\n"

        # 迭代 streamer 输出的文本块
        try:
            for raw_chunk in streamer: # streamer 是同步可迭代的
                if not raw_chunk: continue # 跳过可能的空块

                # 累积原始输出
                full_raw_output += raw_chunk

                # 如果尚未决定停止发送
                if not stop_yielding:
                    # 对当前累积的全部原始输出调用 SQL 提取函数
                    current_extracted_sql = extract_sql(full_raw_output)

                    # 检查新提取的 SQL 是否比已经发送过的更长
                    if len(current_extracted_sql) > len(processed_sql_yielded):
                        # 计算新增加的 SQL 文本部分
                        new_sql_part = current_extracted_sql[len(processed_sql_yielded):]
                        if new_sql_part: # 确保有新内容才发送
                            # 创建包含新 SQL 部分的数据块
                            chunk = ChatCompletionChunk(
                                id=response_id, model=MODEL_ID,
                                choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(content=new_sql_part), finish_reason=None)]
                            )
                            # 发送数据块 (SSE 格式)
                            yield f"data: {chunk.model_dump_json()}\n\n"
                            # 更新已发送 SQL 的记录
                            processed_sql_yielded = current_extracted_sql
                    else:
                        # 如果提取的 SQL 没变长，但原始输出还在增加，说明模型可能开始输出非 SQL 内容
                        # 这是一种启发式停止策略，可以防止发送不必要的文本
                        if len(full_raw_output) > len(processed_sql_yielded): # 检查原始输出长度是否仍在增长
                             print("停止流式发送：提取的 SQL 未增长，可能已开始输出额外内容。")
                             stop_yielding = True
                             # 可以选择在这里就设置 finish_reason

                    # 也可以根据停止标记提前停止发送 (可选策略)
                    if "###" in raw_chunk and not stop_yielding: # 检查新块中是否有停止符
                         print(f"停止流式发送：检测到停止标记 '###'。")
                         stop_yielding = True

        except Exception as e_iter:
             # 处理迭代 streamer 时的错误
             print(f"!!! 迭代 streamer 时出错: {e_iter} !!!")
             finish_reason = "error" # 标记结束原因为错误
             try: # 尝试向客户端发送错误信息
                 error_chunk = ChatCompletionChunk(id=response_id, model=MODEL_ID, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(content=f"\n迭代错误: {e_iter}"), finish_reason="error")])
                 yield f"data: {error_chunk.model_dump_json()}\n\n"
             except Exception: pass # 忽略发送错误本身可能引发的错误

        # --- 确保后台线程结束 ---
        # 等待模型生成线程执行完毕
        if thread.is_alive():
            print("等待后台生成线程结束...")
            # 使用 to_thread 或 run_in_threadpool 在异步上下文中等待同步线程
            await to_thread(thread.join)
            print("后台生成线程已确认结束。")
        else:
            print("后台生成线程已自行结束。")
        end_time = time.time() # 记录结束时间
        generation_time = end_time - start_time # 计算总耗时

        # --- 发送结束块 ---
        # finish_reason 可能在迭代中被设为 error，否则保持默认的 stop
        final_chunk = ChatCompletionChunk(id=response_id, model=MODEL_ID, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(), finish_reason=finish_reason)])
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield f"data: [DONE]\n\n" # SSE 标准结束标志

        # --- 日志记录 ---
        # 基于累积的原始输出计算完成 token 数
        completion_tokens = len(tokenizer.encode(full_raw_output, add_special_tokens=False)) if full_raw_output else 0
        print(f"流式生成处理完毕。总耗时: {generation_time:.2f} 秒。")
        # 计算并打印生成速度 (基于原始输出 token 数)
        if generation_time > 0 and completion_tokens > 0:
            tokens_per_sec = completion_tokens / generation_time
            print(f"生成速度 (基于原始输出估算): {tokens_per_sec:.2f} tokens/sec。 Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")
        # 打印最终发送给客户端的 SQL 长度，用于对比
        print(f"最终发送给客户端的 SQL 长度: {len(processed_sql_yielded)}")

    except Exception as e_stream_outer: # 捕获流式函数顶层的其他错误
        print(f"!!! 流式生成函数顶层发生错误: {e_stream_outer} !!!")
        try: # 尝试发送错误信息
            error_chunk = ChatCompletionChunk(model=MODEL_ID, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(content=f"\n服务器内部流式错误: {e_stream_outer}"), finish_reason="error")])
            yield f"data: {error_chunk.model_dump_json()}\n\n"
            yield f"data: [DONE]\n\n"
        except Exception: pass
        # 确保线程清理
        if 'thread' in locals() and thread.is_alive():
            # 这里不直接 join，因为可能在异步事件循环中，但至少记录下来
            print("警告: 流式处理外层错误，后台线程可能仍在运行。")


# --- 14. API 端点 (处理 /v1/chat/completions POST 请求) ---
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, api_key: str = Security(verify_api_key)):
    """
    处理聊天补全请求的主端点。
    根据请求中的 'stream' 参数决定调用流式或非流式处理逻辑。
    """
    # 检查模型是否已准备就绪
    if not model_ready:
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="模型未准备好。")

    # --- 提取和验证输入 ---
    if not request.messages:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="'messages' 列表不能为空。")

    # 提取最后一个用户消息作为问题
    last_user_message = next((msg for msg in reversed(request.messages) if msg.role == "user"), None)
    if not last_user_message:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="未找到 'user' 消息。")
    question = last_user_message.content

    # --- 确定数据库 Schema ---
    # 优先使用请求中提供的 db_schema
    db_schema = request.db_schema
    if db_schema:
        print("使用请求中提供的 db_schema。")
    else:
        # 如果请求中没有，则使用从 prompt_config.yaml 加载的默认 Schema
        db_schema = DEFAULT_DB_SCHEMA
        if db_schema:
            print("请求中未提供 db_schema，使用配置文件中的默认 db_schema。")
        else:
            # 如果请求和配置文件中都没有 Schema，则报错
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="请求体和配置文件中均未提供数据库 Schema (db_schema)。"
            )

    # --- 确定 System Message ---
    # 尝试从请求消息中查找 system 角色的消息
    system_message_content = next((msg.content for msg in request.messages if msg.role == "system"), None)
    if system_message_content:
        print("使用请求中提供的 system message。")
    else:
        # 如果请求中没有，则使用从 prompt_config.yaml 加载的默认系统消息
        system_message_content = DEFAULT_SYSTEM_MESSAGE_TEMPLATE
        print("请求中未提供 system message，使用配置文件中的默认 system message。")
        # 确保默认系统消息不是空的
        if not system_message_content:
             print("警告: 默认系统消息模板为空！")
             # 可以选择在这里设置一个最终的硬编码后备，或者允许为空
             # system_message_content = "You are a helpful assistant."


    # 获取生成参数，如果请求中未提供，则使用 .env 或代码中定义的默认值
    temperature = request.temperature if request.temperature is not None else DEFAULT_TEMPERATURE
    max_new_tokens = request.max_tokens if request.max_tokens is not None else DEFAULT_MAX_NEW_TOKENS
    top_p = request.top_p if request.top_p is not None else DEFAULT_TOP_P

    # --- 根据 stream 参数调用不同处理逻辑 ---
    if request.stream:
        # --- 处理流式请求 ---
        print(f"开始处理流式请求 (问题: '{question[:50]}...')")
        # 返回 StreamingResponse，其内容由 generate_sql_stream 异步生成器提供
        return StreamingResponse(
            generate_sql_stream( # 调用流式生成函数
                system_message=system_message_content, schema=db_schema, question=question,
                temp=temperature, top_p_val=top_p, max_new_toks=max_new_tokens
            ),
            media_type="text/event-stream" # 设置正确的 MIME 类型用于 Server-Sent Events
        )
    else:
        # --- 处理非流式请求 ---
        print(f"开始处理非流式请求 (问题: '{question[:50]}...')")
        start_gen_time = time.time() # 记录开始时间
        try:
            # 在线程池中运行同步的生成函数，避免阻塞事件循环
            extracted_sql, prompt_tokens, completion_tokens = await run_in_threadpool(
                generate_sql_sync, # 调用非流式生成函数
                system_message=system_message_content, schema=db_schema, question=question,
                temp=temperature, top_p_val=top_p, max_new_toks=max_new_tokens
            )
        except RuntimeError as e:
            # 捕获模型相关的运行时错误
            raise HTTPException(status_code=500, detail=f"SQL 生成失败: {e}")
        except Exception as e:
            # 捕获其他意外错误
            raise HTTPException(status_code=500, detail=f"生成过程中意外错误: {e}")

        end_gen_time = time.time() # 记录结束时间
        gen_duration = end_gen_time - start_gen_time # 计算耗时
        print(f"非流式生成执行完毕。耗时: {gen_duration:.2f} 秒。")

        # 日志记录生成速度 (基于原始输出 token)
        if gen_duration > 0 and completion_tokens > 0:
             tokens_per_sec = completion_tokens / gen_duration
             print(f"生成速度 (基于原始输出): {tokens_per_sec:.2f} tokens/sec。 Prompt: {prompt_tokens}, Completion: {completion_tokens}")
        print(f"最终返回的 SQL 长度: {len(extracted_sql)}")


        # 构建符合 OpenAI 格式的非流式响应
        # 注意: message.content 使用的是后处理提取出的 SQL
        response_message = ChatMessage(role="assistant", content=extracted_sql)
        choice = Choice(index=0, message=response_message, finish_reason="stop") # 假设正常结束为 "stop"
        # Usage 字段仍使用基于原始模型输出计算的 token 数
        usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens)
        response = ChatCompletionResponse(model=MODEL_ID, choices=[choice], usage=usage)

        try:
             # 尝试打印准备返回的响应内容 (用于调试)
             print(f"\n--- 准备返回非流式响应 ---\n{response.model_dump_json(indent=2)}\n--------------------\n")
        except Exception as e_dump:
             print(f"警告: 序列化响应出错: {e_dump}")

        # 返回 Pydantic 模型，FastAPI 会自动序列化为 JSON
        return response

# --- 15. 健康检查端点 ---
@app.get("/health", summary="健康检查", tags=["管理"])
async def health_check():
    """
    提供一个简单的健康检查端点，用于监控服务状态和模型加载情况。
    """
    if model_ready:
        # 如果模型已加载，返回成功状态和设备信息
        return {"status": "ok", "model_ready": True, "device": DEVICE}
    else:
        # 如果模型未加载或加载失败，返回 503 服务不可用错误
        raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail={"status": "error", "model_ready": False, "message": "模型加载失败或未完成。"})

# --- 16. 启动服务器的主入口 ---
# 当直接运行此脚本时 (__name__ == "__main__")
if __name__ == "__main__":
    # 导入 uvicorn ASGI 服务器
    import uvicorn
    print(f"\n准备启动 Uvicorn 服务器，监听地址: http://{API_HOST}:{API_PORT}")
    # 运行 uvicorn 服务器
    # "api_server:app" 指的是当前文件名 (api_server.py) 和 FastAPI 应用实例名 (app)
    # host 和 port 从环境变量或默认值获取
    # reload=False 表示禁用代码更改时的自动重载 (生产环境推荐)
    uvicorn.run("api_server:app", host=API_HOST, port=API_PORT, reload=False)