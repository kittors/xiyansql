# -*- coding: utf-8 -*-
"""
api_server.py (v1.4.4 - Enhanced post-processing for SQL-only output)

使用 FastAPI 构建的 API 服务器，通过 OpenAI 兼容的 /v1/chat/completions 端点暴露本地 XiYanSQL 模型功能。
- 从 .env 加载配置。
- 从 prompt_config.yaml 加载 Schema/Prompt 模板。
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
    from asyncio import to_thread # Python 3.9+
except ImportError:
    # Fallback for older Python versions if needed, though run_in_threadpool is available via starlette
    async def to_thread(func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# --- 1. 加载 .env 配置 ---
load_dotenv()
MODEL_ID = os.getenv("MODEL_ID", "XGenerationLab/XiYanSQL-QwenCoder-3B-2502")
MODEL_DTYPE_STR = os.getenv("MODEL_DTYPE", "float16").lower()
DEVICE_PREFERENCE = os.getenv("DEVICE_PREFERENCE", "auto").lower()
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", 300))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", 0.1))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", 0.9))
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", 8000))
API_KEYS_STR = os.getenv("API_KEYS", "")
ALLOWED_API_KEYS = set(key.strip() for key in API_KEYS_STR.split(',') if key.strip())

# --- 2. 加载 prompt_config.yaml ---
PROMPT_CONFIG_PATH = "prompt_config.yaml"
prompt_config = {}
try:
    with open(PROMPT_CONFIG_PATH, 'r', encoding='utf-8') as f:
        prompt_config = yaml.safe_load(f)
        if prompt_config is None: prompt_config = {}; print(f"警告: 配置文件 {PROMPT_CONFIG_PATH} 为空。")
    print(f"成功加载 Prompt 配置文件: {PROMPT_CONFIG_PATH}")
    DEFAULT_DB_SCHEMA = prompt_config.get("database_schema", "")
    DEFAULT_SYSTEM_MESSAGE_TEMPLATE = prompt_config.get("default_system_message_template", "你是一个 Text-to-SQL 助手。")
    PROMPT_TEMPLATE = prompt_config.get("prompt_template", "{system_message}\n\n### Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:")
except FileNotFoundError:
    print(f"警告: 配置文件 {PROMPT_CONFIG_PATH} 未找到。"); 
    DEFAULT_DB_SCHEMA, DEFAULT_SYSTEM_MESSAGE_TEMPLATE, PROMPT_TEMPLATE = "", "助手。", "{system_message}\n{schema}\n{question}\nSQL:"
except yaml.YAMLError as e: print(f"错误: 解析配置文件 {PROMPT_CONFIG_PATH} 失败。 Error: {e}"); exit(1)
except Exception as e: print(f"加载配置文件 {PROMPT_CONFIG_PATH} 时发生未知错误: {e}"); exit(1)

# --- 3. 打印配置 ---
print("--- API 服务器配置 ---") 
print(f"模型 ID (MODEL_ID): {MODEL_ID}")
print(f"默认 DB Schema 已加载: {'是' if DEFAULT_DB_SCHEMA else '否'}")
print("--------------------------")
if not ALLOWED_API_KEYS: print("\n*** 警告: API_KEYS 未设置或为空! ***\n")

# --- 4. 全局变量 ---
model, tokenizer, DEVICE, MODEL_DTYPE, model_ready = None, None, None, None, False

# --- 5. 模型数据类型映射 ---
dtype_map = { "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32 }
MODEL_DTYPE = dtype_map.get(MODEL_DTYPE_STR, torch.float16)
if MODEL_DTYPE_STR not in dtype_map: print(f"警告: 无效的 MODEL_DTYPE '{MODEL_DTYPE_STR}'。将使用 float16。")

# --- 6. 设备检测逻辑 ---
def get_device() -> str:
    global DEVICE; 
    if DEVICE: return DEVICE
    _device = None; pref = DEVICE_PREFERENCE; print("\n--- 设备检测 ---")
    if pref == "cuda":
        if torch.cuda.is_available(): _device = "cuda"; print("检测到 CUDA 可用...")
        else: print("警告: 偏好 CUDA，但不可用。")
    elif pref == "mps":
        if torch.backends.mps.is_available(): _device = "mps"; print("检测到 MPS 可用...")
        else: print("警告: 偏好 MPS，但不可用。")
    elif pref == "cpu": _device = "cpu"; print("根据偏好选择 CPU。")
    if _device is None:
        print(f"设备偏好为 '{pref}' 或失败。自动检测...")
        if torch.cuda.is_available(): _device = "cuda"; print("自动检测到 CUDA。")
        elif torch.backends.mps.is_available(): _device = "mps"; print("自动检测到 MPS。")
        else: _device = "cpu"; print("将使用 CPU。")
    DEVICE = _device; print(f"最终使用设备: {DEVICE.upper()}"); print("------------------")
    return DEVICE

# --- 7. 模型加载逻辑 ---
def load_model_and_tokenizer():
    global model, tokenizer, model_ready, DEVICE, MODEL_DTYPE; 
    if model_ready: return
    DEVICE = get_device()
    print(f"\n开始加载模型 '{MODEL_ID}' 到设备 '{DEVICE}' (类型 {MODEL_DTYPE})...")
    start_load_time = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=MODEL_DTYPE, device_map="auto", trust_remote_code=True
        ).eval()
        print(f"模型已通过 device_map='auto' 加载。主设备: {getattr(model, 'device', 'N/A')}")
        model_ready = True; load_time = time.time() - start_load_time
        print(f"模型加载成功。耗时: {load_time:.2f} 秒。")
    except Exception as e:
        print(f"!!! 错误: 加载模型失败! Error: {e}"); model_ready = False

# --- 8. FastAPI 应用生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("服务器启动中..."); load_model_and_tokenizer()
    yield
    print("服务器关闭中..."); 
    global model, tokenizer, model_ready, DEVICE; model, tokenizer, model_ready, DEVICE = None, None, False, None;
    if torch.backends.mps.is_available():
         try: torch.mps.empty_cache()
         except Exception as e_mps: print(f"清理 MPS 缓存时出错: {e_mps}")
    print("资源清理完成。")

# --- 9. FastAPI 应用初始化 ---
app = FastAPI(
    title="XiYanSQL 本地 API 服务器 (SQL Only Output)",
    version="1.4.4", # 更新版本号
    lifespan=lifespan
)

# --- 10. API 密钥认证 ---
api_key_header_auth = APIKeyHeader(name="Authorization", auto_error=False)
async def verify_api_key(api_key_header: Optional[str] = Security(api_key_header_auth)) -> str:
    if not ALLOWED_API_KEYS: return "insecure_mode_key"
    if api_key_header is None: raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="请求头缺少 'Authorization'")
    if api_key_header.startswith("Bearer "):
        key = api_key_header.replace("Bearer ", ""); 
        if key in ALLOWED_API_KEYS: return key
    raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="无效的 API 密钥或格式")

# --- 11. Pydantic 模型定义 ---
class ChatMessage(BaseModel): role: str; content: str
class DeltaMessage(BaseModel): role: Optional[str] = None; content: Optional[str] = None
class ChatCompletionChunkChoice(BaseModel): index: int; delta: DeltaMessage; finish_reason: Optional[str] = None
class ChatCompletionChunk(BaseModel): id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}"); object: str = "chat.completion.chunk"; created: int = Field(default_factory=lambda: int(time.time())); model: str; choices: List[ChatCompletionChunkChoice]
class ChatCompletionRequest(BaseModel): model: str; messages: List[ChatMessage]; temperature: Optional[float] = None; max_tokens: Optional[int] = None; top_p: Optional[float] = None; stream: Optional[bool] = False; db_schema: Optional[str] = None
class Choice(BaseModel): index: int; message: ChatMessage; finish_reason: str
class Usage(BaseModel): prompt_tokens: int; completion_tokens: int; total_tokens: int
class ChatCompletionResponse(BaseModel): id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}"); object: str = "chat.completion"; created: int = Field(default_factory=lambda: int(time.time())); model: str; choices: List[Choice]; usage: Usage

# --- 12. 后处理函数: 提取 SQL ---
def extract_sql(raw_output: str, stop_markers: List[str] = ["###", "\n#", "\n\n"]) -> str:
    """
    从模型原始输出中更稳健地提取 SQL 语句。
    策略:
    1. 清理首尾 ```sql / ``` 代码块标记和空白。
    2. 查找第一个 SQL 关键字（SELECT, WITH, INSERT, UPDATE, DELETE）作为起点。
    3. 如果找到起点，从起点开始查找第一个出现的停止标记（来自列表）。
    4. 如果找到停止标记，取起点到标记之间的内容。
    5. 如果未找到停止标记，则认为从起点开始的全部内容都是 SQL (或至少是我们能提取的最佳部分)。
    6. 如果未找到 SQL 起点，尝试返回清理后的原始文本（可能包含非 SQL）。
    """
    if not raw_output:
        return ""

    # 1. 初步清理
    cleaned_output = raw_output.strip()
    if cleaned_output.startswith("```sql"):
        cleaned_output = cleaned_output[len("```sql"):].strip()
    elif cleaned_output.startswith("```"):
         cleaned_output = cleaned_output[len("```"):].strip()
    if cleaned_output.endswith("```"):
        cleaned_output = cleaned_output[:-len("```")].strip()

    # 2. 查找 SQL 起点
    sql_keywords = ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE"]
    start_index = -1
    normalized_output_upper = cleaned_output.upper() # 忽略大小写查找关键字
    for keyword in sql_keywords:
        idx = normalized_output_upper.find(keyword)
        if idx != -1:
            if start_index == -1 or idx < start_index:
                start_index = idx
            # break # 找到第一个就停止？不，应该找最早出现的那个

    if start_index == -1:
        # 未找到明确的 SQL 关键字起点，可能模型只返回了部分片段或非标准 SQL
        # 尝试一个更宽松的策略：查找第一个分号？或者直接返回清理后的结果？
        # 为了安全，我们先返回清理后的结果，因为不确定内容是什么。
        print("警告: 未在模型输出中找到明确的 SQL 关键字起点。")
        # 查找第一个分号作为备选结束点
        first_semicolon = cleaned_output.find(';')
        if first_semicolon != -1:
             # 取到第一个分号（包含）
             maybe_sql = cleaned_output[:first_semicolon + 1]
             print("警告: 使用第一个分号作为 SQL 结束标记。")
             return maybe_sql.strip()
        else:
             # 如果连分号都没有，返回清理后的原始内容
             print("警告: 无法可靠提取 SQL，返回初步清理后的原始输出。")
             return cleaned_output # 返回清理后的原始输出

    # 3. 从 SQL 起点开始查找停止标记
    sql_part_candidate = cleaned_output[start_index:]
    end_index = -1

    for marker in stop_markers:
        idx = sql_part_candidate.find(marker)
        if idx != -1:
            # 如果找到 marker，记录其在 *sql_part_candidate* 中的位置
            if end_index == -1 or idx < end_index:
                end_index = idx

    # 4. 截取 SQL
    if end_index != -1:
        # 取从 SQL 起点到第一个停止标记之前的部分
        final_sql = sql_part_candidate[:end_index]
    else:
        # 未找到停止标记，认为从起点开始的全部都是 SQL
        final_sql = sql_part_candidate

    # 5. 最终清理
    return final_sql.strip()

# --- 13. 核心 SQL 生成逻辑 ---

# --- 非流式生成函数 (应用后处理) ---
def generate_sql_sync(system_message: str, schema: str, question: str, temp: float, top_p_val: float, max_new_toks: int) -> tuple[str, int, int]:
    """同步生成 SQL，返回提取后的 SQL 及 token 计数。"""
    global model, tokenizer, DEVICE, PROMPT_TEMPLATE
    if not model_ready: raise RuntimeError("模型未准备好")
    try: prompt = PROMPT_TEMPLATE.format(system_message=system_message, schema=schema, question=question)
    except KeyError as e: print(f"错误: Prompt 模板占位符 {e}"); prompt = f"{system_message}\n{schema}\n{question}\nSQL:"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    prompt_tokens = inputs['input_ids'].shape[1]
    try: target_device = getattr(model, 'device', DEVICE); inputs = inputs.to(target_device)
    except Exception as e_move:
        print(f"警告: 移输入到 {target_device} 失败, 尝试 {DEVICE}. Err: {e_move}")
        try: inputs = inputs.to(DEVICE)
        except Exception as e_inner: raise RuntimeError(f"无法移动输入到设备 {DEVICE}") from e_inner

    raw_output_text, completion_tokens = "", 0
    try:
        with torch.no_grad(): outputs = model.generate(**inputs, max_new_tokens=max_new_toks, do_sample=temp > 0, temperature=temp if temp > 0 else 1.0, top_p=top_p_val, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        output_ids = outputs[0][inputs['input_ids'].shape[1]:]; completion_tokens = len(output_ids)
        raw_output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        print(f"模型原始输出 (前 100 字符): {raw_output_text[:100]}...")
    except Exception as e_gen: print(f"!!! 模型生成错误: {e_gen} !!!"); raw_output_text = f"生成 SQL 出错: {e_gen}"; completion_tokens = 0

    # --- 应用后处理 ---
    extracted_sql = extract_sql(raw_output_text)
    print(f"提取后的 SQL (前 100 字符): {extracted_sql[:100]}...")

    # 返回提取后的 SQL 和原始 token 计数
    return extracted_sql, prompt_tokens, completion_tokens

# --- 流式生成函数 (应用后处理，只流式传输提取出的 SQL 部分) ---
async def generate_sql_stream(
    system_message: str, schema: str, question: str,
    temp: float, top_p_val: float, max_new_toks: int
) -> AsyncGenerator[str, None]:
    """异步生成器，流式生成 SQL，并在流中进行后处理，只发送提取出的 SQL 内容。"""
    global model, tokenizer, DEVICE, PROMPT_TEMPLATE, MODEL_ID
    if not model_ready: # 错误处理
        error_detail = "模型未准备好或加载失败。"; print(f"错误: {error_detail}")
        error_chunk = ChatCompletionChunk(model=MODEL_ID, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(role="assistant", content=f"错误: {error_detail}"), finish_reason="error")])
        yield f"data: {error_chunk.model_dump_json()}\n\n"; yield f"data: [DONE]\n\n"; return

    # --- 构建 Prompt ---
    try: prompt = PROMPT_TEMPLATE.format(system_message=system_message, schema=schema, question=question)
    except KeyError as e: print(f"错误: Prompt 模板占位符 {e}"); prompt = f"{system_message}\n{schema}\n{question}\nSQL:"

    # --- 输入处理、Streamer 设置、启动后台生成线程 ---
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        prompt_tokens = inputs['input_ids'].shape[1]
        target_device = getattr(model, 'device', DEVICE)
        try: inputs = inputs.to(target_device)
        except Exception: inputs = inputs.to(DEVICE)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(**inputs, max_new_tokens=max_new_toks, temperature=temp if temp > 0 else 1.0, top_p=top_p_val, do_sample=temp > 0, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, streamer=streamer)
        thread = Thread(target=model.generate, kwargs=generation_kwargs); thread.start()
        print("流式生成线程已启动...")

        # --- 迭代 Streamer 并应用后处理，只发送 SQL ---
        response_id = f"chatcmpl-{uuid.uuid4()}"; start_time = time.time()
        full_raw_output = ""        # 累积模型的原始输出
        processed_sql_yielded = ""  # 记录已经发送给客户端的 SQL 部分
        stop_yielding = False       # 标记是否应停止发送内容
        finish_reason = "stop"      # 默认结束原因

        # 发送第一个含角色的块
        first_chunk = ChatCompletionChunk(id=response_id, model=MODEL_ID, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(role="assistant"), finish_reason=None)])
        yield f"data: {first_chunk.model_dump_json()}\n\n"

        # 使用同步 for 循环迭代 streamer
        try:
            for raw_chunk in streamer: # 迭代原始输出块
                if not raw_chunk: continue # 跳过空块

                full_raw_output += raw_chunk # 累积原始输出

                if not stop_yielding:
                    # 对当前累积的全部原始输出进行 SQL 提取
                    current_extracted_sql = extract_sql(full_raw_output)

                    # 检查提取出的 SQL 是否比已发送的部分更长
                    if len(current_extracted_sql) > len(processed_sql_yielded):
                        # 计算新增加的 SQL 部分
                        new_sql_part = current_extracted_sql[len(processed_sql_yielded):]
                        if new_sql_part:
                            # 创建并发送包含新 SQL 部分的数据块
                            chunk = ChatCompletionChunk(
                                id=response_id, model=MODEL_ID,
                                choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(content=new_sql_part), finish_reason=None)]
                            )
                            yield f"data: {chunk.model_dump_json()}\n\n"
                            processed_sql_yielded = current_extracted_sql # 更新已发送 SQL 的记录
                    else:
                        # 如果提取出的 SQL 没有增长，但原始输出仍在增加，
                        # 说明模型开始输出非 SQL 内容了，停止发送内容。
                        if len(full_raw_output) > len(processed_sql_yielded): # 检查原始输出是否仍在增长
                             print("停止流式发送：提取的 SQL 未增长，可能已开始输出额外内容。")
                             stop_yielding = True
                             # 可以考虑在这里就设置 finish_reason = "stop"

                    # 也可以添加一个硬停止，如果原始输出包含 stop marker
                    if "###" in raw_chunk and not stop_yielding: # 更精确地检查新块
                         print(f"停止流式发送：检测到停止标记 '###'。")
                         stop_yielding = True
                         # 确保之前的部分已发送完毕 (理论上上面的逻辑已处理)
        except Exception as e_iter:
             print(f"!!! 迭代 streamer 时出错: {e_iter} !!!"); finish_reason = "error"
             try:
                 error_chunk = ChatCompletionChunk(id=response_id, model=MODEL_ID, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(content=f"\n迭代错误: {e_iter}"), finish_reason="error")])
                 yield f"data: {error_chunk.model_dump_json()}\n\n"
             except Exception: pass

        # --- 确保线程结束 ---
        if thread.is_alive(): print("等待后台生成线程结束..."); await to_thread(thread.join); print("后台生成线程已确认结束。")
        else: print("后台生成线程已自行结束。")
        end_time = time.time(); generation_time = end_time - start_time

        # --- 发送结束块 (finish_reason 可能已更新) ---
        final_chunk = ChatCompletionChunk(id=response_id, model=MODEL_ID, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(), finish_reason=finish_reason)])
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield f"data: [DONE]\n\n" # SSE 结束标志

        # --- 日志记录 (基于原始输出计算 token) ---
        completion_tokens = len(tokenizer.encode(full_raw_output, add_special_tokens=False)) if full_raw_output else 0
        print(f"流式生成处理完毕。总耗时: {generation_time:.2f} 秒。")
        if generation_time > 0 and completion_tokens > 0:
            tokens_per_sec = completion_tokens / generation_time
            print(f"生成速度 (基于原始输出估算): {tokens_per_sec:.2f} tokens/sec。 Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")
            print(f"最终发送给客户端的 SQL 长度: {len(processed_sql_yielded)}") # 打印实际发送长度

    except Exception as e_stream_outer: # 捕获外层错误
        print(f"!!! 流式生成函数顶层发生错误: {e_stream_outer} !!!")
        try:
            error_chunk = ChatCompletionChunk(model=MODEL_ID, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(content=f"\n服务器内部流式错误: {e_stream_outer}"), finish_reason="error")])
            yield f"data: {error_chunk.model_dump_json()}\n\n"; yield f"data: [DONE]\n\n"
        except Exception: pass
        if 'thread' in locals() and thread.is_alive(): await to_thread(thread.join)


# --- 14. API 端点 ( 根据 stream 调用不同逻辑) ---
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, api_key: str = Security(verify_api_key)):
    if not model_ready: raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="模型未准备好。")
    # ... (提取 schema, question) ...
    if not request.messages: raise HTTPException(status_code=400, detail="'messages' 不能为空。")
    last_user_message = next((msg for msg in reversed(request.messages) if msg.role == "user"), None)
    if not last_user_message: raise HTTPException(status_code=400, detail="未找到 'user' 消息。")
    question = last_user_message.content
    db_schema = request.db_schema
    if db_schema: print("使用请求中的 db_schema。")
    else:
        db_schema = DEFAULT_DB_SCHEMA
        if db_schema: print("使用配置文件中的默认 db_schema。")
        else: raise HTTPException(status_code=400, detail="未提供数据库 Schema。")
    system_message_content = next((msg.content for msg in request.messages if msg.role == "system"), None)
    if not system_message_content: system_message_content = DEFAULT_SYSTEM_MESSAGE_TEMPLATE; print("使用默认 system message。")
    else: print("使用请求中的 system message。")
    temperature = request.temperature if request.temperature is not None else DEFAULT_TEMPERATURE
    max_new_tokens = request.max_tokens if request.max_tokens is not None else DEFAULT_MAX_NEW_TOKENS
    top_p = request.top_p if request.top_p is not None else DEFAULT_TOP_P

    # --- 根据 stream 参数调用不同处理逻辑 ---
    if request.stream:
        # --- 处理流式请求 ---
        print(f"开始处理流式请求 (问题: '{question[:50]}...')")
        return StreamingResponse(
            generate_sql_stream( # 调用流式生成函数 (内部已含后处理)
                system_message=system_message_content, schema=db_schema, question=question,
                temp=temperature, top_p_val=top_p, max_new_toks=max_new_tokens
            ),
            media_type="text/event-stream" # SSE MIME 类型
        )
    else:
        # --- 处理非流式请求 ---
        print(f"开始处理非流式请求 (问题: '{question[:50]}...')")
        start_gen_time = time.time()
        try:
            # generate_sql_sync 现在返回提取后的 SQL
            extracted_sql, prompt_tokens, completion_tokens = await run_in_threadpool(
                generate_sql_sync, system_message=system_message_content, schema=db_schema, question=question,
                temp=temperature, top_p_val=top_p, max_new_toks=max_new_tokens
            )
        except RuntimeError as e: raise HTTPException(status_code=500, detail=f"SQL 生成失败: {e}")
        except Exception as e: raise HTTPException(status_code=500, detail=f"生成过程中意外错误: {e}")
        end_gen_time = time.time(); gen_duration = end_gen_time - start_gen_time
        print(f"非流式生成线程执行完毕。耗时: {gen_duration:.2f} 秒。")
        # 日志记录 
        if gen_duration > 0 and completion_tokens > 0:
             tokens_per_sec = completion_tokens / gen_duration
             print(f"生成速度 (基于原始输出): {tokens_per_sec:.2f} tokens/sec。 Prompt: {prompt_tokens}, Completion: {completion_tokens}")

        # 构建非流式响应 (使用提取后的 SQL)
        response_message = ChatMessage(role="assistant", content=extracted_sql) # <-- 使用提取后的 SQL
        choice = Choice(index=0, message=response_message, finish_reason="stop")
        # Usage token 数仍基于原始生成
        usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens)
        response = ChatCompletionResponse(model=MODEL_ID, choices=[choice], usage=usage)
        try: print(f"\n--- 准备返回非流式响应 ---\n{response.model_dump_json(indent=2)}\n--------------------\n")
        except Exception as e_dump: print(f"警告: 序列化响应出错: {e_dump}")
        return response

# --- 15. 健康检查端点 ---
@app.get("/health", summary="健康检查", tags=["管理"])
async def health_check():
    if model_ready: return {"status": "ok", "model_ready": True, "device": DEVICE}
    else: raise HTTPException(status_code=HTTP_503_SERVICE_UNAVAILABLE, detail={"status": "error", "model_ready": False, "message": "模型加载失败或未完成。"})

# --- 16. 启动服务器的主入口 ---
if __name__ == "__main__":
    import uvicorn
    print(f"\n准备启动 Uvicorn 服务器，监听地址: http://{API_HOST}:{API_PORT}")
    uvicorn.run("api_server:app", host=API_HOST, port=API_PORT, reload=False)