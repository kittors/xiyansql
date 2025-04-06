
## 如何适配不同机器:

- M1 Pro MacBook:
编辑 .env 文件，设置 DEVICE_PREFERENCE="mps" (或保持 "auto")。
设置 MODEL_DTYPE="float16" (通常最佳)。
按照上述步骤创建、激活环境并安装依赖、运行脚本。
- 带 4090 的机器 (Linux/Windows):
确保已安装 NVIDIA 驱动和 CUDA Toolkit (PyTorch 安装时会自动下载所需的 CUDA runtime，但驱动需要预先装好)。

编辑 .env 文件，设置 DEVICE_PREFERENCE="cuda" (或保持 "auto")。
设置 MODEL_DTYPE="float16" 或尝试 "bfloat16" (4090 支持 bf16，可能性能更好)。
按照上述步骤创建、激活环境并安装依赖、运行脚本 (Windows 用户注意激活命令的区别)。


## 创建虚拟环境
在你根目录会创建一个venv文件夹
``` bash
python3 -m venv venv # 创建一个名为 'venv' 的虚拟环境文件夹
```

## 激活虚拟环境

- macOS / Linux (bash/zsh):
``` bash
source venv/bin/activate
```

- Windows (cmd.exe):
``` bash 
venv\Scripts\activate.bat
```

- Windows (PowerShell):
``` bash
venv\Scripts\Activate.ps1
```

## 安装依赖
``` bash
pip install -r requirements.txt
```

## 运行脚本
``` bash
# 测试脚本
python run_xiyansql.py
```

``` bash
# 使用配置中的提示词的版本
python api_config_server.py

```

``` bash 
# 不使用任何配置中的提示词的版本，提示词 可以在类似dify的平台配置
python api_server.py

```

## 系统提示词案例

``` Plaintext
你是一个强大的 Text-to-SQL 助手。
给定以下 SQL 表结构，请根据用户的自然语言问题，生成相应的 SQL 查询语句。
**生成的查询语句应使用 `SELECT *` 来获取匹配行的所有字段。**
**特别注意:** 当用户询问 "员工人数" 或类似的概括性人力指标时，这通常可能指代多个具体因子。
**请优先在 `factor_name` 字段中查找并同时查询以下几个关键指标：'员工期初在岗人数', '员工期末在岗人数', '员工平均在岗人数'。**
使用 `IN` 操作符将这些相关的 `factor_name` 包含在查询中，以便返回所有相关的员工数量信息。
确保查询条件（如公司名称、年份、月份）准确无误。

```

## 退出虚拟环境
```bash
deactivate
```