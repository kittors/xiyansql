
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
python api_server.py

```

## 退出虚拟环境
```bash
deactivate
```