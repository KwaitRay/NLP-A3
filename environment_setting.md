# 环境配置 — COMP90042 Assignment 3

> AutoDL / 本地 conda 环境一次性初始化指南。配套 `requirements.txt`。

## 目录

1. [完整安装流程](#1-完整安装流程)
2. [关键依赖说明](#2-关键依赖说明)
3. [PyTorch 单独安装](#3-pytorch-单独安装)
4. [GPU 显存与配置自检](#4-gpu-显存与配置自检)
5. [常见问题排查](#5-常见问题排查)
6. [快速重建环境（一键脚本）](#6-快速重建环境一键脚本)

---

## 1. 完整安装流程

在 AutoDL 命令行（已 SSH 进实例）或本地 Linux/Mac 终端执行：

```bash
# ── 1. 创建并激活 conda 环境 ──
conda create -n nlp_a3 python=3.10 -y
conda activate nlp_a3

# ── 2. 装 pip 依赖（用清华镜像加速，国内必备）──
cd /root/autodl-tmp/Assignment3            # ← AutoDL；本地改成你的项目路径
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ── 3. 装 spaCy 模型 + NLTK 数据 ──
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# ── 4. 把这个 env 注册成 Jupyter kernel ──
python -m ipykernel install --user --name nlp_a3 --display-name "Python (nlp_a3)"
```

完成后在 JupyterLab 右上角切换 kernel 到 **Python (nlp_a3)**，
notebook 里的 setup-2 cell（pip install）就可以跳过了。

---

## 2. 关键依赖说明

| 包 | 版本约束 | 用途 |
|---|---|---|
| `ms-swift` | ≥ 3.0 | Qwen3.5 SFT + DPO 训练框架（ModelScope 官方） |
| `modelscope` | ≥ 1.20 | 模型下载（国内直连，无需 HF token） |
| `transformers` | ≥ 4.46 | Qwen3.5 / bge-m3 / bge-reranker 推理 |
| `peft` | ≥ 0.13 | LoRA adapter 加载 |
| `trl` | ≥ 0.12 | DPO 训练（被 ms-swift 间接调用） |
| `bitsandbytes` | ≥ 0.44 | QLoRA 4-bit 量化 |
| `accelerate` | ≥ 1.0 | 分布式 / 混合精度训练辅助 |
| `sentence-transformers` | ≥ 3.0 | bge-m3 编码 + cross-encoder rerank |
| `bm25s` | ≥ 0.2 | 稀疏检索（**不要用 `rank_bm25`，1.2 M 段会 OOM**） |
| `pystemmer` | ≥ 2.2 | bm25s 英文词干 |
| `faiss-cpu` | ≥ 1.7.4 | 稠密索引（1.2 M × 1024d，CPU 查询 < 1 ms） |
| `spacy` | ≥ 3.7 | NER 实体抽取（rule reorder 用） |
| `nltk` | ≥ 3.9 | WordNet 同义词扩展 |
| `numpy` | < 2.0 | `transformers` / `bitsandbytes` 暂未迁移 numpy 2 ABI |

---

## 3. PyTorch 单独安装

`requirements.txt` **不锁 torch 版本**，原因：

- AutoDL 主流镜像（PyTorch 2.x + CUDA 12）已预装匹配版本，写在 requirements 会被覆盖
- 不同 GPU 需要不同 CUDA build，强锁会出错

### AutoDL 默认镜像
已预装，**无需操作**。验证：
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望：2.x.x True
```

### 本地 / base 镜像（无 torch）

```bash
# CUDA 12.1
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# CUDA 11.8
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# CPU only（仅做 Stage 0 数据预处理，无法训练 / 推理）
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### `bitsandbytes` 与 CUDA 版本

默认 wheel 是 CUDA 12 build。如果实例是 CUDA 11.x：
```bash
pip uninstall -y bitsandbytes
pip install bitsandbytes-cuda118       # 或 -cuda117 / -cuda116
```

---

## 4. GPU 显存与配置自检

激活 env 后跑这一行确认：

```bash
python -c "
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU : {p.name}')
    print(f'VRAM: {p.total_memory // 2**30} GB')
else:
    print('No GPU')
"
```

显存对应训练超参（notebook 的 setup-gpu cell 会自动设）：

| 显存 | GPU 例 | SFT 配置 | dense batch |
|---|---|---|---|
| ≥ 40 GB | A100 / A800 | BS=4 / GA=4 / bf16 | 128 |
| 24 GB | RTX 3090 / 4090 | BS=2 / GA=8 / QLoRA 4-bit | 64 |
| ≤ 16 GB | T4 / RTX 3080 | BS=1 / GA=16 / QLoRA 4-bit | 32 |

有效 batch size 始终保持 16（保证 SFT 超参跨 GPU 等价）。

---

## 5. 常见问题排查

### Q1：`pip install` 卡在某个包不动
→ 切清华镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`
其他可选镜像：阿里云 `https://mirrors.aliyun.com/pypi/simple/`、豆瓣 `https://pypi.doubanio.com/simple/`

### Q2：`ImportError: numpy.core.multiarray failed to import`
→ numpy 2.x 与 transformers 不兼容，降级：
```bash
pip install "numpy<2.0" --force-reinstall
```

### Q3：`bitsandbytes` 报 `CUDA Setup failed`
→ 检查 CUDA：
```bash
python -m bitsandbytes
```
按它提示装对应 CUDA 版本的 wheel（见 §3）。

### Q4：`faiss-cpu` 安装失败 / import 报错
→ Linux 上偶尔 manylinux 兼容问题，改用 conda：
```bash
pip uninstall -y faiss-cpu
conda install -c conda-forge faiss-cpu -y
```

### Q5：`OSError: [E050] Can't find model 'en_core_web_sm'`
→ spaCy 模型未装：
```bash
python -m spacy download en_core_web_sm
```

### Q5b：`python -m spacy download en_core_web_sm` 卡在 `Connection to github.com timed out` 或 `403 Forbidden`
→ AutoDL 国内网络访问 GitHub 不稳定，spaCy 模型托管在 GitHub Releases。开 AutoDL 学术加速：
```bash
source /etc/network_turbo
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```
加速代理仅当前 shell 有效，配完环境就可以关掉（`unset http_proxy https_proxy`）。
pip 装 pypi 包不需要开（已用国内镜像更快）。

### Q6：`LookupError: Resource wordnet not found`
→ NLTK 数据未下：
```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Q7：JupyterLab 看不到 nlp_a3 kernel
→ 重新注册：
```bash
conda activate nlp_a3
python -m ipykernel install --user --name nlp_a3 --display-name "Python (nlp_a3)"
```
然后 JupyterLab 顶部菜单 Kernel → Change Kernel → 选 nlp_a3。

### Q8：`ValueError: ... upgrade torch to at least v2.6 ... CVE-2025-32434`  /  `OSError: ... does not appear to have a file named model.safetensors`
→ 新 transformers (≥4.50) 安全检查 + 旧 torch (<2.6) + 模型只有 `.bin` 文件三者叠加。
**`bge-m3` 在 HF Hub 上只有 `pytorch_model.bin`，没有 safetensors**，所以 `use_safetensors=True` 不可行。

**唯一可靠修法**：降级 transformers 到 CVE 检查之前的版本：
```bash
pip install "transformers>=4.46,<4.50" -i https://pypi.tuna.tsinghua.edu.cn/simple
```
重启 kernel 后即可加载 bge-m3 的 `.bin` 权重（`requirements.txt` 已经钉死 `<4.50.0`，
但若是用 conda env 之外的 site-packages 装的可能仍是新版，跑上面命令强制覆盖）。

### Q9：模型下载特别慢 / 失败
- ModelScope 在国内直连，**优先**用 ModelScope 而非 HuggingFace：
  ```python
  from modelscope import snapshot_download
  snapshot_download("Qwen/Qwen3.5-4B", cache_dir=...)
  ```
- 如必须用 HF，notebook setup-1 已设 `HF_ENDPOINT=https://hf-mirror.com`

---

## 6. 快速重建环境（一键脚本）

把下面保存成 `setup_env.sh`，`bash setup_env.sh` 执行：

```bash
#!/usr/bin/env bash
set -e

ENV_NAME="nlp_a3"
PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/Assignment3}"
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "==> creating conda env: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.10 -y

# 让 conda 在脚本中可用
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "==> installing pip deps from requirements.txt"
cd "$PROJECT_DIR"
pip install -r requirements.txt -i "$MIRROR"

echo "==> downloading spaCy model + NLTK data"
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

echo "==> registering Jupyter kernel"
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

echo "==> done. activate with: conda activate $ENV_NAME"
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

实例释放后重建只需：
```bash
cd /root/autodl-tmp/Assignment3
bash setup_env.sh
```

---

## 关联文件

- `requirements.txt` — pip 依赖清单
- `notebooks/notebook_autodl.ipynb` — AutoDL 适配版主 notebook
- `design.md` — 系统设计文档（含硬件预算、训练超参依据）
- `debug_log.md` — 历史问题排查记录
