# Colab 联调日志 — Assignment 3 Hybrid RAG Fact-Checker

## 会话元信息

- **日期**: 2026-05-10
- **环境**: Google Colab (T4 GPU, 12GB RAM, free tier) + Google Drive 挂载
- **项目**: Climate fact-checking Hybrid RAG pipeline (`COMP90042`)
- **Drive 路径**: `/content/drive/MyDrive/NLP_Assignment3/Assignment3`
- **本地路径**: `D:\学习\研究生\graduate 26s1\90042 NLP\Assignment\Assignment3`
- **运行流程**: notebook.ipynb 顺序执行 setup → Stage 0 数据预处理 → Stage 2 检索

---

## 问题时间线

### 问题 1 — GPU 未检测到（伪问题）

**现象**
```
No GPU detected (Stage 0 prep is fine without one).
```
位置: `cell-setup-3`。

**根因**
Colab 默认分配 CPU runtime，未切到 GPU。

**判断**
不是 error，是预期 print 输出。Stage 0（数据预处理）不需要 GPU，Stage 2 起的 dense retrieval / SFT 才必须 GPU。

**解决方案**
`Runtime` → `Change runtime type` → `T4 GPU` → `Save`。注意切换 runtime 类型会清空 pip 安装，setup-2 需重跑。

---

### 问题 2 — 路径不匹配，FileNotFoundError

**现象**
```
FileNotFoundError: [Errno 2] No such file or directory:
  '/content/drive/MyDrive/comp90042/data/train-claims.json'
```
位置: `cell-1-load-code` 调用 `load_train()`。

**根因**
- `src/paths.py:9` 硬编码 Colab 路径为 `/content/drive/MyDrive/comp90042`
- `cell-setup-1` 同样硬编码错误路径
- 用户实际项目在 `/content/drive/MyDrive/NLP_Assignment3/Assignment3`

**解决方案**

1. `src/paths.py` 增加 `PROJECT_ROOT` 环境变量优先级:
   ```python
   def _detect_root() -> Path:
       env_root = os.environ.get("PROJECT_ROOT")
       if env_root:
           return Path(env_root)
       if os.environ.get("IS_COLAB") == "1":
           return Path("/content/drive/MyDrive/NLP_Assignment3/Assignment3")
       here = Path(__file__).resolve().parent
       return here.parent
   ```

2. `cell-setup-1`:
   - `PROJECT_ROOT` 字面值改对
   - 新增 `os.chdir(PROJECT_ROOT)`
   - 新增 `os.environ["PROJECT_ROOT"] = PROJECT_ROOT`

**涉及文件**: `src/paths.py`、`notebooks/notebook.ipynb` (cell-setup-1)

---

### 问题 3 — Dense Retrieval CUDA OOM

**现象**
```
OutOfMemoryError: CUDA out of memory. Tried to allocate 442.00 MiB.
GPU 0 has a total capacity of 14.56 GiB of which 401.81 MiB is free.
```
位置: `cell-2-dense-code` 第一个 batch 即触发。

**根因**（三重叠加）
1. `batch_size=128` 对 bge-m3 太激进
2. bge-m3 默认 `max_seq_length=8192`；但 EDA 显示 evidence 中位数 18 token、最大 479 token → 激活内存浪费几十倍
3. 全程 fp32（无 fp16）

`DenseRetriever` 当时不暴露 `max_seq_length` 和 `fp16` 参数。

**解决方案** — 改 `src/retrieval/dense.py`:
- 构造函数新增 `max_seq_length=256` 和 `fp16=True`
- `_load_model()` 应用 `model.max_seq_length = self.max_seq_length` 和 `model.half()`
- `build()` 默认 `batch_size: 256 → 32`
- 每个 chunk 编码完 `torch.cuda.empty_cache()`
- `load()` classmethod 同步接受新参数

**显存账单（新版）**

| 项 | 占用 |
|---|---|
| 模型 fp16 | ~1.1 GB |
| 单 batch 激活 (bs=32, seq=256) | ~200 MB |
| **峰值 / T4 14.5 GB** | **<5 GB ✓** |

**追加修复** — cell-2-dense-code 的 cache 检测从 `if DENSE_DIR.exists()` 升级为:
```python
if DENSE_DIR.exists() and (DENSE_DIR / "faiss.index").exists():
```
原因: OOM 时 `dense_index/` 和 `chunks/` 空目录已被创建，旧检测会误判 cache 就绪。

**涉及文件**: `src/retrieval/dense.py`、`notebooks/notebook.ipynb` (cell-2-dense-code)

---

### 问题 4 — HF_TOKEN Warning

**现象**
```
UserWarning: The secret `HF_TOKEN` does not exist in your Colab secrets.
Warning: You are sending unauthenticated requests to the HF Hub.
```

**根因**
未在 Colab Secrets 设 `HF_TOKEN`，HuggingFace Hub 走未认证下载。

**判断**
仅 warning，不阻塞公开模型下载。bge-m3 仍能正常拉取。仅影响下载速率上限。

**解决方案**（可选）
1. HF Settings → 创建 read token
2. Colab 左侧栏 🔑 → `Add new secret` → Name `HF_TOKEN`、Value 粘贴 token、启用 `Notebook access`
3. 重启 runtime 让其生效

不消除也行。

---

### 问题 5 — Drive 中途断开（编码阶段）

**现象**
Dense build 跑到 ~5 个 chunk 时，Drive 连接断开，但 Python kernel 状态还在。

**根因**（按概率排序）
1. **Drive API 限流（最可能）**: 每 ~10s 写一个 26MB `.npy` 到 Drive，FUSE 突发写入易触发 Google Drive 速率限制
2. 网络 / FUSE 抖动
3. 系统资源压力（RAM 接近上限影响 FUSE driver）

**判断与恢复**
- chunks 落盘的不会丢，`if chunk_path.exists(): continue` 续编码
- 重挂 Drive: `drive.mount("/content/drive", force_remount=True)` 不会再弹 OAuth（token 内存缓存）
- 验证 chunk 数 → 直接重跑 cell 2.2 续编码

**预防策略（建议未实现）**
编码先写 `/content/dense_index/`（本地 SSD，飞快、无限流）→ 全部完成后一次性 `cp` 到 Drive。

---

### 问题 6 — ModelScope 模型 ID 404

**现象**
```
modelscope - WARNING - Repo Qwen/Qwen3.5-4B-Instruct not exists on https://www.modelscope.cn
modelscope - ERROR - Repo Qwen/Qwen3.5-4B-Instruct not exists on either ...cn or ...ai
HTTPError: <Response [404]>
```
位置: `cell-2-sft-download` 调 `snapshot_download("Qwen/Qwen3.5-4B-Instruct", ...)`。

**根因**
ModelScope 上 Qwen3.5-4B 系列**只有 base 版本** (`Qwen/Qwen3.5-4B`)，**没有 `-Instruct` 后缀**。Notebook 原始硬编码错了 ID。

实际可用页面: https://www.modelscope.cn/models/Qwen/Qwen3.5-4B/files

**附带问题** — `cell-2-sft-train` 还有两处错误:
- `--model_type qwen3_5_vl`: 这个 model_type 不存在；ms-swift 会自动从 `config.json` 推断，无需指定
- `--freeze_vit true`: Qwen3.5-4B 是纯文本模型，没有 ViT

**解决方案**
1. `cell-2-sft-download` 改 ID: `"Qwen/Qwen3.5-4B-Instruct" → "Qwen/Qwen3.5-4B"`
2. `cell-2-sft-train` 删除 `--model_type qwen3_5_vl` 和 `--freeze_vit true`
3. `cell-2-sft` 标题描述同步: 删 "multimodal; ViT frozen" 字样
4. `cell-1` (README) 系统概览中的 Stage 3 描述同步

**评测含义**
Base 模型未经过 instruction tuning → Track 1/2 (zero-shot) 效果会显著弱于 -Instruct 版本。但**这反而让 SFT 的增益更明显**（Track 3 vs 2 的 delta 大），对论证 SFT 价值有利。

**涉及文件**: `notebooks/notebook.ipynb` (cell-2-sft-download / cell-2-sft-train / cell-2-sft / cell-1)

---

### 问题 7 — 4-Track 评测架构（功能新增）

**需求**
论文需要对比 Base / Base+RAG / SFT / DPO 四种方式的边际收益，定量说明每个组件的贡献。

**设计**

| Track | 检索 | 模型 | 解码 | 实现 |
|---|---|---|---|---|
| 1. Base (claim only) | 无 | base Qwen | greedy | 新增 `NoRagInferer` |
| 2. Base + RAG | BM25+dense+rerank | base Qwen | greedy | 复用 `ZeroShotInferer` |
| 3. + SFT | 同 Track 2 | SFT-LoRA Qwen | greedy | 复用 `ZeroShotInferer` |
| 4. + DPO | 同 Track 2 | SFT+DPO LoRA | self-consistency | 复用 `ModelInferer` |

**关键设计决策**
- **共用 retriever**: Track 2/3/4 都传 `pipeline_zero_shot`，保证检索条件一致 → SFT/DPO 增益可归因
- **共用 base_model**: Track 1/2 共用一份 4-bit 量化模型，省 VRAM
- **SFT/DPO 用 LoRA adapter**: `PeftModel.from_pretrained(base, ckpt)`，不重复加载基座
- **SC 仅 Track 4**: 其他 track 用 greedy，否则 SFT→DPO delta 无法干净归因到 preference alignment
- **Track 1 evidence stub**: `["evidence-0"]` 让 eval.py 不抱怨，F=0 是诚实结果，**只看 Label Acc**
- **缺 checkpoint 不报错**: `run_4tracks` 自动 skip missing inferer，可以只跑 Track 1/2 先验证 pipeline

**新增文件**
- `src/eval_compare.py`: `evaluate_track`、`render_compare_table`、`run_4tracks` + `TrackResult` dataclass

**新增类 / 函数**
- `src/inference.py::NoRagInferer`: claim → base model → label，evidence stub
- `src/prompt.py::NO_RAG_SYSTEM_PROMPT`、`build_no_rag_query`: Track 1 专用 prompt（无 evidence list、无 citation 规则）

**新增 notebook section**
- `3.5` markdown: 设计说明 + 期望解读
- `3.5a`: 加载 base model（QLoRA 4-bit）
- `3.5b`: 检测 SFT/DPO checkpoint，缺失就跳过
- `3.5c`: 装配 inferers + `run_4tracks` + 打印对比表

**输出物**
```
outputs/predictions/track1_base.json
outputs/predictions/track2_base_rag.json
outputs/predictions/track3_sft.json
outputs/predictions/track4_dpo.json
outputs/eval_compare.md   # markdown 对比表带 Δ 列
```

对比表格式:
```
| # | Track | Label Acc | Retrieval F | Harmonic | Δ Harmonic vs prev |
|---|-------|-----------|-------------|----------|--------------------|
| 1 | track1_base     | 0.42 | 0.00 | 0.00 | — |
| 2 | track2_base_rag | 0.48 | 0.21 | 0.29 | +0.29 |
| 3 | track3_sft      | 0.61 | 0.34 | 0.44 | +0.15 |
| 4 | track4_dpo      | 0.65 | 0.36 | 0.46 | +0.02 |
```

**涉及文件**: `src/prompt.py`、`src/inference.py`、`src/eval_compare.py` (新建)、`notebooks/notebook.ipynb` (3.5 系列 cell + cell-1 README 更新)

---

### 问题 8 — Runtime 整体 kill（RAM OOM）

**现象**
所有 189 chunk 跑完后，runtime 整个断开，所有 cell 必须重跑（变量全丢）。

**根因**
`dense.py` build 收尾代码 RAM 爆炸:
```python
all_emb = np.concatenate(
    [np.load(chunks_dir / f"emb_{ci:05d}.npy") for ci in range(n_chunks)],
    axis=0,
)
index = faiss.IndexFlatIP(self._dim)
index.add(all_emb)
```

**RAM 账单（旧版）**

| 项 | 内存 |
|---|---|
| 189 chunks 全 load 进 list | ~5 GB |
| `np.concatenate` 输出（同时存在）| ~5 GB |
| `faiss.add` 内部拷贝 | ~5 GB |
| `evidence` dict | ~1.5 GB |
| Python + bm25 + 其他 | ~2 GB |
| **峰值** | **~15 GB ❌** |

Colab 免费 12 GB RAM → OOM kill → runtime 整个被回收 → 表象为「Drive 断开」。

**状态验证命令**
```python
from pathlib import Path
import os
DENSE_DIR = Path("/content/drive/MyDrive/NLP_Assignment3/Assignment3/outputs/dense_index")
print("faiss.index:", (DENSE_DIR / "faiss.index").exists())
print("meta.json :", (DENSE_DIR / "meta.json").exists())
print("ev_ids.txt:", (DENSE_DIR / "ev_ids.txt").exists())
chunks = sorted((DENSE_DIR/"chunks").glob("emb_*.npy"))
print(f"chunks: {len(chunks)}/189")
```
确认结果: `faiss/meta/ev_ids` 全 False，chunks 189/189。

**解决方案** — 改 `src/retrieval/dense.py` 的 `build()`:
1. **跳过模型加载**: 当所有 chunks 都已存在时，不 load bge-m3
2. **流式 faiss.add**: 一次只 load 一个 chunk，add 完立即 `del`
3. **跳过 texts list 构造**: 仅在需编码时才物化 `texts`
4. 每 20 chunk 调一次 `gc.collect()`
5. 编码完 `del texts; gc.collect()`

**RAM 账单（新版）**

| 项 | 内存 |
|---|---|
| `evidence` dict | ~1.5 GB |
| 单 chunk + faiss 累积 | 0.026 + 5 GB |
| Python + 其他 | ~0.5 GB |
| **峰值** | **~7 GB ✓** |

**涉及文件**: `src/retrieval/dense.py` build 函数

---

## 复用经验

### 1. Colab 路径管理
- `src/paths.py` 用 env var 优先级: `PROJECT_ROOT` > `IS_COLAB` > `__file__` 推断
- `cell-setup-1` 同时设 Python 变量 + env var + `os.chdir` + `sys.path`

### 2. Restart Runtime vs Restart Session
| 操作 | pip 包 | 内存变量 | 用途 |
|---|---|---|---|
| Restart session / kernel | 保留 | 丢失 | 装新版 numpy/torch/transformers 后必须 |
| Disconnect and delete runtime | 全丢 | 全丢 | 切 GPU 类型；OOM 残留显存清理 |
| 切换 runtime 类型（CPU↔GPU）| 全丢 | 全丢 | 自动触发 |

### 3. T4 显存下的 sentence-transformers 编码
- bge-m3 (568M, 1024-d): `batch=32`, `max_seq=256`, `fp16=True` ✓
- 默认参数（`bs=128, max_seq=8192, fp32`）必 OOM
- 关键调用: `model.max_seq_length = N` 和 `model.half()`

### 4. Colab 12 GB RAM 下大向量索引构建
- 禁用 `np.concatenate` 全 chunk 一次性加载
- 流式 `faiss.add`（单 chunk 26MB → 累积 5GB index，无 transient peak）
- 完成编码后 `del texts; gc.collect()` 释放 caller 的 string list

### 5. Drive 持久化的可恢复性
- 编码任务必须 chunked + per-chunk persist
- Cache 检测条件用 `(DENSE_DIR / "faiss.index").exists()`，不要只用 `DENSE_DIR.exists()`
- 中途断开后续编码: `if chunk_path.exists(): continue`

### 6. Drive 频繁写入限流风险
- 每 10s 写 26MB × 189 次有触发 Drive API 限流可能
- 改进方向: 编码先写 `/content/`（本地 SSD），最后一次性 `cp` 到 Drive

### 7. HF_TOKEN warning
- 仅 warning，不阻塞公开模型下载
- 解决: Colab Secrets 设 `HF_TOKEN` + 重启

### 8. 显存 OOM 后的兜底降级顺序
1. `batch_size: 32 → 16`
2. `max_seq_length: 256 → 128`（EDA 中位数 18 token，128 仍覆盖 95%+）
3. `model_name: DEFAULT_MODEL → LIGHT_MODEL`（bge-m3 → bge-small-en-v1.5，384-d，~33M 参数）

---

## 修改文件清单

| 文件 | 改动概要 |
|---|---|
| `src/paths.py` | `_detect_root()` 增加 `PROJECT_ROOT` env var 优先级；Colab fallback 路径纠正 |
| `src/retrieval/dense.py` | 构造函数加 `max_seq_length`/`fp16`；build 默认 `batch_size 32`；流式 `faiss.add`；chunks 全在时跳过模型加载；`gc.collect` 节流 |
| `src/prompt.py` | 新增 `NO_RAG_SYSTEM_PROMPT` + `build_no_rag_query()`（Track 1 用） |
| `src/inference.py` | 新增 `NoRagInferer` 类（Track 1 实现） |
| `src/eval_compare.py` | **新文件**：`evaluate_track`、`render_compare_table`、`run_4tracks` + `TrackResult` dataclass |
| `notebooks/notebook.ipynb` cell-setup-1 | Colab `PROJECT_ROOT` 改对；新增 `os.chdir` + `os.environ["PROJECT_ROOT"]` |
| `notebooks/notebook.ipynb` cell-2-dense-code | 调用对齐新 dense API（max_seq_length/fp16）；cache 检测加 `faiss.index` 二次校验；注释加降级提示 |
| `notebooks/notebook.ipynb` cell-2-sft-download | 模型 ID `Qwen/Qwen3.5-4B-Instruct → Qwen/Qwen3.5-4B`（base 版本）|
| `notebooks/notebook.ipynb` cell-2-sft-train | 删除 `--model_type qwen3_5_vl` 和 `--freeze_vit true`（错误的 VL 配置）|
| `notebooks/notebook.ipynb` cell-2-sft (markdown) | 描述同步：删 "multimodal; ViT frozen" 字样 |
| `notebooks/notebook.ipynb` cell-1 (README) | Stage 3 描述同步；新增 4-track 评测段落 |
| `notebooks/notebook.ipynb` 3.5 系列 cell | 新增 4-track 评测 section（markdown + 3.5a/3.5b/3.5c 三个 code cell）|

---

## 当前进度与待办

### 已完成
- [x] Colab 路径修复（paths.py + setup-1）
- [x] Dense retriever GPU OOM 修复（fp16 + max_seq + batch_size）
- [x] Dense retriever RAM OOM 修复（流式 faiss.add）
- [x] 189 个 dense embedding chunks 全部编码完成（~5 GB on Drive）
- [x] SFT 模型 ID 修复（`Qwen/Qwen3.5-4B`，删 VL 相关参数）
- [x] 4-track 评测架构落地（NoRagInferer + eval_compare.py + 3.5 cell）

### 进行中
- [ ] 跑新版 `cell-2-dense-code` 走 finalize 路径，写出 `faiss.index` / `meta.json` / `ev_ids.txt`（预计 3-5 分钟）

### 未触及
- [ ] Stage 0 (1.2/1.3/1.4) tagging / split / SFT 数据构建在 Colab 跑（产物可本地 dry_run 生成后传 Drive）
- [ ] Stage 2.1 BM25 索引构建
- [ ] Stage 2.3 fusion + rerank pipeline
- [ ] Stage 2.5 SFT (Qwen3.5-4B QLoRA via ms-swift)
- [ ] Stage 2.6 DPO（需 SFT checkpoint + dev_holdout 错样本挖掘）
- [ ] Stage 2.7 Self-consistency inference
- [ ] Stage 3.5 跑 4-track 评测，产出 `outputs/eval_compare.md`

### 待优化（建议）
- [ ] 实现"先写 `/content` 再同步 Drive"机制，规避 Drive 频繁写入限流
- [ ] BM25 索引构建后同样需评估 RAM 峰值，预防 OOM
- [ ] 给 Drive 长连任务加保活机制，规避 idle disconnect

---

## 断点续跑指南 — 重连后的最短恢复路径

Colab 容易因为 Drive 断开 / runtime 抢占 / OOM kill / kernel restart 等原因中断。
触发后**不要无脑全部重跑**，按下面三种情形定位最短路径。

### 情形 A — 仅文件更新（kernel 还活着，变量都在）

**触发场景**: 你在本地改了 `src/*.py` 或 notebook 后传回 Drive 覆盖。

**操作**:
1. 浏览器刷新 notebook 标签页（让 cell 显示新代码）
2. 强制 reload 改过的 src 模块（Python 缓存了旧 import，不会自动看到新版）：
   ```python
   import importlib
   import src.prompt, src.inference, src.eval_compare, src.retrieval.dense
   for m in (src.prompt, src.inference, src.eval_compare, src.retrieval.dense):
       importlib.reload(m)
   print("reloaded")
   ```
3. 只跑修改 / 新增的 cell

**不要**重跑 setup、1.x、2.x。耗时 < 30 秒。

---

### 情形 B — Kernel 还活着但部分变量丢失

**触发场景**: Drive 短暂断开重连、单个 cell 报错被中断、手动 `del` 了某些变量。

**Sanity check**（贴这个，根据输出补缺失部分）:
```python
print("evidence:", "evidence" in dir())
print("dense:", "dense" in dir())
print("bm25:", "bm25" in dir())
print("pipeline_zero_shot:", "pipeline_zero_shot" in dir())
print("MODEL_DIR:", "MODEL_DIR" in dir())
print("base_model:", "base_model" in dir())
```

**按缺失补齐**:

| 缺失变量 | 重跑 cell | 耗时（命中 cache）|
|---|---|---|
| `evidence` | `1.1` | ~30s（解析 174MB JSON）|
| `bm25` | `2.1` | <10s |
| `dense` | `2.2` | <30s（faiss.index 已写好）|
| `pipeline_zero_shot` | `2.3` | ~20s（reranker 模型有 cache）|
| `MODEL_DIR` | `2.5 download` | <5s（cache 命中）|
| `base_model` | `3.5a` | ~30s（model_cache 已下完）|
| `sft_model` / `dpo_model` | `3.5b` | ~5s（adapter 文件小）|

---

### 情形 C — Runtime 完全重启 / 抢占重连

**触发场景**: `Disconnect and delete runtime`、Colab 抢占、RAM OOM kill、切换 GPU 类型。

**按依赖顺序跑**（命中 cache 总耗时 ~5 分钟，不算 4-track inference 本身）:

```
setup-1     (挂 Drive + sys.path)        5s
setup-3     (seed + GPU 检测)            5s
1.1         (load evidence)              30s   ← 后面所有 retriever 都依赖
2.1         (BM25 from cache)            10s
2.2         (dense from cache)           30s
2.3         (pipeline + reranker)        20s
2.5download (cache hit, no re-download)  <5s
3.5a        (load base model 4-bit)      30s
3.5b        (检测 SFT/DPO checkpoint)    <1s
3.5c        (跑 4-track inference)       10-20 min
```

**不需要重跑**:
- `setup-2` (pip install) — 包还在 `/usr/local/lib`，runtime 没回收的话保留
- `1.2 / 1.3 / 1.4` — 产物已落 Drive，后面 cell 直接读文件
- 任何 build 类 cell — 全部 cache 复用

**强制重跑场景**:
- `setup-2` 必须重跑: 切了 runtime 类型（CPU↔GPU）/ 完全 disconnect runtime / 装的包是 numpy 等 Colab 预装包的新版（必须 restart）

---

### 永久化产物 — 永远不要重跑这些

只要 Drive 文件在，下面这些都不需要重新生成。如果不存在才重新构建：

| 路径 | 来源 cell | 大小 |
|---|---|---|
| `outputs/eda/eda_report.md` | 1.1 build_report | 几 KB |
| `outputs/splits/train_split.jsonl` etc. | 1.3 hash split | < 1 MB |
| `outputs/sft_data/sft_train_v1.jsonl` | 1.4 build_dataset | ~5 MB |
| `outputs/bm25_index/` | 2.1 BM25 build | ~200 MB |
| `outputs/dense_index/faiss.index` + `chunks/` | 2.2 dense build | ~5 GB |
| `outputs/model_cache/Qwen3.5-4B/` | 2.5 snapshot_download | ~8 GB |
| `outputs/sft-out/checkpoint-*` | 2.5 train | ~100-500 MB（LoRA adapter）|
| `outputs/dpo-out/checkpoint-*` | 2.6 train | ~100-500 MB |

---

### 决策树速查

```
出问题了 → 跑情形 B 的 sanity check
   │
   ├─ 全 True            → 情形 A，importlib.reload + 跑新 cell
   ├─ 部分 True          → 情形 B，按表补齐
   └─ 全 NameError       → 情形 C，从 setup-1 顺序跑
```

---

## 关键命令速查

### 重新挂载 Drive（同 session 不弹 OAuth）
```python
from google.colab import drive
drive.mount("/content/drive", force_remount=True)
```

### setup-1 重跑（runtime 重启后必须）
```python
import os, sys
PROJECT_ROOT = "/content/drive/MyDrive/NLP_Assignment3/Assignment3"
os.chdir(PROJECT_ROOT)
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
os.environ["IS_COLAB"] = "1"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
```

### 检查 dense_index 状态
```python
from pathlib import Path
DENSE_DIR = Path("/content/drive/MyDrive/NLP_Assignment3/Assignment3/outputs/dense_index")
print("faiss.index:", (DENSE_DIR / "faiss.index").exists())
chunks = sorted((DENSE_DIR/"chunks").glob("emb_*.npy"))
print(f"chunks: {len(chunks)}/189")
```

### 检查 kernel 是否还活着（变量未丢）
```python
print("evidence:", "evidence" in dir(), len(evidence) if "evidence" in dir() else None)
print("dense:", "dense" in dir())
print("bm25:", "bm25" in dir())
```

---

# 会话 2 — 2026-05-10/11 — Qwen3.5 + AutoDL + messages 格式

## 会话元信息

- **日期**: 2026-05-10 / 2026-05-11
- **环境**:
  - Colab T4 (会话起点)
  - **AutoDL Linux 实例 RTX 4080 SUPER 31.5 GB VRAM**（中段切换；驱动错配后重建为 PyTorch 2.5.1+cu124 镜像）
- **范围**: SFT 训练管线打通；推理路径修复；数据格式向 ms-swift 标准 messages 迁移
- **新增参考材料**: `materials/{swift_training,qwen3.5_key_points,训练数据格式}.docx`（来自 ms-swift 官方文档 + CSDN 实战帖）

---

## 问题时间线

### 问题 9 — ms-swift CLI 参数名跨版本改动（连环 3 次）

**现象**（依次出现）
```
ValueError: remaining_argv: ['--train_type', 'lora', '--quantization_bit', '4']
ValueError: remaining_argv: ['--sft_type', 'lora']
```

**根因**

Colab 上的 ms-swift 实际 CLI 路径是 `swift/pipelines/train/sft.py`（非常见的 `swift/llm/`），且警告"will be removed in v5.2"，属于 v3.6+ 过渡分支。该分支的参数名相对官方文档有改动：

| 旧名（design.md / 公开文档） | 这版接受的新名 | 来源 |
|---|---|---|
| `--train_type lora` | **`--tuner_type lora`** | `materials/swift_training.docx` debug 段落 `SftArguments(tuner_type='lora', ...)` |
| `--quantization_bit 4` | **`--quant_bits 4`** | 排除法验证（旧名进 remaining_argv，新名不进） |

中间还试过 `--sft_type lora` 也被拒——排除法把第三种命名候选锁定。

**解决方案** — `notebooks/notebook.ipynb` cell-2-sft-train：换成 `--tuner_type lora` 和 `--quant_bits 4`，并加注释保留两组对应关系，方便以后跨版本回退。

**涉及文件**: `notebooks/notebook.ipynb` cell-2-sft-train

---

### 问题 10 — T4 不支持 bf16 / flash-attn 2.x（硬件层面）

**现象**
- `--bf16 true` 在 T4 上不报错，但训练慢（软件模拟）；混合精度数值不稳的间接表现是 loss 爆炸/收敛差
- `pip install "flash-attn==2.8.3"` 在 Colab T4 上编译失败或运行时报"unsupported architecture"

**根因**

| 硬件 | Compute Capability | bf16 native | flash-attn 2.x |
|---|---|---|---|
| Colab 免费 T4 | 7.5 (Turing) | **❌** | **❌**（要求 SM ≥ 8.0） |
| AutoDL 4080 SUPER | 8.9 (Ada Lovelace) | ✅ | ✅ |
| A100 / H100 | 8.0 / 9.0 | ✅ | ✅ |

我们想"一份 notebook 跨硬件跑"，必须运行时检测。

**解决方案**

1. **SFT CLI**（cell-2-sft-train）— T4 路径用 fp16：
   ```
   --bnb_4bit_compute_dtype float16   # 不是 bfloat16
   --fp16 true                        # 不是 --bf16 true
   ```
2. **推理 cell**（cell 3.5a `74056ebe`）— 自动检测：
   ```python
   _compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
   ```
   `BitsAndBytesConfig(bnb_4bit_compute_dtype=_compute_dtype)` 和
   `from_pretrained(..., torch_dtype=_compute_dtype)` 都跟随。
3. **不装 flash-attn 2.x**，让 transformers 自动选 sdpa attention（PyTorch 内置，T4 上工作良好）。

**涉及文件**: `notebooks/notebook.ipynb` cell-2-sft-train, cell `74056ebe` (3.5a), cell `ae6d1495` (DPO 加载)

---

### 问题 11 — Qwen3.5 是 VL + GatedDeltaNet 模型，依赖栈被低估

**现象**

加载时警告：
```
Please install the package: `pip install "qwen_vl_utils>=0.0.14" "decord" -U`.
```
以及：
```
The fast path is not available because one of the required library is not installed.
Falling back to torch implementation. To install follow ...flash-linear-attention... causal-conv1d
```

**根因**

我们之前 cell 注释写"text-only Qwen3.5-4B base / no ViT"——**完全错**。`materials/qwen3.5_key_points.docx` 明确说：

> Qwen3.5 属于**混合思考的多模态模型**，结合了 linear attention (GatedDeltaNet) 和 full attention。

即使我们做纯文本任务，模型加载时仍要：
- `qwen_vl_utils` — 模型类初始化时检查
- `flash-linear-attention` (fla) + `causal-conv1d` — GatedDeltaNet 的快速 kernel；缺则降级到 torch 实现，慢但能跑

**解决方案** — `cell-setup-2` 重写依赖列表：
```bash
pip install -U "transformers==5.2.*" "qwen_vl_utils>=0.0.14" peft trl liger-kernel \
                bitsandbytes accelerate ms-swift modelscope ...
pip install -U "flash-linear-attention>=0.4.2" --no-build-isolation
pip install -U "git+https://github.com/Dao-AILab/causal-conv1d" --no-build-isolation
```

**故意不装** flash-attn 2.x（见问题 10）。`transformers==5.2.*` 是 docx 锁的版本（5.1 没 Qwen3.5，5.3 视频 dataloader 坏）。

**附带**：思考模式三件套（防 base 模型生成长 `<think>...</think>` 破坏 `LABEL ##[..]##` 格式）：
```
--enable_thinking false
--add_non_thinking_prefix true
--loss_scale ignore_empty_think
```

**涉及文件**: `notebooks/notebook.ipynb` cell-setup-2, cell-2-sft-train, cell-2-sft-download, cell-2-sft (md)

---

### 问题 12 — Track 1 全 154 条 0 秒跑完 + 静默 `AttributeError()`

**现象**
```
predict: 100%|...| 154/154 [00:00<00:00, 967.59it/s]
  WARN claim-XXX: AttributeError()       (× 154)
Track 1 (Base, no RAG) acc=0.2662 F=0.0000 HM=0.0000 (0.1s)
```

`AttributeError()` 没有 message body，每条 claim < 1 ms 就抛错——根本没到 `model.generate()`。`acc≈0.27` 接近 NEI 的占比 31%，说明每条都走了 `except` 兜底默认 `NOT_ENOUGH_INFO`。

**根因**（用 smoke test 脚本探针逐步定位）

`apply_chat_template(..., return_tensors="pt")` 在 transformers 5.x 上**返回 `BatchEncoding`（dict-like），不是 tensor**。我们 `src/inference.py` 三处都把它当 tensor 用：
```python
prompt_ids = tokenizer.apply_chat_template(...).to(self.model.device)
# ...
new_ids = out[0][prompt_ids.shape[1]:]
```
`BatchEncoding.__getattr__('shape')` → 内部走 `self.data['shape']` → KeyError → 转译成无 message 的 `AttributeError`。`predict_all` 的 `except Exception` 把 traceback 完全吞了，只剩单行 WARN。

**解决方案**

1. `src/inference.py` 加统一 helper：
   ```python
   def _apply_template_to_device(tokenizer, msgs, device):
       encoded = tokenizer.apply_chat_template(
           msgs, return_tensors="pt", add_generation_prompt=True,
           enable_thinking=False,
       )
       prompt_ids = encoded if torch.is_tensor(encoded) else encoded["input_ids"]
       return prompt_ids.to(device)
   ```
   `ModelInferer.predict` / `NoRagInferer.predict` 全部走 helper。`ZeroShotInferer` 继承自动跟进。
2. notebook `cell-2-infer-code` 的内联 `infer_one()` 同步打补丁。
3. **同时** 改 `predict_all`，前 3 个错误打完整 traceback 而非只打 repr：
   ```python
   if _err_traces_shown < 3:
       print(f"  WARN {cid}: {e!r}")
       _tb.print_exc()
   ```
   防止以后再被静默 AttributeError 困住。

**复用价值**：transformers 5.x 在多处行为变了，BatchEncoding 是高频踩坑。鸭子类型判断（`torch.is_tensor` else `["input_ids"]`）比依赖 tokenizer 类名更稳。

**涉及文件**: `src/inference.py`, `notebooks/notebook.ipynb` cell-2-infer-code

---

### 问题 13 — AutoDL 实例 PyTorch + driver + CUDA toolkit 三向不匹配

**现象**

`pip install -U "git+.../causal-conv1d" --no-build-isolation` 编译失败：
```
RuntimeError: ('The detected CUDA version (%s) mismatches the version that was
used to compilePyTorch (%s).', '12.4', '13.0')
```
更上面还有：
```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
(found version 12060). Please update your GPU driver
```

**根因**

| 组件 | 版本 |
|---|---|
| PyTorch | 2.11.0+**cu130** |
| CUDA toolkit | 12.4 |
| GPU driver | 12.6（上限） |

PyTorch 用 CUDA 13 编译，但 driver 只支持到 12.6。即使 causal-conv1d 不装，后续 bitsandbytes 4-bit / liger-kernel / 训练 forward 都会接连炸。这是个**镜像层面**的根因。

**解决方案**

不在原实例修，**直接销毁 + 重建**，选 AutoDL 标准镜像：`PyTorch 2.5.1 + CUDA 12.4 + Python 3.12`。新实例验证：
```bash
nvidia-smi  # CUDA 12.6 driver
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# torch 2.5.1+cu124, cuda 12.4, ok True
```

后续 causal-conv1d / fla 编译都顺利。

**复用经验**：环境冲突类问题，与其在原镜像上往回卸装重装，不如**重建实例**。AutoDL 创建一个新实例只要几分钟，远比折腾 conda + cu* 各种版本号便宜。

**涉及文件**: 无代码改动（环境层）；`scripts/test_qwen35_inference.py` 的 Section 1 会自动 dump 这些版本号

---

### 问题 14 — Python 模块缓存让 src/inference.py 修改不生效

**现象**

修复问题 12 后，文件磁盘上有改动（`grep "_apply_template_to_device" src/inference.py` 命中 3 行），但 Track 1 重跑还是 0 秒报 154 条 AttributeError —— **完全跟没修一样**。

**根因**

Notebook kernel 第一次 `from src.inference import NoRagInferer` 时 Python 把 `src.inference` 模块缓存进了 `sys.modules`。之后即使文件改了，再 `from ... import` 拿到的还是旧版本的 `NoRagInferer` 类对象。

**解决方案**

1. **强烈推荐** Runtime → Restart session。一次干净，不留隐式状态。
2. 不想重启时（避免重下模型）的最小干预：
   ```python
   import sys
   for m in list(sys.modules):
       if m == "src.inference" or m.startswith("src.inference."):
           del sys.modules[m]
   from src.inference import NoRagInferer  # 重新导入
   ```
3. 用诊断 cell 强制确认加载到的是哪一份：
   ```python
   import src.inference
   print("file:", src.inference.__file__)
   print("has helper?", hasattr(src.inference, "_apply_template_to_device"))
   ```

**复用经验**：改了 `src/*.py` 之后，先跑 1 条诊断 print 确认新代码在跑，再做大批量调用。否则一旦失败，无法分辨"代码 bug"和"模块缓存"。

---

### 问题 15 — SFT 数据迁移到 ms-swift messages 标准格式

**背景**

之前 SFT 数据格式是 `{id, system, query, response, _meta}`（query-response 形式）。`materials/训练数据格式.docx` §2.1 说 ms-swift 的 AutoPreprocessor **能自动转**这种格式，但官方推荐 **messages 列表**为标准/无歧义格式。考虑到我们已经撞过 v3.6+ 分支若干隐式行为变化，迁移到 messages 更稳。

**新 schema**（per docx §2.1 / §3.3）

SFT：
```json
{"messages": [
  {"role": "system",    "content": "<SYSTEM_PROMPT>"},
  {"role": "user",      "content": "<query>"},
  {"role": "assistant", "content": "LABEL ##[indices]##"}
]}
```

DPO：
```json
{"messages": [system, user, assistant=chosen],
 "rejected_response": "..."}
```

**改动**

| 文件 | 改动 |
|---|---|
| `src/sft_dataset.py` | `build_sft_record` + `build_hard_negative_record` 输出 messages 三元组；docstring 加 docx 出处 |
| `src/dpo_pairs.py` | `build_dpo_pair` / `synthesise_disputed_contrast` 改写；新增 `_messages_with_chosen()` helper 复用 [system,user] 并替换 assistant 为 chosen |
| `tests/test_sft_dataset.py` | 断言改成 `r["messages"][2]["content"]` 风格；本地 all green |
| `tests/test_dpo_pairs.py` | fixture 改成 messages 形式；本地 all green |
| `notebooks/notebook.ipynb` cell-1-sft (md) | 描述改成 messages 格式 |

**ms-swift 兼容性**：保留 `id` / `_meta` 顶级字段——ms-swift 忽略未知 key，下游（curriculum sort / DPO 配对 / ablation 切片）依然能用。

**用户需要做的**：在 Colab/AutoDL 重跑 `cell-1-sft-code` 重新生成 `outputs/sft_data/sft_train_v1.jsonl`。

**涉及文件**: 见上表

---

## 复用经验（会话 2 增量）

### 9. transformers 5.x 的隐式行为变化清单
- `apply_chat_template(return_tensors="pt")` 返回 `BatchEncoding` 而非 Tensor → 必须 `if torch.is_tensor(x) else x["input_ids"]`
- `BatchEncoding.__getattr__` 缺 key 时抛**无 message 的 AttributeError** —— 排查 silent error 时第一个怀疑这个
- tokenizer 类名变成 `TokenizersBackend`（不是 `Qwen2Tokenizer`），代码不要 `isinstance(tok, Qwen2Tokenizer)` 这种硬判断

### 10. Qwen3.5（VL + GatedDeltaNet）必备依赖栈
```
"transformers==5.2.*"            # 5.1 缺模型，5.3 视频坏
"qwen_vl_utils>=0.0.14"          # VL 模型加载即检查
"flash-linear-attention>=0.4.2"  # GatedDeltaNet kernel
git+.../causal-conv1d            # 同上配套
liger-kernel                     # 可选但训练显存大幅省
```
**不装** flash-attn 2.x（要 SM ≥ 8.0），sdpa attention 在 Turing 上更稳。

### 11. T4 vs Ampere+ 的 dtype 选择（硬件运行时检测）
```python
_compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
```
所有 `BitsAndBytesConfig` / `from_pretrained` / SFT CLI 都跟随。**永远不要**硬编码 `bfloat16`。

### 12. Qwen3.5 思考模式三件套（防 base 模型乱输出）
训练侧（CLI）：
```
--enable_thinking false
--add_non_thinking_prefix true
--loss_scale ignore_empty_think
```
推理侧（code）：
```python
tokenizer.apply_chat_template(msgs, ..., enable_thinking=False)
```
两端必须一致，否则 train/inference 行为不匹配。

### 13. ms-swift CLI 参数命名跨版本差异速查

| 概念 | 旧名 | 新名（v3.6+） |
|---|---|---|
| 训练方式 | `--train_type`（公开文档） / `--sft_type`（v1.x） | **`--tuner_type`** |
| 量化位数 | `--quantization_bit` | **`--quant_bits`** |
| 思考开关 | (无) | `--enable_thinking false` |
| 思考前缀 | (无) | `--add_non_thinking_prefix true` |
| 损失忽略空 think | (无) | `--loss_scale ignore_empty_think` |
| Liger fused kernel | (无) | `--use_liger_kernel true` |
| 长度分组（替代 packing） | (无) | `--group_by_length true` |
| ckpt 数量上限 | (无) | `--save_total_limit N` |

跑前先 `swift sft --help | grep -iE "(tuner|quant|think|liger)"` 确认。

### 14. 改 src/*.py 后的强制确认习惯
1. `grep` 文件验证磁盘有新代码
2. notebook 跑诊断 cell 验证 `sys.modules['src.X'].__file__` 指向新文件 + 新 attr 存在
3. 才跑业务 cell

### 15. 错误处理别吞 traceback
`predict_all` 的 `except Exception as e: print(repr(e))` 是反模式——空 AttributeError 直接看不出来源。改成"前 N 次打完整 traceback，之后只打 repr"。一行 `traceback.print_exc()` 救命。

### 16. 环境层冲突优先重建实例而不是降级
AutoDL / Colab 这种容器化环境，建实例比 conda 里把 PyTorch 从 cu130 降到 cu124 + 重装一堆 CUDA 包快得多。镜像选错了，5 分钟重建胜过 1 小时排错。

### 17. Standalone smoke-test 脚本的价值
`scripts/test_qwen35_inference.py` 不依赖 notebook、不依赖 RAG 索引，独立验证：env / 模型加载 / tokenizer 行为 / 推理路径。一次跑出 4 个 section，能在 5 分钟内回答："模型本身能不能跑、能不能听懂格式、SC 在易题/难题上分别表现如何"。新模型 / 新硬件第一步永远是它。

### 18. SC（self-consistency）在易题上是浪费
4080 SUPER + 4-bit 量化 + base 模型，简单 SUPPORTS claim 5 次采样 5/5 一致（T=0.7 都没扰动）。SC 真正有价值的场景是 **DISPUTED / 模糊 / 模型置信度低** 的样本。Track 4 的 SC 可以考虑只对低置信度样本启用（节省 5× 成本）。

---

## 修改文件清单（会话 2 增量）

| 文件 | 改动概要 |
|---|---|
| `cell-setup-2` | 依赖列表完全重写：transformers==5.2.*、qwen_vl_utils、fla、causal-conv1d、liger-kernel；明确不装 flash-attn 2.x（T4 不支持） |
| `cell-2-sft-download` | 注释改正：Qwen3.5 是 VL + 混合思考模型；保留只用文本路径 |
| `cell-2-sft` (markdown) | 描述加 T4 硬件警告 + 显存策略堆叠 |
| `cell-2-sft-train` | `--train_type` → `--tuner_type`；`--quantization_bit` → `--quant_bits`；新增 thinking 三件套 + `--use_liger_kernel` + `--group_by_length` + `--save_total_limit`；T4 路径用 `--fp16 true` + `bnb_4bit_compute_dtype float16` |
| `cell-2-infer-code` | `infer_one` 内联同款 BatchEncoding 处理 + `enable_thinking=False` |
| cell `74056ebe` (3.5a) | `_compute_dtype = bf16 if supported else fp16`；`BitsAndBytesConfig` / `from_pretrained` 都跟随；显式注释不指定 attn_implementation |
| cell `ae6d1495` (3.5 load adapter) | DPO 加载也用 `_compute_dtype` |
| cell-1-sft (markdown) | SFT schema 描述从 `{system,query,response}` 改成 messages 三元组；附 docx 引用 |
| `src/inference.py` | 新增 `_apply_template_to_device` helper；`ModelInferer` / `NoRagInferer` 都过它；`predict_all` 前 3 个错误打完整 traceback |
| `src/sft_dataset.py` | `build_sft_record` + `build_hard_negative_record` 输出 messages 格式；docstring 加 docx 出处 |
| `src/dpo_pairs.py` | 新增 `_messages_with_chosen` helper；`build_dpo_pair` + `synthesise_disputed_contrast` 改写；docstring 改 |
| `tests/test_sft_dataset.py` | 断言改成 messages 形式；all green |
| `tests/test_dpo_pairs.py` | fixture 改成 messages 形式；all green |
| **新建** `scripts/test_qwen35_inference.py` | Standalone smoke test：env 探针 + model 加载（auto dtype）+ tokenizer 行为探针 + 4 个推理 section（4a no-RAG / 4b RAG fake ev / 4c SC on DISPUTED + mixed ev / 4d 可选 real RAG，gated on cached indices） |

---

## 实测数据（AutoDL 4080 SUPER, 4-bit）

`scripts/test_qwen35_inference.py` 在 base Qwen3.5-4B 上跑出的关键数字：

- **加载**：8GB 模型下载 ~9 min（ModelScope）；加载耗时 7.9 s；4-bit 加载后 VRAM 2.9 GB
- **Section 4a (no-RAG, greedy)**：3 条样本 SUPPORTS / REFUTES / NEI 中，前两条**正确**；NEI 那条 base 模型给 REFUTES（base 模型常见缺陷：没有"我不知道"概念）
- **Section 4b (RAG fake ev, greedy)**：模型严格按 `LABEL ##[1,2]##` 输出，证明 prompt 格式可学
- **Section 4c (SC, easy SUPPORTS)**：5/5 一致 → 该 claim SC 无价值；后改成 DISPUTED claim 验证 SC 真实分歧情况

**结论**：base 模型已具备 prompt 跟随能力，但 NEI 类需要 SFT 教，SC 在易题上无收益。直接进入 SFT 阶段。

---

## 当前进度（会话 2 收口）

### 已完成
- [x] swift CLI 参数名问题三连解（问题 9）
- [x] T4 hardware 自适应（问题 10）
- [x] Qwen3.5 依赖栈补全（问题 11）
- [x] BatchEncoding bug + helper（问题 12）
- [x] AutoDL 镜像重建（问题 13）
- [x] 模块缓存绕过手册（问题 14）
- [x] SFT/DPO 数据迁移到 messages 格式 + tests 全绿（问题 15）
- [x] Standalone smoke test 脚本，AutoDL 上跑通

### 进行中
- [ ] AutoDL 上重新生成 messages 格式的 `sft_train_v1.jsonl`（cell-1-sft-code 重跑）
- [ ] AutoDL 上跑 SFT（cell-2-sft-train 取消注释 `!{cmd}` 实跑）
- [ ] 跑改进版 4c 看 SC 在 DISPUTED claim 上能否拉开分歧

### 未触及
- [ ] DPO 训练（依赖 SFT checkpoint）
- [ ] 4-track 完整评测（依赖 SFT + DPO checkpoint）
- [ ] 真实 RAG 在 AutoDL 上的索引构建（BM25 + dense）→ 才能开 `--with-real-rag` 跑 smoke test 4d

---

## 断点续跑指南（AutoDL 版补充）

| 缺失 | 重做 | 耗时 |
|---|---|---|
| 整个实例（被释放/欠费） | AutoDL 重建实例 + git clone + 重装依赖 | ~15 min |
| Python env | `pip install -U ...`（按 cell-setup-2 列表） | ~5 min |
| Qwen3.5-4B 模型权重 | `python -m scripts.test_qwen35_inference` 自动下载 | ~10 min（首次） |
| messages 格式 SFT 数据 | 跑 cell-1-sft-code 重新 build_dataset | <30 s |
| BM25 索引 | 跑 cell 2.1 | ~3 min |
| dense 索引 | 跑 cell 2.2（首次 ~30 min；用 4080 比 T4 快很多） | ~15 min（4080 估算） |
