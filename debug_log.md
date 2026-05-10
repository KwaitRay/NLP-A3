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
