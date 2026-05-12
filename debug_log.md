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

### 问题 16 — transformers torch.load CVE 检查 + ModelScope 镜像无 safetensors（双重夹击）

**症状**：AutoDL 上 `python -m scripts.build_indexes` 跑到 dense 索引构建阶段（加载 bge-m3）时报：

```
ValueError: Due to a serious vulnerability issue in `torch.load`, even with
`weights_only=True`, we now require users to upgrade torch to at least v2.6
in order to use the function. This version restriction does not apply when
loading files with safetensors.
See https://nvd.nist.gov/vuln/detail/CVE-2025-32434
```

调用链：`SentenceTransformer.load()` → transformers `_load_pretrained_model()` → `load_state_dict()` → `check_torch_load_is_safe()` → 抛错。

**根因（两层叠加）**：
1. **transformers 包装层 CVE 缓解** — `transformers/modeling_utils.py` 检测到马上要走 `torch.load` 而非 `safetensors`，且当前 torch < 2.6，直接拒绝加载。CVE-2025-32434 是 pickle 反序列化漏洞，即使 `weights_only=True` 也不安全。
2. **ModelScope 镜像缺 safetensors** — `BAAI/bge-m3` 在 ModelScope 上**只发 `pytorch_model.bin`（2.27 GB）**，没有 `model.safetensors`。所以 sentence-transformers 无法走 safetensors 优先路径，被迫退到 .bin → 撞墙。
3. **HF 直连不通** — AutoDL 在墙内，`huggingface.co` 不可达（`[Errno 99] Cannot assign requested address`），不能直接补下 safetensors。

**修复决策**：**本地离线转换 .bin → .safetensors，不升级 torch**。

理由：
- 升 torch 到 2.6 会破坏 flash-attn 2.x / bitsandbytes / flash-linear-attention 整条供应链（debug_log 问题 13 重建实例就是为了固定这套依赖）
- `torch.load` 这个 API 本身在 2.5 下能用，**限制只在 transformers 的包装层**；直接在用户代码里调 `torch.load` 不受限
- bin 文件已经在磁盘上了，**0 网络消耗**，CPU 30s 完事

**落地**：新建 `scripts/convert_bin_to_safetensors.py`：
- 扫 `models/*/`，跳过已有 `*.safetensors` 的目录（Qwen3.5-4B 是 sharded safetensors，自动跳过）
- 对 single-file `pytorch_model.bin` 调 `torch.load` + `safetensors.save_file(state_dict.contiguous().clone())`
- 原 `.bin` 改名 `.bak`（保留回滚能力），sentence-transformers 自动选 safetensors
- 警告并跳过 sharded `pytorch_model-*-of-*.bin`（需重建 index.json，不安全），建议改走 `HF_ENDPOINT=https://hf-mirror.com` 重下

**HF 镜像 fallback**（如果将来确实需要单独补文件）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='BAAI/bge-m3', filename='model.safetensors', local_dir='models/bge-m3')"
```

**涉及文件**：`scripts/convert_bin_to_safetensors.py`（新建）；`TODO.md` Step 1 插入该步；`optimization_plan.md` §7.2 artifact 清单 + §8 风险表。

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

---

# 会话 3 — 2026-05-11 — Phase 1 baseline 跑通 + 数字异常诊断

## 会话元信息

- **日期**: 2026-05-11
- **环境**: AutoDL 4080 SUPER 31.5 GB VRAM；BM25 (146 MB) + dense (9.2 GB) 索引 + 4 个模型权重 + messages 格式 SFT 数据全部就绪
- **范围**: 跑 `scripts.phase1_eval --tracks 1,2 --prompts v1` 的首次完整评估；结果数字触发新一轮诊断

---

## 问题 17 — Track 1 Acc 几乎等于 NEI 占比，疑似 parser fallback 全部走 NEI

**现象**

`python -m scripts.phase1_eval --tracks 1,2 --prompts v1 --dataset diag_test` 跑完：

| Track | F | Acc | HM | 耗时 |
|---|---|---|---|---|
| 1 (no-RAG, greedy) | 0.0000 | **0.3223** | 0.0000 | 69.7 s / 121 条 |
| 2 (RAG, greedy)    | 0.1169 | 0.4215     | 0.1830 | 199.3 s / 121 条 |

`diag_test` 的 label 分布：

| label | n | 占比 |
|---|---|---|
| NOT_ENOUGH_INFO | 40 | 33.06% |
| SUPPORTS | 38 | 31.40% |
| REFUTES | 22 | 18.18% |
| DISPUTED | 21 | 17.36% |

**Track 1 Acc=0.3223 = 39/121** 离"全猜 NEI 多数类"的 40/121 只差 1 条，可疑。

**根因假设（两条）**

1. **(a) Parser fallback 全部走 NEI**：模型生成的文本不含 `LABEL ##[..]##` 格式 → `parse_response` 返回 `default_label='NOT_ENOUGH_INFO'`（见 `src/prompt.py:228`）→ Track 1 Acc 恰好等于 NEI 比例。如果是这条，**需要**改 prompt 或 parser，**不需要** Phase 4 数据扩充。
2. **(b) 模型真在生成 4 类标签，但偏 NEI**：模型能输出 SUPPORTS / REFUTES，但对 non-NEI 类的判别力很弱，恰好猜对 NEI 那部分。如果是这条，prompt 帮不上忙，需要 Phase 4 SFT 数据扩充。

**两者的修复路径完全不同**，必须先区分清楚再行动。

**修复路径不明的次要支撑论据**

跟问题 12 的"0 秒报 154 条 AttributeError"不一样：
- 当前 69.7 s / 121 条 = 0.58 s/条，**模型是真的在生成**（167 it/s 那次是 0 推理直接 fallback；这次 1.74 it/s 是正常生成速度）
- `_apply_template_to_device` helper 已就位（问题 12 修复保留）
- 但 `predict_all` 没保存 raw model output，只存了 parsed 后的 `(claim_label, evidences)`，**所以无法从 saved JSON 直接确认 hypothesis (a)**

**诊断工具**：新建 `scripts/diagnose_phase1.py`

零模型加载、纯分析现有 `outputs/eval_phase1/track*_v1_diag_test.json` 文件。每个 (track, prompt) 输出：

- **predicted vs gold label distribution** — 看预测分布是否塌缩到 NEI
- **confusion matrix (4×4)** — 看 non-NEI gold 被预测成什么
- **per-gold-label correctness** — 拆开每个 gold label 的准确率
- **evidence recall (Track 2+)** — 拼检索质量与模型是否在 cite 正确 ev
- **defaulting-to-NEI heuristic flag** — 自动判定：predicted NEI > 50% **且** non-NEI gold 中 ≥50% 被预测成 NEI → ⚠️ pattern
- **sample mispredictions** — 3 条 non-NEI gold → NEI、3 条 non-NEI gold → 另一个 non-NEI、3 条 correct non-NEI（供眼检）

**运行**：

```bash
# AutoDL 上跑 phase1_eval 之后
python -m scripts.diagnose_phase1 --dataset diag_test
```

**输出**：`outputs/eval_phase1/diagnose_<dataset>.md` + stdout 跨 run 摘要。

**期望读法**

- 如果 `非-NEI acc` 普遍接近 0 + `预测 NEI 占比` > 50% → **(a) parser fallback 模式确认** → 下一步：用 `scripts.test_qwen35_inference` 重跑几条 diag_test 实例，打印 raw output 看具体输出长什么样；改 prompt v2 或在 `parse_response` 加更宽容的容错。
- 如果 `非-NEI acc` 显著大于 0 且 confusion matrix 在对角线附近有信号 → **(b) base 模型真在判别只是偏弱** → Phase 2 跑 v2/v3/v4 看 prompt 能否拉起，然后进 Phase 4。

**Track 2 F=0.1169 的次问题**：先放一放。Track 2 Acc=0.4215 比 Track 1 +9 条说明 RAG 真有用；F 偏低可能是 (1) 检索没命中 gold ev，或 (2) 模型 cite 的 index 错。诊断脚本里的 evidence recall 行能回答 (1)；(2) 需 raw output 才能查。如果 (1) 不严重，**Phase 2 prompt sweep 后再回头看 (2)**。

**涉及文件**

- `scripts/diagnose_phase1.py`（**新建**）
- `TODO.md` — Step 2.5（新增）插入诊断步骤；Step 3 后移
- `optimization_plan.md` §9 进度更新 + §10 决策日志追加一行
- `outputs/PROGRESS.md` — 会话 8 记录

---

## 复用经验（会话 3 增量）

### 19. 数字像多数类基线时第一反应：parser fallback，不是模型偏置

只看 Acc 一个数会被"恰好等于多数类占比"误导。Phase 1 之后**强制走 confusion matrix + 预测分布**，而不是只盯 Acc / HM。`diagnose_phase1.py` 把这步固化成 1 行命令。

### 20. predict_all 应该保留 raw output 选项（待办）

`src/inference.py:predict_all` 只存 parsed 结果。一旦怀疑 parser fallback，必须再跑一次 inference 才能拿到 raw text 验证。建议加 `--save-raw` flag：写 sidecar `outputs/eval_phase1/track*_*_<dataset>.raw.jsonl`（每行 `{cid, raw_text}`），代价是 inference 慢 < 1%，但诊断时省一次完整重跑。

**实施时机**：等 (a) vs (b) 真相落地后再考虑——如果是 (a) 那确实需要这个 flag；如果是 (b) 则 raw output 用处不大。
**2026-05-12 更新**：诊断已确认是 (b) — Track 1 confusion matrix 显示模型在主动判别（SUPPORTS 65.8% / REFUTES 59.1%），并非 fallback。raw output flag 优先级降为 P3。

### 21. prompt 教不会 base 模型没有的概念（量化证据）

Phase 2 prompt sweep + diagnose 同时覆盖三个变体，提供**量化反例**：

| 指令 / Instruction | per-label acc 变化 | 副作用 |
|---|---|---|
| v2 "Use NEI when ev off-topic" | NEI 0.350 → **0.550** (+20pp) | REFUTES 0.500 → 0.227 (−27pp)：模型把不该 NEI 的 REFUTES 也判 NEI |
| v3 "Use DISPUTED when ev split" | DISPUTED 0.286 → **0.476** (+19pp) | REFUTES 0.500 → **0.000** (−50pp)；NEI 0.350 → 0.175；总 Acc 暴跌至 0.256 |
| v4 v3 + 4 few-shot | DISPUTED 0.286 → 0.333 (+5pp) | NEI **0.050**（甚至比 Track 1 还低！）；few-shot 的 NEI 单例反而压死了 NEI 预测 |

**结论**：base 模型对 NEI/DISPUTED 是**能力性缺失**（Track 1 acc 分别 0.025 / 0.000）。prompt 能让模型**输出**这些 label，但不能让它**挑对**该用的 claim。修复必须靠 SFT（数据示范"什么样的证据 → NEI"）+ DPO（对抗"证据矛盾 vs 单边支持"的 chosen/rejected 对）。

**复用**：以后看到一个 label 的 acc 很低时，先做 Track 1 ablation 确认 base 是否本来就不会；如果不会，prompt 工程是浪费。

### 22. F-score 天花板 = retrieval recall，SFT 前必须先审计检索

Phase 2 sweep 顺手发现的**最大教训**：4 个 prompt 在 Track 2 上的 **evidence recall 全部 ≈ 0.11**（macro 0.10-0.11, micro 0.09-0.10），与 prompt 无关——**检索 pipeline 的 top-5 输出本身就只命中 11% 的 gold ev**。

数学：当前架构下，即使 label classification 100% 正确：
- F = 2·P·R / (P+R)，R = 0.11，假设 P ≈ 0.13 → F ≈ 0.12
- HM = 2·Acc·F / (Acc+F)，Acc=1.0, F=0.12 → **HM ≈ 0.21（硬上限）**

也就是说：**如果不先解决检索，无论 SFT 多猛，HM 最多到 0.21**。Phase 4 SFT 实际能拉的红利不超过 +0.02。

**触发的新 phase（§3.5）**：在 SFT 前跑 `scripts.retrieval_ceiling.py` 扫四组：`final_k` sweep / 检索器 ablation / fusion 权重 / synonym multi-query。最便宜的 win 估计是 `final_k=5 → 20`（recall 可能翻 2-3 倍，prompt 长度涨 ~1500 token，4080 SUPER 还远没到 OOM）。

**复用**：每次 sweep prompt / model 时**同时打印 evidence recall**——`diagnose_phase1.py` 已实现。如果 recall 在 4-5 个变体上都一样，瓶颈不在 sweep 的维度上，停止 sweep 转向被忽略的 retrieval 维度。

### 23. final_k sweep 揭示 base 模型的"信息密度悖论"

Phase 3.5 把 `final_k` 从 5 扫到 20（k ∈ {5, 10, 20}）端到端实测：

| k | non-NEI acc | NEI acc | 总 Acc | HM | predicted NEI 占比 |
|---|---|---|---|---|---|
| 5  | 0.457 | **0.350** | 0.422 | 0.183 | 25% |
| 10 | 0.494 | 0.175 | 0.388 | 0.196 | 12% |
| 20 | **0.580** | **0.025** | 0.397 | **0.203** | 3% |

**单调权衡**：k 大 → non-NEI 涨（更多 gold ev 进 cite）+ NEI 跌（20 条 ev 里"总有看起来沾边的"，模型不肯说不知道）+ HM 略升。

**关键启示**：base 模型把"evidence 多"误读成"必有答案"。NEI 不是输出格式问题，是**判断"这些 ev 都不相关"的能力问题**。SFT 必须用 NEI oversample + hard-neg 教这个判断（§0.5.3 硬约束 1，现在 quantified）。

**生产决策**：锁 k=20 而不是 k=10 因为：
1. HM 最高（虽然 0.007 差距是噪声级，但单调趋势真实）
2. 对 SFT 来说**起点 NEI acc 是 0.025 还是 0.175 无关**（SFT 重写行为，不保留 base 习惯）
3. k=20 把所有 error 压到 NEI 一类 → SFT 信号最清晰
4. Phase 4 投影：SFT 把 NEI acc 修到 0.50 时 → k=20 HM ≈ 0.219，k=10 HM ≈ 0.207

`RetrievalConfig.final_k` default 5 → 20；`phase1_eval --final-k` default 同步；
非 default k 自动加 `_kN` 后缀防覆盖（`diagnose_phase1._strip_k_suffix` 解析）。

### 24. notebook `snapshot_download(cache_dir=…)` 不会复用 `models/` —— 必须 cache-first

`scripts/download_models.py` 把 Qwen3.5-4B 放在 `models/Qwen3.5-4B/`（仓库相对路径，便于 `ls / scp / git status`）。但 `notebook_autodl.ipynb` 的 `sec2-5-download` cell 默认调用：

```python
MODEL_CACHE = str(Path(CACHE_ROOT) / "model_cache")   # /root/autodl-tmp/nlp_a3_cache/model_cache
MODEL_DIR = snapshot_download("Qwen/Qwen3.5-4B", cache_dir=MODEL_CACHE)
```

`snapshot_download` **只看自己的 `cache_dir`**，不知道 `models/Qwen3.5-4B/` 存在 → 在 AutoDL 上触发 8 GB 重复下载（已发生：2026-05-12 SFT 启动时被截断挽救）。

**修复模式（已落在 `phase1_eval.py:load_model_and_tokenizer` + `notebook_autodl.ipynb` sec2-5-download / sec3-5a）**：

```python
from src.paths import MODELS_DIR
_local = MODELS_DIR / "Qwen3.5-4B"
if (_local / "config.json").exists():
    MODEL_DIR = str(_local)           # cache hit
else:
    MODEL_DIR = snapshot_download(...)  # fallback download
```

**复用要点**：任何 ModelScope/HF `snapshot_download` 调用前都先看 `models/` —— 这是项目约定（design.md D-013）。Notebook 的每个模型加载 cell 都要走这个模式（包括将来 DPO 加载基座、ablation 加 9B 模型时）。

### 25. AutoDL `git pull` / `pip install` 超时 → `source /etc/network_turbo`

**症状**：
```
fatal: unable to access 'https://github.com/...': Failed to connect to github.com port 443 after 130930 ms: Connection timed out
```

或 `pip install` 长时间卡住、ModelScope 也偶尔超时。

**根因**：AutoDL 实例在国内，默认不能直连 github.com / HuggingFace。

**修复**：AutoDL 官方提供"学术加速通道"（白名单代理，免费，不计流量）：

```bash
source /etc/network_turbo
git pull origin main   # 几秒回
```

**作用域**：当前 shell session。**关 shell 失效**，每次新 ssh / 新 jupyter terminal 都要重 `source`。

**Fallback**（如果 `/etc/network_turbo` 不存在，旧镜像）：
- ghproxy: `git remote set-url origin https://ghproxy.com/https://github.com/<user>/<repo>.git`
- 或 `kkgithub.com` / `hub.fastgit.xyz` 等镜像
- pip：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ...`

**已落在 `TODO.md` 附录 + `requirements.txt` 顶部注释**。

### 26. ModelScope canonical 路径 = `Qwen3.5-4B`（一个点），不是 `Qwen3___5-4B`

老版本 ModelScope 在 hub URL 里把 `.` 转义成 `___`（triple underscore），导致 fork 出的代码到处写：

```
outputs/model_cache/Qwen/Qwen3___5-4B   # 旧的，已 stale
```

**现在 ModelScope ≥ 1.x** 直接用 `.` 落盘：

```
.../Qwen/Qwen3.5-4B    # 实际下载目录
```

`scripts/download_models.py:_download_via_modelscope` 也把内容 move 到 `models/Qwen3.5-4B/` 统一命名。

**所有代码 / 文档 / notebook 都用 `Qwen3.5-4B`** —— 2026-05-12 清掉了仓库里最后 5 处 `Qwen3___5-4B` 残留（optimization_plan / TODO / test_qwen35_inference docstring）。

**新代码约定**：模型目录名 = repo basename（`Qwen/Qwen3.5-4B` → `Qwen3.5-4B`），由 `paths.resolve_model_path()` 统一解析，**不要硬编码 `outputs/model_cache/<owner>/<basename>` 路径**（那是 ModelScope 内部细节，不稳定）。

### 27. ms-swift 4.2.0 的 FSDP2 import vs AutoDL torch 2.5.1 — 不降版本，只 patch 一行

**症状**（2026-05-12 SFT 启动时）：

```
ImportError: cannot import name 'FSDPModule' from 'torch.distributed.fsdp'
  (/root/miniconda3/lib/python3.12/site-packages/torch/distributed/fsdp/__init__.py)
```

栈底：`swift/callbacks/activation_cpu_offload.py:4` 触发，链路：

```
swift.callbacks/__init__.py → from .mapping import callbacks_map
swift.callbacks.mapping     → from .activation_cpu_offload import ActivationCpuOffloadCallBack
swift.callbacks.activation_cpu_offload → from torch.distributed.fsdp import FSDPModule as FSDP2
```

**根因**：`FSDPModule` 是 **torch 2.6+** 才暴露的 API。ms-swift 4.2.0 给 activation CPU offload 加 callback 时假设 torch ≥ 2.6，没做向后兼容。AutoDL 标准镜像锁在 torch 2.5.1+cu124（问题 13 选定的，flash-attn 2.x / bitsandbytes / fla 全栈靠它）。

**关键决策：不降 ms-swift，不升 torch**。

| 路径 | 改动 | 风险 |
|---|---|---|
| 降 ms-swift（试过 `<4.2`）| 装到了用 `swift/llm/` 老模块路径的更早版本 | 丢 `--tuner_type` / `--enable_thinking` / `--add_non_thinking_prefix` —— 我们 Phase 4 训练命令全部用这些，相当于回到 debug_log 问题 9 的初态 |
| 升 torch 到 2.6+ | flash-attn 2.x / bitsandbytes / fla 全栈要重测；问题 13 重建实例的努力作废 | 高 |
| **patch FSDP2 import 为可选**（采纳） | `scripts/patch_swift_fsdp2.py` 一行 try/except + stub 类 | 极低 — callback 不用就不会实例化 stub |

`materials/qwen3.5_key_points.docx` 推荐 latest ms-swift + flash-attn 2.8.3 + transformers 5.2.* + python 3.12，**隐含假设 torch 2.6+**（但没明说）。`materials/swift版本适配.docx` 列出版本矩阵，torch 2.5.x 行推荐 swift 配 vLLM 加速场景，FSDP2 是 2.6+ 才的特性。

**修复**：`scripts/patch_swift_fsdp2.py`

- 扫 `swift/**/*.py` 找到所有 module-load 期就 import `FSDPModule` 的文件（当前 4.2.0 只有一处）
- 把单行 `from torch.distributed.fsdp import FSDPModule as FSDP2` 包成：

  ```python
  try:
      from torch.distributed.fsdp import FSDPModule as FSDP2
  except ImportError:
      class FSDP2:  # stub for torch<2.6
          pass
  ```

- 用一个 sentinel comment (`torch < 2.6 fallback. The class is only used by`) 标记已 patch 状态 → 幂等
- 最后 `importlib.import_module("swift.callbacks")` 自检通过

**集成到 notebook**：`cell-setup-2` pip install 完后自动 `os.system(f"{sys.executable} -m scripts.patch_swift_fsdp2")` —— **新建 AutoDL 实例不再撞这个坑**。

**复用要点**：

1. ms-swift 任何 callback 撞老 torch API 时同样可用此模式（扫 + try/except + stub）
2. 看 `materials/swift版本适配.docx` 的版本矩阵确认 torch 与 ms-swift 哪个特性绑哪个 torch 版本
3. **不要因为单个文件 import 失败就降整个库** —— 看清楚 callback 是不是真用得到（90% 训练命令用不到 FSDP / activation offload），patch 一行远比降版本干净

### 28. peft × transformers 5.2 — `HybridCache` 被移除，老 peft 硬引用炸

**症状**（2026-05-12 SFT 启动时第二次冒）：

```
ImportError: cannot import name 'HybridCache' from 'transformers'
  (.../transformers/__init__.py)
File ".../peft/peft_model.py:37"
  from transformers import Cache, DynamicCache, EncoderDecoderCache, HybridCache, ...
```

**追踪**：`import swift` → `transformers.trainer_utils` → `peft.peft_model` → `from transformers import HybridCache` ✗

但 `pip show transformers` 已经是 **5.2.0**，`qwen3_5` model_type 也能识别 —— **transformers 5.2 故意移除了 `HybridCache`**（cache 系统重做），但 peft < 0.17 还硬编码引用它。

**根因**：transformers 4.x → 5.x 是 breaking-change major bump，cache 类被重命名 / 移到 `transformers.cache_utils`。peft 必须出新版同步。

**误导**：`scripts/patch_swift_fsdp2.py` 的 generic try/except `import swift` → 笼统报 "ms-swift not installed in this environment"。真正错误得看 `python -c "import swift"` 的完整 traceback 倒数第 2 行（`ImportError: cannot import name '<X>' from 'transformers'`）才看得到。**Lesson**: patch 脚本的 except handler 应该 print 完整 chained exception，不要只看顶层。

**修复**：

```bash
source /etc/network_turbo
pip install -U peft -i https://pypi.tuna.tsinghua.edu.cn/simple
# 实测 peft 0.17+ 兼容 transformers 5.2.* （删去了 HybridCache 引用）
python -c "from peft import PeftModel; import swift; print('OK')"
```

**已落到 requirements.txt**：之前只 pin `peft>=0.14.0`，上限不够紧；改成 `peft>=0.14.0` 不动（pip 自动取最新；但 AutoDL 上要 force-reinstall 时记得手动 `-U`）。

**复用要点**：

1. **transformers 5.x 是 breaking change，所有依赖 transformers 的库（peft / trl / accelerate）都要同步升**。pin transformers 时必须 pin 所有"transformers-aware"依赖到 5.x-兼容版本。
2. **诊断 chained ImportError 看倒数第 2 行**，不要看 traceback 顶层（顶层永远只是触发点，根因在最深的 import）。
3. **patch_swift_fsdp2 的报错 UX 应该改**：catch ImportError 时 print exception 的 full repr，不要笼统说"not installed"。低优先级 TODO。

### 29. JupyterLab autosave vs git pull 死锁 — `nbstripout --install` 解，但要 close notebook

**症状**（本次 session 撞了 4 次）：

```
$ git pull origin main
error: Your local changes to the following files would be overwritten by merge:
        notebooks/notebook_autodl.ipynb
Please commit your changes or stash them before you merge.
Aborting
```

`git stash` → 成功收到 stash → `git pull` 还冒同样错。`rm` + `git checkout HEAD --` → 也无效，立即又脏。

**根因**：**JupyterLab 那个 notebook 还在浏览器里打开着**。JupyterLab 每 ~30 s autosave 一次，把内存里的 cell outputs / execution_count 写回 `.ipynb`。流程：

```
t=0  git checkout HEAD --   ← working tree 干净
t=1  JupyterLab autosave    ← working tree 脏（outputs 回写）
t=2  git pull               ← 报"would be overwritten"
```

**只要 JupyterLab tab 开着，git 永远追不上 autosave 的写入节奏**。

**修复**（这条死循环正确解法，按顺序）：

1. **JupyterLab UI：`File → Close and Shut Down Notebook`**（必须 Shut Down 杀 kernel，单 Close tab 不够，autosave background 还活着）
2. 终端：`git fetch origin main && git reset --hard origin/main` （硬覆盖，绕开所有 dirty 检测）
3. 验证：`git log --oneline -3` 看到远端 head；`ls scripts/*.py` 看到新文件
4. JupyterLab 重新打开 notebook + Restart Kernel

**长期解 — `nbstripout`**：

```bash
pip install nbstripout
nbstripout --install   # 装到 .git/config 的 filter
# 从此 git add 自动剥 jupyter outputs/execution_count；
# JupyterLab autosave 再多次也不会触发 dirty。
```

**注意**：`nbstripout --install` 后**当前已脏的 working tree 不会自动变干净**，需要 `git add --renormalize .` 重规范化一次。

**已落到 requirements.txt**（torch 2.5 + torch 2.6 两份都有），新建实例 `pip install -r requirements.txt` 完后跑一次 `nbstripout --install`。

**复用要点**：

1. **改 notebook 之前先在 JupyterLab UI 里 Shut Down 它**。kernel 还活着就别在终端做 git 操作。
2. `git stash` 不能突破 autosave 循环 —— stash 之后 JupyterLab 立刻写回，下一秒又脏。
3. `git reset --hard` 是终极武器，但**也只在 JupyterLab 关掉后才稳定有效**。
4. 跨实例迁移项目时 `nbstripout --install` 是 0 成本卫生措施，建议加进 onboarding checklist。

### 30. SFT 训练时 loss 看起来"太低"（0.01-0.1）不是 bug，是 prompt masking 的正常表现

**实测**（2026-05-12 SFT v2 启动，step 1-35）：

| step | loss | grad_norm | lr | epoch |
|---|---|---|---|---|
| 1  | 0.1146 | 3.79 | 8.3e-06 | 0.004 |
| 5  | 0.0967 | 1.64 | 4.2e-05 | 0.019 |
| 15 | 0.0221 | 0.21 | 1.3e-04 | 0.058 |
| 25 | 0.0187 | 1.03 | 2.0e-04 | 0.096 |
| 35 | 0.0105 | 0.07 | 2.0e-04 | 0.135 |

第一反应："起点 0.11 太低了，是不是 loss masking 出问题？"

**实际是正常**。原因：

1. **SFT 只在 assistant 响应上算 loss**（system / user 的 ~1500 prompt tokens 全部 `loss_mask=0`，不参与 CE）
2. **响应 = `LABEL ##[i,j]##` ≈ 5-10 tokens**，其中 4 个 LABEL token 是 SUPPORTS/REFUTES/NOT_ENOUGH_INFO/DISPUTED（base 模型本来就认识）
3. 加上 `--loss_scale ignore_empty_think` 再 mask 掉空的 `<think></think>` 块 → 有效计算 token 更少
4. base Qwen3.5 在给定 prompt context 下，对 LABEL token 的 prior probability 已经很高（4 选 1）→ 即使没训过，CE loss 也只有 ~0.5 量级
5. 训了 25 step 之后 LoRA 已经学到 `LABEL ##[..]##` 这个固定输出格式 → loss < 0.05 是模型 99%+ confidence 输出格式正确

**判定 SFT 是否真在学的正确方式**：

- **不要看 loss 绝对值**（会被 masking 误导）
- ✅ **看 Track 3 vs Track 2 的 acc / HM delta**（端到端指标才是真实信号）
- ✅ 看 **NEI 类的 per-label acc**（base 0.025 → SFT 目标 ≥ 0.40，硬约束 §0.5.3.1）
- ✅ 看 **non-NEI 类是否回退**（不应该从 0.580 跌到 0.40 以下，否则 SFT over-corrected）

**复用要点**：

1. **SFT loss 的"绝对值参考线"几乎没意义**，因为受 prompt/响应长度比、loss_mask、loss_scale 多重影响。永远用端到端任务指标判定。
2. **第一次跑某个 SFT 配置时记下 step 1 / 5 / 25 / 100 的 loss 数字作为"健康指纹"**。再跑相同配置时偏离 > 50% 才需要警觉。
3. **grad_norm 是 loss 的搭档指标**：维持 < 5 + 不发散即可；忽然 → 0 是 LoRA 已收敛或学不动，忽然 → 100 是数值不稳。
4. 别把 loss 和 perplexity 混淆。SFT loss = per-token CE on UNMASKED tokens；perplexity 要 exp(loss)，但 masking 让 perplexity 也失真。

### 31. ms-swift VL adapter 加载到 AutoModelForCausalLM 失败 — 走 `swift export --merge_lora` 而不是 peft

**症状**（2026-05-12 SFT 训完跑 Track 3 evaluation 时）：

第一次报错：

```
ValueError: Target modules ^(model\.language_model(?=\.).*\.(gate_proj|in_proj_qkv|
  o_proj|in_proj_b|up_proj|q_proj|down_proj|k_proj|in_proj_z|in_proj_a|
  v_proj|out_proj))$ not found in the base model.
```

试换 `AutoModelForImageTextToText` 加载 VL wrapper，没报错但：

- adapter 装上后 LoRA params 显示 `0.0M`（实际 0 也可能是 < 1M）
- Track 3 输出跟 Track 2 字节级相同（HM/F/Acc 完全一样 → LoRA 没生效）
- **Track 2 baseline 还掉了**：HM 0.203 → 0.133，速度从 1.6 s/it → 5.4 s/it

**根因**：

| 组件 | ms-swift 训练时 | 我们的 phase1_eval 加载时 |
|---|---|---|
| 模型类 | `Qwen3_5VLForConditionalGeneration`（VL wrapper） | `Qwen3_5ForCausalLM`（纯 LM）|
| 模块结构 | `model.language_model.layers.X.self_attn.q_proj` | `model.layers.X.self_attn.q_proj` |
| adapter regex | 含 `model.language_model.` 前缀 | regex 不匹配 |
| adapter state_dict keys | 含 `language_model.` | inject 后的 key 不含，load 失败 |

ms-swift 把 Qwen3.5 当 VL 模型加载训练，target_modules regex 和 LoRA state_dict 都烤死了 `language_model.` 前缀。换 `AutoModelForImageTextToText` 加载 VL wrapper 理论上结构对了，**但 transformers 5.2 的 `Qwen3_5ForConditionalGeneration` 跟 ms-swift 训练时的 VL 类**可能不是同一个内部布局（生成路径也不同 → Track 2 速度变 3 倍慢、数字变差）。

**修复 — 不修 phase1_eval 的加载路径，而是把 LoRA 合并进 base**：

```bash
# 在 AutoDL 上
source /etc/network_turbo
swift export \
    --adapters /root/autodl-tmp/nlp_a3_cache/sft-out/checkpoint-final \
    --merge_lora true \
    --output_dir /root/autodl-tmp/nlp_a3_cache/sft-out/merged
# 产生一个完整的 Qwen3.5-4B + SFT 权重已 baked 的 base model（~8 GB）

# 然后 phase1_eval 用 --sft-merged-dir 加载它（走 AutoModelForCausalLM 普通路径）
python -m scripts.phase1_eval --tracks 2,3 --prompts v1 --dataset diag_test \
    --sft-merged-dir /root/autodl-tmp/nlp_a3_cache/sft-out/merged
```

**`phase1_eval.py` 改动**：

- 加 `--sft-merged-dir PATH` flag，跟 `--sft-adapter` 互斥
- Track 3 时优先用 `--sft-merged-dir`（走干净的 AutoModelForCausalLM）
- `--sft-adapter` 保留给非 VL 模型用；遇到 < 1M LoRA params 时 warn 指向 `--sft-merged-dir`
- 加载器**仍用 `AutoModelForCausalLM`**（不再走 AutoModelForImageTextToText，避免 Track 2 baseline 漂移）

**复用要点**：

1. **ms-swift 训练 + peft load 的 model class 对齐问题 90% 是底层 VL/LM wrapper 不匹配**。诊断方法：训练前后分别 print `type(model).__name__` + sample module paths，看是否一致。
2. **不要硬凑 model class，用 merge_lora 旁路最干净**。代价是磁盘多 8 GB merged 模型，但部署 / eval 时不再担心 adapter 加载。
3. **`PeftModel.from_pretrained` 不报错不代表 LoRA 已应用** —— peft 默认 `strict=False`，state_dict load 不上的 key 静默跳过。判定方法：
   - 数 LoRA 参数：`sum(p.numel() for n,p in m.named_parameters() if 'lora_' in n)`，应 > 1M（典型 r=16 在 4B 上 ~32M）
   - 跑两段输出 diff：Track 2 和 Track 3 字节级相同 → 适配器没生效
4. **`AutoModelForImageTextToText` 改 model class 会改 generation 路径** —— Track 2 baseline 也会跟着变（实测 −7pp HM）。不要为了 adapter 兼容性牺牲 baseline。

### 32. SFT 数据 class-imbalance collapse — 训练前必看 label distribution

**症状**（2026-05-12 Track 3 首次评估）：

| | Track 2 v1 (base+RAG) | **Track 3 v1 (SFT+RAG)** | SFT 效果 |
|---|---|---|---|
| predicted NEI 占比 | 24.8% (≈ gold 33%) | **92.6%** | over-correct ×3.7 |
| NEI acc | 0.350 | **0.975** | +63pp（学得太好）|
| non-NEI acc | 0.580 | **0.062** | **−52pp**（完全崩盘）|
| HM | 0.201 | **0.140** | **−6pp** |

SFT 训完后 Track 3 比 Track 2 baseline 还低 0.06 HM。模型本质学到了"看到任何 evidence 都先猜 NEI"。

**根因 — 训练 label 分布塌缩到主导类**：

v2 训练数据 4166 条里的 NEI 来源：

| 来源 | 数量 | 占比 |
|---|---|---|
| `nei_underspec` real × **4 (weak_buckets)** | 1212 | 29% |
| 所有类的 `hard_neg = 1` (synth NEI 标) × scale factor | 2083 | **50%** |
| 其他真实 NEI ev | 0 | 0% |
| **合计 NEI** | **3295** | **79.1%** ← 远超 gold 33% |

`n_hard_neg=1` 是隐形罪魁：每个 real 样本配一个 hard-neg（labeled NEI），weak_buckets 又给 nei_underspec ×4，nei_underspec 的 hard-neg 也 ×4 → 双重放大 NEI。

模型在 SGD 下，对 majority class 的 likelihood 最大化策略就是"全猜 majority class"。Track 3 NEI 97.5% acc 是"训练正确"，但本质是 dataset bias 的胜利，不是任务能力。

**修复 — 重平衡（v2 revision，2026-05-12 PM）**：

```python
# src/build_stage0.py _TRAIN_WEAK_BUCKETS
_TRAIN_WEAK_BUCKETS = {
    ("scenario", "nei_underspec"): 2,       # 4 → 2 (real NEI 减半)
    ("scenario", "disputed_conflict"): 3,   # 2 → 3 (DISPUTED 仍弱，加强)
    ("scenario", "refutes_clear"): 2,       # 不变
}
# train config 也改：
n_hard_neg = 0   # 关键：去掉 hard-neg 同义重复
```

实测重建后分布：

| label | v2-old | **v2-new** | gold |
|---|---|---|---|
| SUPPORTS | 10.4% | 27.6% | 31.4% |
| REFUTES | 6.2% | 16.5% | 18.2% |
| NEI | **79.1%** | **38.7%** | 33.1% |
| DISPUTED | 4.3% | 17.2% | 17.4% |

NEI 从 79.1% → 38.7%（接近 gold 33% + 适度 oversample）。总记录 4166 → 1567 (× 2.7 缩小)，训练时间 4h → ~1.5h。

**Sanity check 写进 `src/build_stage0.py`**：每个 split build 完打印 SFT vs gold label distribution + ratio，**ratio > 2× 或 < 0.5× warn**。这次重建：所有 ratio 在 [0.63, 1.89] 区间，安全。

**复用要点**：

1. **SFT data class balance 是 #1 杀手**。majority class 占 > 60% 几乎必塌。Build time 就要做 distribution check，别等 SFT 训完 4h 才发现 Track 3 比 Track 2 还差。
2. **`n_hard_neg=1` 在已有强 oversample 的情况下会"双倍放大" majority class**。两个机制叠加要看总和。
3. **诊断 class-collapse 的 1-line 指标**：`predicted NEI share` 远 > `gold NEI share` 且 `non-NEI acc < 0.10` → 模型在赌 majority。
4. **重建数据后 sanity check 是必须**：让 `build_stage0` 输出"vs gold split"的对比表，每个 label 的 ratio 一眼看见。
5. **Hard constraint 1（NEI oversample）的最低剂量**：nei_underspec ×2（real）+ disputed ×3 提供足够的"识别 off-topic ev"信号，不需要 hard_neg synth。

### 33. DataLoader worker SIGABRT 中途崩 — save_steps 缩短 + num_workers=0 兜底

**症状**（2026-05-12 PM SFT v5 重训）：

```
Train:  43%|████▎     | 127/294 [47:42<1:02:44, 22.54s/it]
RuntimeError: DataLoader worker (pid 30933) is killed by signal: Aborted.
DataLoader worker (pid(s) 30933) exited unexpectedly
```

step 127/294 跑了 47 分钟，离首个 checkpoint (step 200) 还差 73 步 → **0 ckpt 可恢复，要从头**。

**诊断**（用户实测）：

| 资源 | 状态 |
|---|---|
| 磁盘 | 39/50 GB used, 12 GB free, 78% | 紧但够（写盘需求小）|
| RAM | 64/503 GB used, **431 GB available** | 完全不是 RAM 问题 |
| 失败 run dir | **144 KB** | 训练时根本没写盘出大文件 |

→ **不是 OOM 也不是磁盘**（OOM 会 SIGKILL 不是 SIGABRT；磁盘 12 GB 够当时写）。

**SIGABRT 的常见来源**（DataLoader worker subprocess 内）：

1. **Rust tokenizer fast 实现 panic** —— 单条样本里有罕见 unicode / 大 token id 会让 `tokenizers` 库内部 `assert!` 触发。`abort()` 抛 SIGABRT。
2. **C 扩展随机崩**（bitsandbytes / liger / fla 个别 op 在某些 dtype 边界条件下）。
3. **CUDA worker subprocess 状态污染**（DataLoader fork 时 CUDA context 有时坏，但 prefetch=2 缺省下少见）。

无法事先复现 / 复盘单一原因。**做防御性配置兜底**：

**修复（已落到 `notebook_autodl.ipynb` sec2-5-train，复用经验 33）**：

```python
"  --save_steps 50 --eval_steps 50 --save_total_limit 3",   # 200 → 50
"  --dataloader_num_workers 0",                              # 新增
```

- `--save_steps 200 → 50`: 崩了最多丢 50 步 ≈ 19 min，不再丢 47 min（覆盖 step 127 的崩盘场景）。
- `--dataloader_num_workers 0`: 强制主进程做数据加载，**消灭 worker subprocess 失败模式**。代价 ~5-10% 训练慢（小数据集场景几分钟差距）。

**复用要点**：

1. **SFT 训练默认 `save_steps` 不要太大**。多数 ms-swift 教程给 200/500/1000 是为大数据集做的；我们 1567 records × 3 epochs = 294 steps，`save_steps 200` 意味着前 67% 的训练没有任何安全网。**按 `save_steps ≤ total_steps / 5` 选**。
2. **`dataloader_num_workers=0` 是稳定性保险**：小数据集 + GPU 是 bottleneck 的场景，多 worker 没明显加速但增加 transient crash 面积。Phase 5 SFT 之后再调回去（DPO / 大数据集才有必要）。
3. **SIGABRT vs SIGKILL 区分**：SIGABRT = 内部 abort（C 代码 assert / panic），SIGKILL = OOM killer。前者改代码 / 改 worker 配置；后者降 batch size / 升 RAM。
4. **transient crash 不一定复现**：第二次跑同样配置同样数据，大概率不崩同一个 step。所以**别试图二分定位**，先加防御性配置（save_steps 小 + 单进程）让训练能跑完。

### 34. SFT class-collapse 不是单纯 class-imbalance — recall ceiling 才是根因

**症状**（2026-05-12 PM SFT v3-rebalanced + merged_v3，从 v5-180014/checkpoint-294 烤出）：

| | Track 2 v1 baseline | **v2-cut-1 (broken)** | **v3-rebalanced** |
|---|---|---|---|
| 训练数据 NEI 占比 | — | 79.1% | **38.7%**（×2 大改） |
| predicted NEI 占比 | 24.8% | 92.6% | **94.2%** ← 反而更差 |
| NEI acc | 0.350 | 0.975 | ~0.975 |
| non-NEI acc | 0.580 | 0.062 | **0.049** |
| 总 HM | 0.201 | 0.140 | **0.140** |

复用经验 32 的修复（`n_hard_neg=0` + `nei_underspec ×2`）**完全没用**。把训练 NEI 占比从 79% 砍到 38.7%，模型 inference 仍然 94% 预测 NEI。

**Lesson**：**class-imbalance 不是根因**。我们只解决了"训练数据看起来不偏"，但模型依然学到 NEI 默认行为。问题在更深的层级。

**真根因（强假设）—— 训练 / 推理分布不匹配，根源是 retrieval 的低 recall**：

训练时模型看到的 evidence 来源：

| Claim 类型 | 训练时 ev = ? |
|---|---|
| NEI (nei_underspec) | 5 条 real gold（这些 gold "看起来不沾边"）|
| SUPPORTS / REFUTES / DISPUTED | 1-5 条 real gold + 15-19 条 **random padding** (`pad_with_random=True`) |

`pad_with_random=True` 让所有非 NEI 样本的 evidence 实际上是"少数 gold + 大量 random ev"。模型学到的判别 boundary 是：

```
"看到 5 条都不沾边" → NEI
"看到 5 条中有 ≥1 条沾边" → SUPPORTS / REFUTES / DISPUTED
```

推理时 RAG 给出的 top-20 ev：

| 维度 | 训练 (build_dataset) | 推理 (real RAG) |
|---|---|---|
| Recall (gold 在前 20 中的比例) | NEI: 5 gold / SUPPORTS: 1-5 gold | macro @ k=20 = **0.333**（Phase 3.5 实测）|
| 平均 gold 数 | NEI=5 / 其他=2-3 | **~0.7** (= 2 gold × 33% recall) |
| 噪声 ev 占比 | NEI=0% / 其他=75-95% | **~96%** (= 19 noise / 20 total) |

**推理时 RAG 看起来 ≈ 训练时的 NEI 样本**（绝大多数 ev 不沾边）。模型按训练时学到的 boundary 判：**NEI**。

这不是 class imbalance 的问题，是 **representation alignment** 问题。**recall 拉不上去，SFT 永远塌**。

**为什么 rebalance 没用**：

`n_hard_neg=0` 删了 hard-neg 那 2083 条 synth NEI，但**没改变非 NEI 样本的 pad_with_random=True 行为**。所有 SUPPORTS/REFUTES 样本仍然是"少数 gold + 多数 random"。boundary 没变，结果没变。

**Lesson**：

1. **class balance 是诊断信号不是根因**。看到 92% predicted NEI 别只想到"训练数据 NEI 多"，要追问"推理时 evidence 分布跟训练哪类样本最像"。
2. **`pad_with_random=True` 是 silent killer**：它让训练分布跟低 recall 推理时高度重合，模型学到的捷径就是 "evidence 模糊 → NEI"。
3. **SFT 的前置条件是 retrieval recall 足够高**。当 recall@k 太低时（< 0.5），retrieved evidence 跟随机 ev 几乎不可区分，SFT 学不到 label↔evidence 的 semantic mapping。
4. **战略转向 retrieval-first**（design D-019，2026-05-12 PM）：先把 recall@20 优化到 ≥ 0.5，再回头训 SFT。或者 SFT 改用 gold-only 数据（`pad_with_random=False`）避开训练时引入噪声。

**复用价值**（对未来项目）：

- **训练 RAG SFT 之前先量化 retrieval-inference gap**：跑 `diagnose_phase1` 看 evidence recall macro。若 < 0.5，SFT 风险大。
- **`pad_with_random` 这类"凑数 evidence"的 dataset 设计要谨慎**：让训练 evidence 分布跟推理一致才有效。否则要么不 pad（gold-only），要么提高 recall 让推理也"干净"。
- **rebalance class weight 解不了 representation alignment**：调整 label 占比是表面治标，调整 input distribution（recall / padding strategy）才是治本。

### 35. bge-reranker-base 在 climate 域是**负贡献** — Phase 3.5b 实测翻盘

**症状**（2026-05-12 PM Phase 3.5b 检索深度审计）：

`scripts.retrieval_ceiling --mode retriever` 输出震惊：

| config | recall@5 | recall@10 | recall@20 | recall@50 | recall@100 |
|---|---|---|---|---|---|
| BM25 only | 0.136 | 0.185 | 0.263 | 0.340 | 0.393 |
| dense only | 0.170 | 0.235 | 0.319 | 0.444 | 0.541 |
| **fused (no rerank)** | **0.200** | **0.273** | **0.360** | 0.485 | 0.579 |
| full (fused + **rerank**) ← 之前 baseline | 0.119 | 0.210 | 0.333 | 0.485 | 0.579 |

**关掉 cross-encoder reranker → recall@5 从 0.119 → 0.2003，×1.68 提升**。

**Lesson**：

- **fused (no rerank)** 全 k 上都比 full pipeline 强或持平
- k=5 / k=10 提升最显著（fused 的 raw 排序就是对的，reranker 反而把对的往后推）
- k ≥ 50 reranker 不再影响 recall（top-50 集合相同，只重排序内部）— 所以 reranker 唯一作用是**把对的往前排，但它推错了方向**

**根因（推测，未深查）**：

`BAAI/bge-reranker-base` 训练时见的语料以新闻 / wiki / 百科为主，对 **climate science 域的 specialized vocabulary**（GHG、albedo、CMIP6 模型名、IPCC 报告引用）不熟。reranker 给"看起来更通用语义相关"的段落打高分，把含专有名词的真正 gold ev 推后。

**修复（已落到 src/retrieval/pipeline.py + scripts/phase1_eval.py）**：

- `RetrievalConfig.use_rerank` default `True → False`
- `phase1_eval.build_pipeline` 加 `use_rerank: bool = False` 参数
- `phase1_eval --rerank` opt-in flag（默认 off，开启时输出加 `_rerank` 后缀防覆盖 production 表）
- 不删 `src/retrieval/rerank.py` 也不删 `models/bge-reranker-base/`：保留 ablation 价值

**Track 2 baseline 预期改善**（待 AutoDL 实测）：

```
recall@5  0.119 → 0.200  (+0.081, ×1.68)
F-score   0.135 → ~0.18  (估算)
HM        0.201 → ~0.22-0.24
```

只关一个开关白拿 +0.02 HM。

**复用要点**：

1. **跨域 reranker 不一定有用**。bge-reranker-base 在 MS MARCO / wikipedia 域好，但 climate 这种专业术语密集场景可能负贡献。**先 ablation 再用 reranker**。
2. **`fused (no rerank)` 应该是任何 BM25+dense pipeline 的 baseline 之一**。reranker 是 +1 个组件，跟 ablation 表对比；不是默认必须的。
3. **reranker 影响只在 small k 显著**：k 越大它的相对作用越小，因为它只重排不增删 top-N candidate set。
4. **报告 Results 章节金牌**："Reranker hurts recall@5 by ×1.68 on climate domain — domain-specific vocabulary mismatch with bge-reranker-base's training distribution. We disable it." 这种 unexpected ablation finding 比"reranker improves F by +X"更有 publishable value（README §307）。

### 36. HyDE/sub-claim multi-query 在 top-20 没用，深度 recall 有效 — Phase 3.5b llm_rewrite mode 实测

**症状**（2026-05-13 Phase 3.5b LLM rewrite audit）：

跑 `scripts.retrieval_ceiling --mode llm_rewrite --no-rerank` 完整 recall@k 曲线：

| base config | r@5 | r@10 | r@20 ← SFT 用 | r@50 | r@100 |
|---|---|---|---|---|---|
| **baseline (claim only)** | **0.201** | **0.300** | **0.357** | 0.467 | 0.558 |
| HyDE only | 0.164 | 0.270 | 0.357 | 0.506 | 0.602 |
| sub-claims only | 0.151 | 0.224 | 0.331 | 0.479 | 0.583 |
| **HyDE + sub-claims** | 0.193 | 0.248 | 0.339 | **0.511** | **0.616** |

**关键观察**：

1. **recall@20 上 HyDE+sub-claims = 0.339，baseline = 0.357**。HyDE 不能在 SFT context length 内提升 recall。
2. **recall@50 / @100 HyDE+sub-claims +0.044 / +0.058 over baseline**。LLM rewrite 引入的 gold ev 真的存在，但排名都落在 20-100 之间。
3. **HyDE only 在 recall@20 持平 baseline，recall@50 +0.039 over baseline** —— HyDE 是 multi-query 的主要贡献来源，sub-claim 反而单独用时退步。
4. **sub-claims only 在所有 k 都是 4 个 config 里最差**。`parse_subclaims` 倾向于把 claim paraphrase 成几乎同义的 atomic 句，没引入新关键词。

**报告价值**：经典 HyDE 文献（Gao et al. 2022, MS MARCO / NQ）讲的是"HyDE 提升 top-5"，但**我们这里 HyDE 的浅层 precision 反而下降，深层 recall 上升**。Domain shift（climate vs MS MARCO）+ retrieval system 本身 baseline 强（fused-no-rerank 已经做了 80% 的事）共同导致 HyDE 的 marginal benefit 集中在 long tail。

**结论 — retrieval-first 战略转向部分失败**：

- recall@20 触到 0.357 上限，**无论什么 retrieval 配置都过不去**
- 选 final_k=50 + HyDE+sub-claims 能拿 r@50 = 0.511，但 prompt 长度翻倍 → SFT 时间 4h、VRAM 边缘、信息密度悖论加剧
- LLM rewrite 在深度 recall 有真实贡献，**留作 Phase 6 official_dev / test 预测时的可选 inference enhancement**（推理时 retrieval 用 multi-query，SFT 训练时不用）

**改换战术 — 复用经验 36（本条）的主修复路径**：保留 retrieval baseline (final_k=20, recall@20=0.357)，**改 SFT 训练数据 `pad_with_random=False`** 让训练分布跟推理一致：

```python
# src/build_stage0.py step_sft train kwargs
dict(k=20, pad_with_random=False,    # ← True → False
     n_hard_neg=0, apply_curriculum=True,
     weak_buckets=_TRAIN_WEAK_BUCKETS)
```

效果（本地实测 2026-05-13）：

| | 旧 (pad_with_random=True) | **新 (pad_with_random=False)** |
|---|---|---|
| 总记录 | 1567 | 1567 (label distribution 不变) |
| n_shown 分布 | 全部 = 20 (强制 pad) | 1: 14% / 2: 18% / 3: 14% / 4: 9% / 5: 45% |
| 非 NEI 样本 evidence | 1-5 gold + 15-19 random (~96% noise) | **只 1-5 gold (0% noise)** |
| NEI 样本 evidence | 5 gold (off-topic) | 5 gold (off-topic, 不变) |

**为什么这个改动可能救 SFT**（复用经验 34 的 representation alignment 视角）：

旧训练时模型看到 "1-5 gold + 多 noise" → 学到 "noise 多就 NEI"。推理时 recall@20 = 0.357 → 真实 ev ≈ 67% noise → 触发 NEI shortcut。

新训练时 **非 NEI 样本完全无 noise**，模型必须用语义判别 label（不能靠 noise 比例）。NEI 样本则是 5 条全 off-topic gold ev。两类语义边界拉开。

**风险**：训练 vs 推理仍不完全对齐（训练 ≤ 5 ev，推理 20 ev）。如果模型只学到"1-5 ev → 走 label，更多 ev → NEI"则仍会塌（但相比旧版本 noise 比例不同，比"96% noise vs 67% noise"差别小）。

**Track 3 v6 目标**（待 AutoDL 实测）：

- HM > 0.213（当前 Track 2 v1 baseline，免费 +0.030 from no-rerank lock）
- NEI acc 不再 0.97 极端，0.30-0.60 健康区间
- non-NEI acc 不再 0.05 崩盘，≥ 0.45

**如果还塌**：retrieval ceiling + SFT representation alignment 都试过了。**接受 Track 2 v1 HM 0.213 作为 final 提交**，SFT 实验全留作 ablation。报告 Results 章节按 README §307 narrative：3 个失败 story（prompt → retrieval → SFT alignment）。

**复用要点**：

1. **HyDE 价值在 long-tail recall，不在 top-5 precision** —— domain 偏离 MS MARCO 越远越如此
2. **训练分布对齐 > 类平衡 > 类指令** ：复用经验 21 (prompt 无效) + 32 (class rebalance 不够) + 36 (本条：alignment 才是关键)
3. **每次 SFT 改 padding/sampling 策略时，build_stage0 输出的 `_print_label_dist` + n_shown 分布都要看** —— label 分布同时 input 分布也要检查
4. **试到 retrieval recall 真的不能动了，再回头改训练数据格式** —— retrieval-first → data-format-second 是合理顺序，反过来会浪费迭代




