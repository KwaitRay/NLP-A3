# 部署与操作 Runbook

> 第一次上手这个项目？按本文档从头跑到尾即可。每一步都给：**做什么 → 命令 → 调用脚本 → 产出 → 意义 → 预计时间**。
>
> 配套阅读：技术设计 `design.md`，方案 `~/.claude/plans/fancy-mapping-lemur.md`，进度日志 `outputs/PROGRESS.md`。

---

## 总览

| 阶段 | 在哪跑 | 耗时 | 关键产出 |
|---|---|---|---|
| Phase 0 — 环境准备 | 本地 | 5 min | venv + 依赖装好 |
| Phase 1 — 本地数据准备 | 本地 | 2 sec | SFT 训练数据 1972 条 |
| Phase 2 — 自检 | 本地 | 3 sec | dry_run_report.md exit 0 |
| Phase 3 — 上传 Colab | 浏览器 | 10 min | Drive 上有完整项目 |
| Phase 4 — Colab 索引 | Colab T4 | 35-65 min | BM25 + bge-m3 索引（缓存到 Drive） |
| Phase 5 — Colab SFT | Colab T4 | 75-100 min | LoRA adapter |
| Phase 6 — Colab DPO（可选） | Colab T4 | 25 min | DPO adapter |
| Phase 7 — Colab 推理 | Colab T4 | 5-10 min | dev / test 预测 JSON |
| Phase 8 — 本地汇总 | 本地 | 1 min | ablation_report.md |
| Phase 9 — 提交打包 | 本地 | 5 min | submission zip + PDF |

**总计 ~3-4 小时**（不含 Colab session 间断恢复时间）。

---

## Phase 0：环境准备（本地，~5 min）

### 0.1 克隆仓库 + 切到项目根目录

```bash
git clone <your-repo-url> Assignment3
cd "Assignment3"
```

如果已经在项目里就 `cd` 过去：
```bash
cd "D:/学习/研究生/graduate 26s1/90042 NLP/Assignment/Assignment3"
```

### 0.2 创建虚拟环境

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 0.3 装本地依赖

```bash
pip install numpy pandas nltk transformers
```

**意义**：本地只需要 Stage 0 + 测试需要的轻量依赖。重型的 torch / bm25s / faiss / ms-swift 都装在 Colab 上，不污染本地环境。

---

## Phase 1：本地数据准备（~2 秒）

### 1.1 确认原始数据齐备

```bash
ls data/
# 期望看到：
#   train-claims.json (328 KB, 1228 条)
#   dev-claims.json (41 KB, 154 条)
#   test-claims-unlabelled.json (24 KB, 153 条)
#   dev-claims-baseline.json (50 KB, 格式参考)
#   evidence.json (174 MB, 1.2M 段)
#   evidence.md (链接说明)
```

如果 `evidence.json` 缺失，从下面任一链接下载放到 `data/evidence.json`：
- Google Drive: https://drive.google.com/file/d/1JlUzRufknsHzKzvrEjgw8D3n_IRpjzo6/view
- Canvas LMS: https://canvas.lms.unimelb.edu.au/courses/234957/pages/evidence-dot-json-download

### 1.2 一键生成 SFT 训练数据

```bash
python -m src.build_stage0
```

**调用脚本**：`src/build_stage0.py`（依次调 `src/eda.py` → `src/stage0_tag.py` → `src/splits.py` → `src/sft_dataset.py`）

**产出**：
```
outputs/
├── eda/eda_report.md                       # 数据画像统计
├── splits/
│   ├── train_split.jsonl    (986 条)        # SFT 训练源
│   ├── dev_holdout.jsonl    (121 条)        # SFT 早停 + DPO 偏好对来源
│   ├── diag_test.jsonl      (121 条)        # 诊断切片用，永不污染选型
│   ├── official_dev.jsonl   (154 条)        # 官方 dev 镜像（带 tag）
│   └── split_summary.md
└── sft_data/
    ├── claims_tagged.jsonl  (1382 条)       # 全量 train+dev + 三维 tag
    ├── tag_distribution.md                 # tag 分布交叉表
    ├── sft_train_v1.jsonl   (1972 条)       # ms-swift SFT 训练数据
    ├── sft_dev_holdout_v1.jsonl (121 条)    # ms-swift validation
    └── sft_diag_test_v1.jsonl   (121 条)    # 诊断推理用
```

**意义**：完成 Stage 0 全部工作。这一步在本地跑 = 占 Colab 时间的 0%；并且因为是确定性的（`md5(salt||id) % 10`），跨会话重跑结果完全一致。

**重跑**：默认 idempotent，文件存在则跳过。强制重建：
```bash
python -m src.build_stage0 --force
```

---

## Phase 2：本地自检（~3 秒）

### 2.1 跑 dry-run

```bash
python -m scripts.dry_run
```

**调用脚本**：`scripts/dry_run.py`

**做什么**：
1. 检查 `data/evidence.json` 是否存在且 > 100 MB
2. idempotent re-run Stage 0（应全部 [skip]）
3. 检查 10 个关键 artifact 文件存在 + 非空
4. 检查 split 总数 = 1228（防止 hash 切分逻辑回归）
5. 试导入 retrieval 模块（验证类签名正确）
6. stub-mode 推理 + ablation 跑 275 条合成预测
7. 跑全部 8 套单元测试

**产出**：`outputs/dry_run_report.md`，应看到 `✅ All local checks passed` + exit 0。

**意义**：上 Colab 前最后一道防线。如果本地有 bug，这里会暴露；exit 0 = "可以推 Colab 了"。每次改了 `src/` 任何代码后都该重跑一次。

---

## Phase 3：上传 Colab（~10 min）

### 3.1 把项目同步到 Google Drive

把整个项目目录传到 Drive 的 `MyDrive/comp90042/`：

```
MyDrive/comp90042/
├── data/                    # 含 evidence.json 174 MB
├── src/                     # 代码包
├── outputs/                 # Phase 1 产出（Colab 会复用 sft_data/）
├── notebooks/notebook.ipynb # 主 notebook
├── eval.py
└── design.md / DEPLOY.md / README.md
```

**推荐方式**（任选一种）：
- **Google Drive 网页拖拽** —— 最直观，但 174 MB evidence.json 上传慢
- **rclone**（推荐）：本地装 [rclone](https://rclone.org/)，一行同步：
  ```bash
  rclone sync . gdrive:comp90042 --exclude .git/** --exclude .venv/** --exclude swxy/** --progress
  ```
- **gdown** 反向：把 evidence.json 单独传一次，之后只 sync 代码

### 3.2 在 Colab 打开 notebook

1. 浏览器开 [colab.research.google.com](https://colab.research.google.com)
2. **File → Open notebook → Google Drive → `MyDrive/comp90042/notebooks/notebook.ipynb`**
3. **Runtime → Change runtime type → T4 GPU**
4. **Runtime → Run all**（或下面分阶段手动跑）

**意义**：Colab 是免费 T4 GPU 的唯一渠道（符合规范 §2.7）。Drive 挂载让后续会话不必重传数据。

---

## Phase 4：Colab 索引建立（~35-65 min，**一次性**）

### 4.1 运行 §0 Setup cells

3 个 cell 依次：
- 检测 IS_COLAB → 设 `PROJECT_ROOT = /content/drive/MyDrive/comp90042`
- mount Drive
- `pip install ms-swift modelscope transformers peft trl bitsandbytes accelerate sentence-transformers bm25s faiss-cpu spacy nltk pystemmer`
- set seeds + GPU sanity

**意义**：每次新 session 必跑（Colab 不持久化已装的 pip 包）。Drive mount 让 `outputs/` 持久化。

### 4.2 运行 §1 Data Loading + EDA

会自动 load `data/evidence.json`（~1.5 秒）和重跑 Stage 0（~2 秒，应全部 [skip]）。

**意义**：把数据加载 + EDA 报告 + Stage 0 数据全部 ready。

### 4.3 §2.1 BM25 索引（~2-4 min）

```python
# 在 cell 里执行
bm25 = BM25Retriever()
bm25.build(evidence, save_dir=BM25_DIR)
```

**调用类**：`src/retrieval/bm25.py:BM25Retriever`

**产出**：`outputs/bm25_index/{bm25, ev_ids.txt}`（~50 MB）

**意义**：稀疏检索第一阶段。建一次后所有后续 session 都从 cache 加载（< 1 秒）。

### 4.4 §2.2 Dense embedding（~30-60 min）

```python
dense = DenseRetriever(model_name="BAAI/bge-m3")
dense.build(evidence, save_dir=DENSE_DIR, batch_size=128)
```

**调用类**：`src/retrieval/dense.py:DenseRetriever`

**产出**：`outputs/dense_index/{faiss.index, ev_ids.txt, meta.json, chunks/*.npy}`（~5 GB）

**意义**：1.2M 段 × 1024-d float32 embedding。**这是整个项目最长的一次性计算**。每 50k 段 checkpoint 一次 → 即使 session 12h 超时，下次 session 从中断点续跑。

**显存紧张时退路**：换 `LIGHT_MODEL = "BAAI/bge-small-en-v1.5"`（384-d，30 min 内可跑完）。

### 4.5 §2.3-2.4 融合 + 重排 + k-sweep（~10 min）

```python
reranker = CrossEncoderReranker()  # bge-reranker-base, 自动下载
pipeline = RetrievalPipeline(evidence, bm25, dense, reranker, cfg=RetrievalConfig(final_k=5))

# k-sweep on dev
recall_curve(retrieved, dev, ks=(1,3,5,10,20,50,100,200))
```

**调用类**：`src/retrieval/{rerank.py, pipeline.py}`

**产出**：dev 上 k-sweep 数据（用于报告 figure 1-2）

**意义**：定 final_k 这个超参（默认 4 + NEI=5）。**只在 dev 上调，绝不碰 test**。

---

## Phase 5：Colab SFT（~75-100 min）

### 5.1 §2.5 下载 Qwen3.5-4B

```python
from modelscope import snapshot_download
MODEL_DIR = snapshot_download("Qwen/Qwen3.5-4B-Instruct",
                              cache_dir="/content/drive/MyDrive/qwen3.5-4b")
```

**产出**：模型权重 ~8 GB 缓存到 Drive。

**意义**：ModelScope 是 Qwen 官方分发渠道（HuggingFace 可能滞后）。**走 ModelScope 不需要 HF 账号**。

**回退**：若 Qwen3.5-4B 还未发布，改 `Qwen/Qwen2.5-VL-3B-Instruct`（同族同尺寸）。

### 5.2 §2.5 SFT 训练

在 cell 里 `!{cmd}` 执行：

```bash
swift sft \
  --model {MODEL_DIR} --model_type qwen3_5_vl --use_hf false \
  --train_type lora --target_modules all-linear --freeze_vit true \
  --quantization_bit 4 --bnb_4bit_compute_dtype bfloat16 \
  --dataset {SFT_DIR}/sft_train_v1.jsonl \
  --val_dataset {SFT_DIR}/sft_dev_holdout_v1.jsonl \
  --output_dir /content/drive/MyDrive/sft-out \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 --warmup_ratio 0.03 --max_length 1024 \
  --gradient_checkpointing true --bf16 true \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --save_steps 200 --eval_steps 200 \
  --resume_from_checkpoint /content/drive/MyDrive/sft-out/last
```

**产出**：`/content/drive/MyDrive/sft-out/checkpoint-XXX/`（LoRA adapter ~50 MB）

**意义**：让 Qwen3.5-4B 学会按 `LABEL ##[1,3]##` 格式回答气候 fact-checking 任务。

**显存预算**：~11-13 GB 实测在 T4 16 GB 内安全。

**Session 超时**：每 200 step 存 adapter 到 Drive。下次 session 加 `--resume_from_checkpoint` 续跑。

---

## Phase 6：Colab DPO（可选，~25 min）

### 6.1 §2.6 生成偏好对

在 dev_holdout 上跑 SFT 模型 → 错预测构造 (chosen, rejected)：

```python
from src.dpo_pairs import build_dpo_dataset, synthesise_disputed_contrast, write_dpo_jsonl

# 1. 推理 dev_holdout
sft_inferer = ModelInferer(pipeline, sft_model, tokenizer)
preds = predict_all(devhold_claims, sft_inferer)

# 2. 构造偏好对
dpo_records = build_dpo_dataset(sft_records, preds, devhold_gold)
dpo_records += synthesise_disputed_contrast(sft_records, n=30)
write_dpo_jsonl(dpo_records, "outputs/dpo_data/dpo_pairs.jsonl")
```

**调用模块**：`src/dpo_pairs.py`

**产出**：`outputs/dpo_data/dpo_pairs.jsonl`（~50-80 条偏好对）

### 6.2 §2.6 DPO 训练

```bash
swift rlhf --rlhf_type dpo \
  --model /content/drive/MyDrive/sft-out/checkpoint-best \
  --train_type lora --quantization_bit 4 \
  --dataset outputs/dpo_data/dpo_pairs.jsonl \
  --beta 0.1 --learning_rate 5e-6 --num_train_epochs 1 \
  --max_length 1024 --max_prompt_length 896 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --output_dir /content/drive/MyDrive/dpo-out --bf16 true
```

**产出**：`/content/drive/MyDrive/dpo-out/checkpoint-XXX/`

**意义**：教模型纠正最常见的标签错误。**进度紧时第一个砍**——SFT 后直接跑 §2.7 推理也满足规范。

---

## Phase 7：Colab 推理 + 输出预测（~5-10 min）

### 7.1 §2.7 自一致性推理

```python
from src.inference import ModelInferer, predict_all

inferer = ModelInferer(
    retriever=pipeline, model=dpo_model, tokenizer=tokenizer,
    n_samples=5, temperature=0.7,
)

# Dev 预测（用于报告指标）
dev_preds = predict_all(dev, inferer, "outputs/dev-claims-predictions.json")

# Test 预测（用于 leaderboard）
test_preds = predict_all(test, inferer, "outputs/test-claims-predictions.json")
```

**调用模块**：`src/inference.py:ModelInferer + predict_all`

**产出**：
- `outputs/dev-claims-predictions.json`（154 条预测，eval.py 格式）
- `outputs/test-claims-predictions.json`（153 条预测）

### 7.2 §3.1 用 eval.py 校验

```bash
python eval.py --predictions outputs/dev-claims-predictions.json \
               --groundtruth data/dev-claims.json
```

**产出**：F-score / Accuracy / Harmonic Mean 三项指标，**这就是要写进报告的核心数字**。

### 7.3 §3.3 跑诊断切片（在 diag_test）

```python
diag_inferer = ModelInferer(retriever=pipeline, model=dpo_model, tokenizer=tokenizer)
diag_preds = predict_all(diag_claims, diag_inferer, "outputs/diag-test-predictions.json")
```

**意义**：拿 diag_test 上的预测才能跑诊断切片表（per-domain / per-scenario / per-difficulty）。

---

## Phase 8：本地汇总报告（~1 min）

把 Colab 跑出来的所有 prediction JSON 同步回本地（rclone 反向 sync 或手动下载）。

### 8.1 跑 ablation harness

```python
# 写一个汇总脚本（如 scripts/build_ablation.py），或在 notebook §3.2 cell 里跑
from src.ablation import AblationConfig, AblationHarness, quick_harness, DEFAULT_CONFIGS

h = quick_harness()
h.add(DEFAULT_CONFIGS[0], "outputs/preds/A1.json")  # BM25 + zero-shot
h.add(DEFAULT_CONFIGS[1], "outputs/preds/A2.json")  # + dense
# ... 一直加到 B3 (flagship=True)
report = h.render("outputs/ablation")
```

**调用模块**：`src/ablation.py`

**产出**：`outputs/ablation/ablation_report.md`，含：
1. 主 ablation 表（dev 上 9 个配置 × {F, A, HM}）
2. per-domain 切片（diag_test 上 8 domains × {F, A}）
3. per-scenario 切片（7 scenarios）
4. per-difficulty 切片（3 levels）
5. per-label 切片（4 labels）

**意义**：这 5 张表是报告的核心证据。

### 8.2 用 eval_helpers 复核

```python
from src.eval_helpers import score_predictions
import json
preds = json.load(open("outputs/dev-claims-predictions.json"))
gold = json.load(open("data/dev-claims.json"))
print(score_predictions(preds, gold))
# {'f_score': ..., 'accuracy': ..., 'harmonic_mean': ..., 'n': 154}
```

**意义**：与 `eval.py` 1e-15 精度一致，但能切片，方便 figure / table 用。

---

## Phase 9：提交打包（~5 min）

### 9.1 准备 PDF 报告

按 [ACL 模板](https://github.com/acl-org/acl-style-files) 写，**正文 ≤ 7 页**。

文件命名：`COMP90042_<teamname>.pdf`

### 9.2 准备 zip

```bash
# 把 notebook 改名成 Group ID
cp notebooks/notebook.ipynb COMP90042_<teamname>.ipynb

# 打包（**不带** evidence.json / checkpoint / embeddings）
zip -r COMP90042_<teamname>_resource.zip \
    COMP90042_<teamname>.ipynb \
    src/ \
    tests/ \
    scripts/ \
    eval.py \
    design.md DEPLOY.md README.md \
    -x "*/__pycache__/*"
```

### 9.3 复现性自检

提交前过 checklist：
- [ ] notebook 全 cell **clear all output → restart → run all** 不报错
- [ ] `python eval.py` 三个数与报告一致
- [ ] zip 体积 < 50 MB（不含模型 / 数据）
- [ ] `python -m scripts.dry_run` exit 0
- [ ] `pdf` + `zip` 两个文件命名规范

### 9.4 上传到 LMS

LMS 提交箱（28 April 2026 开放）→ 上传 PDF + zip。

---

## 故障排查

| 症状 | 排查 / 修复 |
|---|---|
| Colab 提示 "no GPU" | Runtime → Change runtime type → T4 GPU |
| `evidence.json not found` | 重新下载并放到 `data/evidence.json`，或检查 PROJECT_ROOT 路径 |
| `swift: command not found` | `pip install ms-swift modelscope` |
| OOM during SFT | (1) 检查 `--freeze_vit true`；(2) `--max_length 768`；(3) `--gradient_accumulation_steps 32` |
| OOM during dense embedding | 改 `LIGHT_MODEL = "BAAI/bge-small-en-v1.5"` |
| Colab session 12h 超时 | SFT 自动从 `--resume_from_checkpoint` 续；embedding 也按 chunk 续跑 |
| `eval.py` 报 KeyError | 预测 JSON 里少了 `claim_label` 或 `evidences`，检查 `predict_all` 输出 |
| Qwen3.5-4B ModelScope 找不到 | 改 `Qwen/Qwen2.5-VL-3B-Instruct`（同族 fallback） |
| ms-swift 不支持模型类型 | 切 Unsloth 后端（仅文本路径） |
| 测试套件红了 | `python -m tests.<test_name>` 单跑看哪个用例失败 |
| dry-run [fail] | 看 traceback，优先修复 import 错误，再看 artifact 缺失 |

---

## 时间预算

按"一次过"理想路径：

| 阶段 | 累计耗时 |
|---|---|
| Phase 0 (env) | 5 min |
| Phase 1 (Stage 0) | 7 min |
| Phase 2 (dry-run) | 7 min |
| Phase 3 (upload) | 17 min |
| Phase 4 (indexing, **一次性**) | 1h 22 min |
| Phase 5 (SFT) | 3h 02 min |
| Phase 6 (DPO) | 3h 27 min |
| Phase 7 (inference) | 3h 37 min |
| Phase 8 (ablation) | 3h 38 min |
| Phase 9 (submission) | 3h 43 min |

**真实预算预留 8-12 小时** 留 OOM 调试 / Colab 排队 / session 中断 buffer。**项目截止前留 24 小时** 跑最后一次 run all 防回归。

---

## 命令速查

```bash
# === 本地 ===
python -m src.build_stage0          # Phase 1: 一键 Stage 0
python -m scripts.dry_run           # Phase 2: 上 Colab 前自检
python -m tests.test_<name>         # 单跑某套测试

# === Colab ===
# Phase 4-7 在 notebook 里逐 cell 跑

# === 提交前 ===
python eval.py --predictions outputs/dev-claims-predictions.json \
               --groundtruth data/dev-claims.json    # 校验数字
zip -r COMP90042_<team>_resource.zip ...             # 打包
```

---

## 文档关系

| 文档 | 用途 |
|---|---|
| **`DEPLOY.md`**（本文件） | 操作 runbook，从克隆代码到提交 |
| `design.md` | 技术设计权威源（架构、决策记录、术语） |
| `~/.claude/plans/fancy-mapping-lemur.md` | 方案（plan-mode 双语） |
| `outputs/PROGRESS.md` | 跨 session 进度日志 |
| `outputs/dry_run_report.md` | 自检审计 trail |
| `README.md` | 课程规范（只读，不改） |
| `eval.py` | 官方评测脚本（只读，不改） |
