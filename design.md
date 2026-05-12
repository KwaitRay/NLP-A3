# COMP90042 Assignment 3 — Climate Fact-Checking RAG: 系统设计文档

> **Authoritative tech design.** 项目落地时优先读此文档；仓库中 `~/.claude/plans/fancy-mapping-lemur.md` 是 plan-mode 阶段的双语方案，本文档把决策固化为单一可执行设计。
>
> 中文为主语；技术标识符（模型 ID、文件路径、库名、命令）保持原文。

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [硬约束](#2-硬约束)
3. [数据画像](#3-数据画像)
4. [系统总览](#4-系统总览)
5. [Stage 0 — SFT 数据构造](#5-stage-0--sft-数据构造)
6. [Stage 1 — Hybrid 检索](#6-stage-1--hybrid-检索)
7. [Stage 2 — Query 重写](#7-stage-2--query-重写)
8. [Stage 3 — SFT (Qwen3.5-4B QLoRA)](#8-stage-3--sft-qwen35-4b-qlora)
9. [Stage 4 — DPO 偏好对齐](#9-stage-4--dpo-偏好对齐)
10. [Stage 5 — 自一致性推理](#10-stage-5--自一致性推理)
11. [Stage 6 — 评估、消融、误差归因](#11-stage-6--评估消融误差归因)
12. [代码组织](#12-代码组织)
13. [Notebook 组织 + 状态标记](#13-notebook-组织--状态标记)
14. [复现与提交](#14-复现与提交)
15. [测试矩阵](#15-测试矩阵)
16. [风险登记册](#16-风险登记册)
17. [设计决策记录](#17-设计决策记录)
18. [术语表](#18-术语表)

---

## 1. 背景与目标

气候事实核查系统（automated fact-checking for climate claims），COMP90042 研究生 NLP 课程项目，**截止 2026-05-22**。给定一条声明：

1. **检索**（Retrieval）：从 ~120 万段证据语料中找出最相关的若干段；
2. **分类**（Classification）：基于检索到的证据将声明判为
   `{SUPPORTS, REFUTES, NOT_ENOUGH_INFO, DISPUTED}` 之一。

评测主指标 = **证据 F-score 与分类准确率的调和平均**（`eval.py`）。

**核心目标**（按规范权重）：
- 方法论严谨性（**Soundness, 8 分**）
- 工作量与深度（Substance, 5 分）
- 新颖性（**Novelty, 6 分**）
- 结果与分析（Results, 6 分）
- 写作清晰度（Writing, 7 分）
- 学术引用（Citation, 3 分）

**非目标**：
- 不追求 leaderboard 第一名（参与可选，不影响分数）。
- 不做闭源 API；不做手写规则；不做 ≥ 5B 参数训练。

---

## 2. 硬约束

来自项目规范，违反 = 0 分。

| 编号 | 约束 | 我们的应对 |
|---|---|---|
| §2.1 | 必须有序列建模组件（RNN/LSTM/GRU/Transformer 任一） | Transformer 全程 ✓（Qwen3.5、bge-m3、bge-reranker） |
| §2.2 | LLM 仅限开源 | Qwen3.5 / bge-* 系列；ModelScope 下载 |
| §2.3 | 必须能在免费 Colab T4（~15 GB VRAM, ~12 GB RAM）跑通 | QLoRA 4-bit + ViT 冻结 + grad_accum |
| §2.4 | 禁用闭源 API（GPT/Claude/Gemini/Copilot） | 无 |
| §2.5 | 禁用手写 if-then 分类规则 | 标签由模型 logits 给出 |
| §2.6 | 禁用外部数据集（FEVER/Climate-FEVER 训练集等） | 仅用提供的 train + dev |
| §2.7 | Notebook 必须能复现汇报结果 | 缓存到 Drive，全 cell 可重跑 |
| §2.8 | 禁止上传 checkpoint / 数据 | `.gitignore` 排除 |
| §2.9 | 报告 ACL 模板，正文 ≤ 7 页 | 按模板写 |
| §2.10 | 必须使用官方 ipynb 模板 | `notebooks/notebook.ipynb` 在 3 个固定 section 下扩展 |

**Plan-mode 阶段曾考虑租多卡服务器训练 9B**，因与 §2.3/§2.7 直接冲突已**放弃**。Qwen3.5-9B 仅作"int4 推理 ablation"，不作主模型。

---

## 3. 数据画像

文件：`data/{train-claims,dev-claims,test-claims-unlabelled}.json` + `evidence.json`（174 MB，1,208,827 段）。

| 数据 | 规模 | 备注 |
|---|---|---|
| train | 1,228 claims | 含 label + 1–5 条金标 evidence |
| dev | 154 claims | 同上，**仅用于最终汇报指标，不参与训练 / 选型** |
| test | 153 claims | 仅 claim_text，无 label / evidence |
| evidence.json | 1.2M 段 | 中位 18 token / 106 字符；最长 479 token / 3148 字符 |

**类别分布（不平衡）**：
| label | train | dev | 比例 |
|---|---|---|---|
| SUPPORTS | 519 | 68 | 42% / 44% |
| NOT_ENOUGH_INFO | 386 | 41 | 31% / 27% |
| REFUTES | 199 | 27 | 16% / 17% |
| DISPUTED | 124 | 18 | 10% / 12% |

**金标证据数量** — 关键先验：
| label | min | median | max | 备注 |
|---|---|---|---|---|
| SUPPORTS | 1 | 2 | 5 | |
| REFUTES | 1 | 2 | 5 | |
| **NOT_ENOUGH_INFO** | **5** | **5** | **5** | **始终恰好 5 条** ⚡ |
| DISPUTED | 2 | 3 | 5 | |

NEI 类金标恒为 5 条 → **直接驱动"标签条件 k"策略**（推理时若预测 NEI 则 k=5，否则 k=4）。

完整 EDA 报告：`outputs/eda/eda_report.md`。

---

## 4. 系统总览

```
┌──────────────────── Stage 0: SFT 数据构造 (本地) ─────────────────────┐
│  EDA → 三维打标 (scenario × domain × difficulty)                     │
│       → hash 切分 (train_split / dev_holdout / diag_test)             │
│       → 防泄漏断言 (六路 ∩ = ∅)                                       │
│       → ms-swift 格式合成 + 硬负样本 + 课程排序                        │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
┌──────────────────── Stage 1: Hybrid Retrieval (Colab) ───────────────┐
│  BM25 (bm25s, top-200) + Dense (bge-m3, top-200)                     │
│       → 加权融合 (0.3*bm25 + 0.7*dense, top-150)                     │
│       → Cross-encoder rerank (bge-reranker-base, top-50)             │
│       → Rule reorder (NER boost / 去重 / 多样性, top-20)              │
│       → 标签条件 k (NEI=5, else=4)                                   │
└─────────────────┬──────────────────────────┬─────────────────────────┘
                  ▼                          │ HyDE 反哺
┌──── Stage 2: Query Rewrite (Colab) ────┐  │
│  synonym (WordNet) +                   │  │
│  sub-claim split (LLM) +               │  │
│  HyDE (LLM) → multi-query              ├──┘
└─────────────────────┬──────────────────┘
                      ▼
┌──────────────────── Stage 3: SFT (Colab T4 / AutoDL Ampere+) ─────────┐
│  Qwen3.5-4B (mixed-thinking VL + GatedDeltaNet)  ←  ModelScope        │
│  纯文本任务：禁思考 (--enable_thinking false + loss_scale ignore_empty)│
│  LoRA (r=16, α=32) on all-linear                                      │
│  QLoRA 4-bit；T4 → fp16；Ampere+ → bf16（运行时检测）                  │
│  显存堆叠：grad_ckpt + Liger + group_by_length                        │
│  ms-swift sft, 3 epoch, lr=2e-4, max_length=1024                      │
│  prompt: "Claim ... [1] ev1 [2] ev2 ... → LABEL ##[1,3]##"           │
│  数据格式：messages 标准三元组 (system/user/assistant)                 │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
┌──────────────────── Stage 4: DPO (Colab) ──────────────────────────────┐
│  推理 dev_holdout → 错预测构造 (chosen=gold, rejected=pred)           │
│  + DISPUTED-vs-SUPPORTS 对抗对                                        │
│  ms-swift rlhf --rlhf_type dpo, β=0.1, lr=5e-6, 1 epoch              │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
┌──────────────────── Stage 5: 推理 (Colab) ─────────────────────────────┐
│  自一致性: 5 samples @ T=0.7, top_p=0.9                              │
│  Majority vote on label                                              │
│  Max-confidence sample 的 evidences                                  │
│  → outputs/dev-claims-predictions.json                               │
│  → outputs/test-claims-predictions.json                              │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
┌──────────────────── Stage 6: 评估 + 消融 + 归因 (本地) ────────────────┐
│  9 路 ablation (A1-C2) → 主表 (官方 dev)                              │
│  3 张诊断切片表 (domain × 8 / scenario × 7 / difficulty × 3)          │
│  在 diag_test 上 → 错率最高的桶 = 数据-中心下一轮目标                  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Stage 0 — SFT 数据构造

参考 `~/.claude/skills/llm-data-pipeline` 的 M1→M2→M3→M4 闭环结构，**精简到本任务必需的子集**。

### 5.1 三维打标（scenario × domain × difficulty）

**目的**：打标不只是为采样配比，更是**误差归因抓手**——按维度切片后能精确定位"哪一类问题在拖后腿"。

#### A. Scenario（7 类，label-driven）

| scenario | 判定规则 |
|---|---|
| supports_clear | label=SUPPORTS, n_ev ≤ 2 |
| supports_aggregated | label=SUPPORTS, n_ev ≥ 3 |
| refutes_clear | label=REFUTES, n_ev ≤ 2 |
| refutes_aggregated | label=REFUTES, n_ev ≥ 3 |
| nei_topic_off | NEI 但 evidence 与 claim 余弦低（embedding 阶段细化） |
| nei_underspec | NEI 默认归此（heuristic） |
| disputed_conflict | label=DISPUTED |

#### B. Domain（8 类，关键词 + NER 启发式）

| 代号 | 关键词样例 |
|---|---|
| temperature | warming, °C, hiatus, anomaly |
| co2_atmospheric | CO2, ppm, emissions, GHG |
| sea_level | sea level, ice sheet, glacier, Antarctica, Greenland |
| extreme_weather | hurricane, flood, drought, wildfire |
| paleoclimate | ice core, MWP, LIA, Holocene, proxy |
| models_attribution | climate model, GCM, IPCC, RCP, attribution |
| policy_economics | renewable, subsidy, carbon tax, Kyoto, Paris |
| general_other | 兜底类 |

**实测分布**（train 1,228 条）：
- temperature 265 (22%) | co2 215 (18%) | sea_level 92 (7%) | models 53 | extreme 37 | policy 27 | paleo 20 | **general_other 519 (42%)** ← 启发式覆盖率不足，Colab 上将用 sentence-transformer 聚类细化
- DISPUTED + paleoclimate + hard 这种交叉桶非常稀少，需要分层上采样

#### C. Difficulty（3 级，三源加权）

```
score = 0.30 * length_norm + 0.25 * n_ev_norm + 0.45 * label_prior
       (claim 越长 / 证据越多 / 标签越难 → 越接近 1)
label_prior = {SUPPORTS:0.20, REFUTES:0.50, NEI:0.65, DISPUTED:0.85}

level = easy if score < 0.40 else medium if score < 0.65 else hard
```

实测：SUPPORTS 0 hard / DISPUTED 0 easy → 课程学习有真实梯度信号。

进阶版（Colab）：可叠加 `llm_judge`（zero-shot Qwen 给 0-1 分）和 `model_loss`（SFT 训练时该样本的 loss）。

### 5.2 防泄漏切分（hash split + 6 路断言）

```
bucket = md5(salt || claim_id) % 10
  0-7 → train_split   (~80%)
  8   → dev_holdout   (~10%)  仅供 SFT early-stop + DPO 偏好对生成
  9   → diag_test     (~10%)  完全保留；仅出现在诊断切片表
```

**实测**：986 / 121 / 121 / 154 (官方 dev)。

**六路硬断言**（任一非空 → `RuntimeError` 终止）：
```python
assert train ∩ dev_holdout = ∅
assert train ∩ diag_test = ∅
assert dev_holdout ∩ diag_test = ∅
assert train ∩ official_dev = ∅
assert dev_holdout ∩ official_dev = ∅
assert diag_test ∩ official_dev = ∅
```

`salt` 写死在 `src/splits.py`：哈希切分必须**确定性**，跨会话重跑要得到完全相同的分配。

### 5.3 ms-swift SFT 数据组装

每条 train_split claim → 一条 SFT 记录（**ms-swift messages 标准格式**，per `materials/训练数据格式.docx` §2.1）：

```jsonc
{
  "id": "claim-2967",
  "messages": [
    {"role": "system",    "content": "You are a climate fact-checking expert. ..."},
    {"role": "user",      "content": "Output rules: 1. ...\nClaim: <text>\nEvidence:\n[1] <ev1>\n[2] <ev2>\n...\nAnswer:"},
    {"role": "assistant", "content": "SUPPORTS ##[1,3]##"}
  ],
  "_meta": {"domain": "...", "scenario": "...", "difficulty": "easy",
            "n_gold": 2, "n_shown": 5, "shown": ["evidence-N", ...]}
}
```

> **格式选择**：ms-swift 的 AutoPreprocessor 也接受 query-response 形式 `{system, query, response}` 并自动映射，但 messages 是无歧义的官方标准格式。早期版本（2026-04 之前）我们用 query-response，2026-05 撞到 v3.6+ 分支若干隐式行为变化后统一迁移到 messages（见 `debug_log.md` 问题 15）。`id` / `_meta` 顶级字段对 ms-swift 透明，下游 curriculum / DPO 配对 / ablation 切片仍能消费。

**两种模式**：
- `gold_only`（默认）：用 claim 自带的 gold evidences。简单但与推理分布不一致。
- `retrieval`：把 claim 喂给检索器拿 top-k，再把 gold 强制塞进去（保住 SFT 信号）。**Colab 上启用此模式**。

**硬负样本扩充**：每条 claim 配上 k 条**随机非 gold** 证据，重新打标 NEI → 训练 NEI/topic_off 信号。

**课程排序**：每 epoch 内 `easy → medium → hard`，跨 epoch 重 shuffle 但保持坡度。

**实测产出**：1972 records (1228 + 744 hard-neg)，0.1 秒构造（O(N) 优化前 376 秒）。

### 5.4 数据质量门槛（M2 风格，4 维）

| 维度 | 计算 | 用途 |
|---|---|---|
| factual_alignment | claim-evidence 句嵌入余弦 | 检测明显错配 |
| evidence_sufficiency | sum(ev_tokens) / claim_tokens | 太短 / 太冗均扣分 |
| label_confidence | 距类心距离 | 离群样本扣分 |
| format_compliance | 是否能 parse 成 4 类之一 | 训练前校验 |

**`_aggregate.action = drop` if 任一维 < 0.3** → 丢弃前打日志。

> v1 实现（本地完成）只做 `format_compliance` 严格校验；其他三维需要 sentence-transformer，留待 Colab。

---

## 6. Stage 1 — Hybrid 检索

参考 `swxy/swxy/backend/app/service/core/rag/nlp/{search_v2,query}.py` 的工业级设计。

### 6.1 多路召回

| 路 | 模型 / 库 | top | 显存 | 备注 |
|---|---|---|---|---|
| 稀疏 | `bm25s` | 200 | RAM ~1.5 GB | NumPy / SciPy 稀疏；不要用 `rank_bm25`（Python 纯解释，OOM） |
| 稠密 | `BAAI/bge-m3` (568M, 1024d) | 200 | ~2 GB | 多语言；FAISS `IndexFlatIP` |
| 备选 | `BAAI/bge-small-en-v1.5` (33M, 384d) | 200 | 0.3 GB | bge-m3 OOM 时退此 |

**全语料 embedding 一次性 30–60 min**（Colab T4）。**必须缓存到 Drive**，分块 50k 段持久化避免 session 超时丢失。

### 6.2 加权融合（vs swxy 0.05/0.95）

```
score = 0.3 * bm25_norm + 0.7 * dense_norm
```

为什么不是 swxy 的 0.05/0.95？气候 claim 含大量**实体 + 数值**（"1.5 °C", "ppm", "IPCC"），lexical 信号比通用问答更重要 → BM25 权重提到 0.3。两侧各自 min-max 归一化后求和。

替代方案：RRF（Reciprocal Rank Fusion）`1/(60+rank)`，score-agnostic，更稳健。两者都实现，dev 上比较。

### 6.3 Cross-encoder rerank

`BAAI/bge-reranker-base` (278M)，对 (claim, candidate) pair 直接打分。Top-100 候选 reranking ~30 s on T4。

**可选 LoRA 微调**：用 BM25 hard negatives 训 2 epoch 对比损失，dev 上看是否真有提升；不提升则**直接 zero-shot 出货**。

### 6.4 规则重排

参考 swxy `query.py:FulltextQueryer` 的实体加权 + 多样性思路：

1. **NER 实体加权**：spaCy `en_core_web_sm` 抽 claim 实体；命中证据每加 1 个实体 +0.05 score（capped at 5×）
2. **去重**：top-20 内任意两条 token-Jaccard > 0.85 → 删后者
3. **多样性**：（可选）相同 source / topic 段落只保留 top-2

### 6.5 标签条件 k

```
if predicted_label == "NOT_ENOUGH_INFO": k = 5    # NEI 金标恒为 5
else: k = 4                                        # 其他类 median 2-3，k=4 最佳 F-score
```

但**预测标签**来自 Stage 5 推理，与 Stage 1 检索逆序依赖。两种实现：
- **Two-pass**：第一次 k=5 推理拿标签，第二次按标签调 k 重新检索 + 重新推理。慢但准确。
- **Threshold-based**：用 reranker 分数阈值 τ 替代固定 k（dev 上扫 τ）。一遍过。

**默认走 threshold-based**；time permitting 跑 two-pass 作 ablation。

---

## 7. Stage 2 — Query 重写

| 子模块 | 实现 | 触发时机 |
|---|---|---|
| 同义词扩展 | NLTK WordNet 单词替换（noun/verb，每词 ≤ 2 个变体） | 推理前 |
| 子声明分解 | Qwen 零样本 prompt → 1-3 atomic sub-claims | claim 含"and/or"或长度 > 30 token |
| HyDE | Qwen 零样本生成假设证据句 → embedding 与 claim embedding 加权 (α=0.6) | 推理前 |
| 实体抽取 | spaCy `en_core_web_sm` | 给规则重排用 |

**多查询融合**：每条变体单独 encode 后**取 top-K 候选 → RRF 融合**，最终再 rerank。

> Stage 2 模块全部本地可单元测试（`tests/test_query_rewrite.py` 7 用例绿）。HyDE 实际生成需要 SFT 后的模型。

---

## 8. Stage 3 — SFT (Qwen3.5-4B QLoRA)

### 8.1 框架对比 + 选择

| 框架 | 优势 | 劣势 | 选择 |
|---|---|---|---|
| **ms-swift** | ModelScope/阿里官方，Qwen 系列原生适配，CLI + Python API；Qwen3.5 配套 doc 完整 | 文档以中文为主；CLI 参数名跨版本有改动（见 §8.6） | **主选** |
| Unsloth | 2-5× 加速，显存省 70%，notebook 多 | Qwen3.5 / GatedDeltaNet 适配滞后 | 备选（仅文本路径） |
| TRL `SFTTrainer` | HF 官方最透明 | 显存优化弱；Qwen3.5 思考模式要自己处理 | 兜底 |
| ~~LLaMA-Factory~~ | 仍维护中 | 同任务多 ~30% 显存 | 不选 |

### 8.2 模型本质 — Qwen3.5-4B 是 mixed-thinking VL（不是纯文本 base）

> 早期版本（2026-04）的 design.md 把 Qwen3.5-4B 当纯文本 base 处理（写过 "ViT 冻结 / freeze_vit"），**完全错**。
>
> `materials/qwen3.5_key_points.docx` 明确：Qwen3.5 是 **mixed-thinking 多模态模型**，结合 GatedDeltaNet (linear attention) 和 full attention，带 image/video heads。我们做纯文本 fact-checking，但模型加载时仍依赖 `qwen_vl_utils`、GatedDeltaNet kernel 等。`--freeze_vit` 在新版 ms-swift 里**不是有效参数**，且 `--model_type qwen3_5_vl` 也不存在（model_type 由 ms-swift 从 `config.json` 自动推断）。

### 8.3 ms-swift CLI（当前版本，v3.6+）

```bash
swift sft \
  --model {MODEL_DIR} --use_hf false \
  --tuner_type lora --target_modules all-linear \
  --quant_bits 4 --bnb_4bit_compute_dtype <COMPUTE_DTYPE> \
  --enable_thinking false --add_non_thinking_prefix true --loss_scale ignore_empty_think \
  --dataset outputs/sft_data/sft_train_v1.jsonl \
  --val_dataset outputs/sft_data/sft_dev_holdout_v1.jsonl \
  --output_dir outputs/sft-out \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 --warmup_ratio 0.03 --max_length 1024 \
  --gradient_checkpointing true --use_liger_kernel true --group_by_length true \
  <PRECISION_FLAG> \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --save_steps 200 --eval_steps 200 --save_total_limit 2
```

**两条硬件路径** — 由变量 `<COMPUTE_DTYPE>` 和 `<PRECISION_FLAG>` 区分：

| 硬件 | `<COMPUTE_DTYPE>` | `<PRECISION_FLAG>` | 说明 |
|---|---|---|---|
| Colab T4 (Turing SM 7.5) | `float16` | `--fp16 true` | T4 无原生 bf16，fp16 是唯一选项 |
| AutoDL Ampere+ / 4080 SUPER (SM 8.9) | `bfloat16` | `--bf16 true` | bf16 数值范围大，更稳 |

代码侧（cell 3.5a 加载推理用模型）用运行时检测：
```python
_compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
```

**思考模式三件套**（`--enable_thinking false / --add_non_thinking_prefix true / --loss_scale ignore_empty_think`）：Qwen3.5 是 mixed-thinking 模型，不显式禁用就会输出 `<think>...</think>LABEL ##[..]##`，破坏我们的 parser。`loss_scale ignore_empty_think` 进一步避免模型学到生成空 think 块。

**显存堆叠**：
- `--quant_bits 4` — QLoRA base
- `--gradient_checkpointing true` — 激活重算
- `--use_liger_kernel true` — fused MLP/RoPE 等 kernel，省 ~10-20% 显存
- `--group_by_length true` — Transformers 后端不支持 GatedDeltaNet 的 packing/padding_free，按长度分组减 padding 浪费
- `--save_total_limit 2` — 防 ckpt 把 Drive 撑爆

### 8.4 显存预算（双路径）

| 组件 | T4 (15 GB) | 4080 SUPER (32 GB) |
|---|---|---|
| Qwen3.5-4B (4-bit base) | ~2.6 GB | ~2.6 GB |
| LoRA adapter + grads | ~0.5 GB | ~0.5 GB |
| Activations (seq 1024, bs=1) | ~6-8 GB | ~6-8 GB |
| Optimizer state (8-bit AdamW) | ~0.5 GB | ~0.5 GB |
| GatedDeltaNet workspace | ~0.5 GB | ~0.5 GB |
| **总计** | **~10-12 GB（余量 3-4 GB）** | **~10-12 GB（余量 20+ GB，可放大 batch / max_length）** |

**T4 路径耗时**：3 epochs × 1972 records ≈ 75-100 min（无 fla 加速则 +30%）。每 200 step 存 LoRA adapter 到 Drive 防 12h session 超时。
**AutoDL 路径耗时**：估算 ~25-35 min（4080 SUPER 实际 GPU 计算 ~3-4× T4）。

### 8.5 Qwen3.5 依赖栈（cell-setup-2 必装）

```bash
pip install -U "transformers==5.2.*" "qwen_vl_utils>=0.0.14" \
                peft trl liger-kernel bitsandbytes accelerate \
                ms-swift modelscope
pip install -U "flash-linear-attention>=0.4.2" --no-build-isolation
pip install -U "git+https://github.com/Dao-AILab/causal-conv1d" --no-build-isolation
```

**版本约束**：
- `transformers==5.2.*` — 5.1 缺 Qwen3.5 模型类，5.3 视频 dataloader 坏（per docx 指引）
- `qwen_vl_utils>=0.0.14` — VL 模型加载即检查
- `flash-linear-attention` + `causal-conv1d` — GatedDeltaNet 的快速 kernel；缺则降级到 torch 实现，能跑但慢

**故意不装** flash-attn 2.x：要求 SM ≥ 8.0（Ampere+），T4 (Turing 7.5) 不支持。装上反而编译失败。Ampere+ 上不装也无所谓（默认 sdpa attention 已足够）。

### 8.6 ms-swift CLI 参数名跨版本对照

我们撞到的 v3.6+ 过渡分支命名（CLI 路径 `swift/pipelines/train/sft.py`）：

| 概念 | 旧名（公开文档/v1.x） | 当前接受名 |
|---|---|---|
| 训练方式 | `--train_type lora` / `--sft_type lora` | **`--tuner_type lora`** |
| 量化位数 | `--quantization_bit 4` | **`--quant_bits 4`** |
| `--freeze_vit` / `--model_type qwen3_5_vl` | (曾经存在) | **已移除/不存在** |

跑前先 `swift sft --help | grep -iE "(tuner|quant|think|liger)"` 验证。详见 `debug_log.md` 问题 9。

### 8.4 Prompt 设计（参考 swxy `chat.py` 的引用编号）

```
SYSTEM: You are a climate fact-checking expert. Given a claim and several
        numbered evidence passages, decide whether the claim is SUPPORTED,
        REFUTED, has NOT_ENOUGH_INFO, or is DISPUTED, based on the evidence.

USER: Output rules:
      1. Output exactly one label as the first token: SUPPORTS / REFUTES /
         NOT_ENOUGH_INFO / DISPUTED.
      2. After the label, list the evidence numbers you relied on, in the
         form ##[1,3]##.
      3. Do not output anything else.

      Claim: {claim_text}
      Evidence:
      [1] {ev1}
      [2] {ev2}
      ...
      Answer:

ASSISTANT: SUPPORTS ##[1,3]##
```

**为什么是这个格式**：
- Label 在最前 → first-token logit mask 可强约束（见 §10.3）
- `##[..]##` 是稀有 token 序列，parser 容易 anchor
- 数字索引 → SFT 训练时强制把 evidence index 与 evidence ID 解耦：模型学的是"基于第几条证据下判断"，evidence ID 在 prompt assembly 阶段定（每条 claim 不同）

---

## 9. Stage 4 — DPO 偏好对齐

### 9.1 偏好对来源

只用 `dev_holdout`（121 条），**绝不用官方 dev**：

```python
for cid in dev_holdout:
    pred = sft_model(claim, retrieved)
    if pred.label != gold[cid].label or set(pred.evidences) != set(gold[cid].evidences):
        chosen   = build_target_response(gold.label, gold.ev, shown_ids)
        rejected = build_target_response(pred.label, pred.ev, shown_ids)
        emit_dpo_pair(claim, chosen, rejected)
```

### 9.2 对抗合成（DISPUTED-vs-SUPPORTS）

DISPUTED 是最难 4 类——它要求模型识别"看似支持但其实存在争议"的微妙状态。从 `supports_clear` 样本随机抽 30 条，构造：
- chosen = `SUPPORTS ##[1,...]##`（原 gold）
- rejected = `DISPUTED ##[1]##`（人工对抗）

教模型不要轻易跳到 DISPUTED。

### 9.3 训练（ms-swift）

```bash
swift rlhf --rlhf_type dpo \
  --model outputs/sft-out/checkpoint-best \
  --tuner_type lora --quant_bits 4 \
  --bnb_4bit_compute_dtype <COMPUTE_DTYPE> \
  --enable_thinking false --add_non_thinking_prefix true --loss_scale ignore_empty_think \
  --dataset outputs/dpo_data/dpo_pairs.jsonl \
  --beta 0.1 \                          # 经典 DPO
  --learning_rate 5e-6 \
  --num_train_epochs 1 \
  --max_length 1024 --max_prompt_length 896 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --gradient_checkpointing true --use_liger_kernel true \
  --output_dir outputs/dpo-out \
  <PRECISION_FLAG>
```

`<COMPUTE_DTYPE>` / `<PRECISION_FLAG>` 跟 SFT 同（T4 → fp16，Ampere+ → bf16，见 §8.3）。**思考模式三件套必须与 SFT 时一致**，否则 DPO 看到的 base policy 与训练时的 prompt 模板不匹配。

**显存技巧**：PEFT 的 `ref_model=None` 自动用"关闭 adapter 的同模型"作参考策略，省一份显存。

`dpo_pairs.jsonl` schema（**ms-swift messages 标准格式**，per docx §3.3 — DPO/ORPO/CPO/SimPO/RM）：
```jsonc
{
  "messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "<chosen response>"}
  ],
  "rejected_response": "<rejected response>"
}
```

---

## 10. Stage 5 — 自一致性推理

### 10.1 推理流程

```python
def predict(claim_text):
    retrieved = pipeline.retrieve(claim_text)            # Stage 1+2
    shown_ids = [eid for eid, _ in retrieved]
    user_msg = build_user_query(claim_text, retrieved)

    # 自一致性：5 次采样
    labels, ev_lists = [], []
    for _ in range(5):
        out = model.generate(prompt, T=0.7, top_p=0.9, max_new=32)
        lbl, evs = parse_response(out, shown_ids)
        labels.append(lbl); ev_lists.append(evs)

    # 多数投票
    final_label = Counter(labels).most_common(1)[0][0]
    # 在投出 final_label 的样本里选证据列表最长的（置信度代理）
    best = max([i for i,l in enumerate(labels) if l == final_label],
               key=lambda i: len(ev_lists[i]))
    return {"claim_label": final_label, "evidences": ev_lists[best]}
```

预期增益：accuracy +2–3%，evidence F-score 持平或微涨。

### 10.2 三种推理器（src/inference.py）

| 类 | 用途 | 模型 | 解码 |
|---|---|---|---|
| `ModelInferer` | SFT/DPO 后的 B1-B3 ablation | 真模型 + tokenizer | 5 sample @ T=0.7 |
| `ZeroShotInferer` | A1-A4（无 SFT）ablation | 真模型但 greedy | num_beams=1 |
| `RetrievalOnlyInferer` | 检索 F-score 隔离测试 | 无 LLM；标签 = "SUPPORTS" / 随机 | — |

### 10.3 约束解码（可选）

约束首 token 在 4 个 label tokens 中：
```python
allowed = tokenizer.convert_tokens_to_ids(["SUPPORTS","REFUTES","NOT","DISPUTED"])
def first_token_constraint(input_ids, scores):
    if input_ids.shape[1] == prompt_len:  # 第一个生成 token
        mask = torch.full_like(scores, float("-inf"))
        mask[:, allowed] = scores[:, allowed]
        return mask
    return scores
```

实测先不加，看 SFT 后 label 自然分布；若出现非法 label > 1% 再开。

### 10.4 鲁棒性

`predict_all` 在单条 claim 异常时**降级到 NEI + dummy evidence** 而不是终止整批：
- 保证最终 JSON 通过 `eval.py` schema 校验（labels valid + evidences 非空）
- WARN log 单独记录，便于事后归因

---

## 11. Stage 6 — 评估、消融、误差归因

### 11.1 三层指标

1. **`score_predictions(preds, gold)`** —— 复刻 `eval.py:24-76` 的 F / A / HM 三指标，**与官方算法 1e-15 精度一致**（已固化为测试）。
2. **`score_per_bucket(preds, gold, bucket_fn)`** —— 同算法但按桶分组。`bucket_fn` 是 `claim_id → str` 的任意映射，给三张诊断表用。
3. **`recall_at_k(retrieved, gold, k)`** —— 检索-only 指标，用于 Stage 1 调参。

### 11.1b 评估驱动的迭代工作流（per D-015）

执行顺序**不是** "Stage 0 → 3 → 6"，而是：

```
Phase 1  Track 1 + Track 2 评估（base + RAG, 当前 prompt）
            ↓ per-bucket 诊断切片（domain × scenario × difficulty）
Phase 2  Prompt 迭代（zero/few-shot, CoT, NEI 触发条件）
            ↓ 每版重测 → 锁最优 prompt
Phase 3  最优 prompt + RAG 重新评估
            ↓ 识别仍弱的桶（acc < 0.3）
Phase 4  SFT 数据按"弱桶倾斜"重新生成（调 sampler 权重，不改 build_sft_record 流水线）
            ↓ Track 3 (SFT) 训练 + 评估
Phase 5  对比扩充前后弱桶 acc，→ 决定是否进 DPO
```

每个 phase 之间产出的诊断切片表持久化到 `outputs/eval_phase{N}.md`，作为下一阶段决策依据 + 报告写作素材。

### 11.2 9 路 ablation 配置

| ID | 配置 | 想看什么 |
|---|---|---|
| A1 | BM25 + zero-shot Qwen3.5 | 最朴素 baseline |
| A2 | + dense (bge-m3) | dense 召回贡献 |
| A3 | + cross-encoder rerank | 重排贡献 |
| A4 | + rule reorder + HyDE | RAG 工程化贡献 |
| B1 | A4 + Qwen3.5 SFT | SFT 贡献 |
| B2 | + DPO | 偏好对齐贡献 |
| B3 | + self-consistency | 推理稳健性贡献（**flagship**） |
| C1 | B3 inferred with Qwen3.5-9B int4 | 模型尺度敏感性（仅推理 ablation） |
| C2 | B3 without curriculum | 课程贡献 |

### 11.3 报告图表（6–9 个）

1. retrieval recall@k 曲线（BM25 / +dense / +rerank / +rule）
2. dev 上 k 扫描的 F-score 曲线
3. 4 类混淆矩阵（突出 DISPUTED 混淆模式）
4. ablation 主表（9 行 × {F/A/HM}）—— 在**官方 dev** 上算
5. **每领域诊断切片表**（在 `diag_test`，8 domains × {F,A}）
6. **每场景诊断切片表**（7 scenarios × A）
7. **每难度诊断切片表**（3 levels × A）
8. 课程开/关 loss 曲线
9. （可选）DPO reward gap 曲线

### 11.4 数据中心闭环（M5 风格）

报告 Discussion 必写：
1. 在 `diag_test` 上把每条错预测带 `(domain, scenario, difficulty)` 写到 `error_log.jsonl`
2. `error_rate_per_bucket = errors / samples` 排序
3. 前 3 个桶 → "下一轮迭代精确目标"
4. 不必真的迭代 —— 把"识别 bucket X 是瓶颈，下一步会做 Y"写进报告就拿 novelty + analysis 分

### 11.5 4-Track 边际增益分析（notebook §3.5）

§11.2 的 9 路 ablation 是**完整消融矩阵**；4-track 是其中**最关键的 4 行**（A1 / A4 / B1 / B3）的"逐 cell 演进式"展示，每加一个组件就跑一遍指标。专为论文 Discussion 的"哪个组件贡献多少"设计。

#### 配置矩阵

| Track | 检索 | 模型 | 解码 | Inferer 类 | 期望 HM 增量 |
|---|---|---|---|---|---|
| 1. Base + prompt eng. | 无 | Qwen3.5-4B base | greedy | `NoRagInferer` | — (baseline) |
| 2. Base + RAG | BM25+dense+rerank | Qwen3.5-4B base | greedy | `ZeroShotInferer` | +0.20 ~ +0.30 |
| 3. SFT + RAG | 同 Track 2 | + SFT LoRA | greedy | `ZeroShotInferer(sft)` | +0.05 ~ +0.15 |
| 4. SFT+DPO+RAG+SC | 同 Track 2 | + DPO LoRA | self-consistency 5 sample | `ModelInferer` | +0.01 ~ +0.03 |

#### 关键设计决策

- **共用 `pipeline_zero_shot`**：Track 2/3/4 检索条件完全一致 → SFT/DPO 增益可干净归因
- **共用 `base_model`（4-bit）**：Track 1/2 共用一份模型副本，省 VRAM
- **SFT/DPO 用 `PeftModel.from_pretrained(base, ckpt)`**：不重复加载基座
- **SC 仅 Track 4 开启**：其他 track greedy → SFT→DPO delta 不被采样噪声干扰
- **Track 1 evidence 是 stub** (`["evidence-0"]`) → F=0 by design，**只读 Label Acc** 作为参数知识基线
- **缺 checkpoint 不报错**：`run_4tracks` / 各 track cell 自动 skip 缺失的 inferer，可只跑 Track 1+2 验证 RAG 链路

#### Notebook §3.5 Cell 顺序（7 cell）

```
3.5a              load base model + 共享配置 (EVAL_N / eval_claims / PRED_DIR)
3.5-track1        Base + prompt engineering          → track1_result
3.5-track2        Base + RAG                         → track2_result   (Δ vs Track1)
3.5-load-adapters load SFT / DPO LoRA (缺失 → skip)
3.5-track3        SFT + RAG                          → track3_result   (Δ vs Track2)
3.5-track4        SFT + DPO + RAG + Self-Consistency → track4_result   (Δ vs Track3)
3.5-summary       聚合 → outputs/eval_compare.md + 终端表格
```

每个 track cell 跑完立即打印 `acc / F / HM / Δ HM / 时长`，**不等所有 track 跑完才出结果**。

#### 输出物

```
outputs/predictions/track1_base.json
outputs/predictions/track2_base_rag.json
outputs/predictions/track3_sft.json
outputs/predictions/track4_dpo.json
outputs/eval_compare.md          # markdown 对比表带 Δ 列
```

每个 JSON 都通过 `eval.py` schema 校验，可单独丢给官方脚本复核。

#### 运行规模开关

3.5a cell 的 `EVAL_N` 是统一开关：
- `EVAL_N = len(dev)` (154) → 最终汇报跑这个
- `EVAL_N = 30` → 快速 smoke test（约 5× 提速），用于训练后立刻检查方向

#### 健康判断（用于 Discussion 写作）

| 现象 | 解读 | 行动 |
|---|---|---|
| Track 1 acc ≈ 0.25 | 模型 / parser 失效（≈ 4 类随机基线） | 检查 base 模型加载、tokenizer 配对 |
| Track 2 F = 0 | RAG 检索没接通 | 回 §6 检查 `pipeline.retrieve(claim)` 输出 |
| Track 2 acc < Track 1 acc | RAG 上下文挤压注意力 | 通常 SFT 后修正；先继续 |
| Track 3 Δ HM < +0.02 | SFT 未学到任务格式 | 检查 epoch 数 / 学习率 / `sft_train_v1.jsonl` 格式 |
| Track 4 Δ HM < +0.005 | DPO 偏好信号弱 | 检查 dpo_pairs.jsonl 错预测覆盖、β / 学习率 |

---

## 12. 代码组织

仓库根目录：
```
.
├── data/                    # 官方数据 (含 174 MB evidence.json，gitignore)
├── eval.py                  # 官方评测脚本（只读，不改）
├── notebooks/
│   ├── notebook.ipynb                           # 主交付物
│   └── GroupID__COMP90042_Project_2026.ipynb    # 官方模板存档
├── src/                     # Python package
│   ├── data_io.py           # 数据加载/写入
│   ├── paths.py             # Colab vs 本地路径
│   ├── eda.py               # EDA 报告生成
│   ├── tagging.py           # scenario × domain × difficulty 启发式
│   ├── splits.py            # hash 切分 + 防泄漏
│   ├── stage0_tag.py        # tagging 入口
│   ├── prompt.py            # SFT prompt 模板 + parser
│   ├── sft_dataset.py       # ms-swift 格式构造 + 硬负样本 + 课程
│   ├── query_rewrite.py     # 同义词 + 子声明 + HyDE
│   ├── dpo_pairs.py         # DPO 偏好对构造
│   ├── eval_helpers.py      # 复刻 eval.py + 切片
│   ├── inference.py         # 自一致性推理 + 三种推理器
│   ├── ablation.py          # 消融 harness + 报表渲染
│   ├── build_stage0.py      # Stage 0 一键运行
│   └── retrieval/
│       ├── bm25.py          # bm25s 包装
│       ├── dense.py         # bge-m3 + FAISS
│       ├── fuse.py          # 加权 / RRF 融合
│       ├── rerank.py        # cross-encoder + 规则重排
│       └── pipeline.py      # 端到端编排
├── tests/                   # 30+ 个本地单元测试，全绿
│   ├── test_prompt.py       (8)
│   ├── test_eval_helpers.py (3)  ← 复刻 eval.py 1e-15 精度
│   ├── test_sft_dataset.py  (3)
│   ├── test_fuse.py         (4)
│   ├── test_query_rewrite.py(7)
│   ├── test_dpo_pairs.py    (5)
│   ├── test_inference.py    (4)
│   └── test_ablation.py     (3)
├── scripts/
│   └── dry_run.py           # 上 Colab 前自检（exit 0 即可）
├── outputs/                 # 全部 gitignore
│   ├── eda/eda_report.md
│   ├── sft_data/{claims_tagged,sft_train,sft_dev_holdout,sft_diag_test}_v1.jsonl
│   ├── splits/{train_split,dev_holdout,diag_test,official_dev}.jsonl
│   ├── PROGRESS.md          # 跨 session 进度日志
│   └── dry_run_report.md    # 自检审计 trail
├── design.md                # ← 本文件
└── README.md                # 项目规范
```

**模块依赖原则**：
- `paths.py` / `data_io.py` 是叶子节点，无外部依赖
- 重型依赖（torch / sentence-transformers / bm25s）**惰性导入**到方法内 → 本地无 GPU 也能 import 整个 package
- 测试只依赖 numpy / nltk / transformers，不需要 GPU 库

---

## 13. Notebook 组织 + 状态标记

`notebooks/notebook.ipynb`（45 cells）port 进官方模板，3 个固定 section 标题保留：
- `# 1.DataSet Processing` → 1.1 EDA / 1.2 三维打标 / 1.3 切分 / 1.4 SFT 数据
- `# 2.Model Implementation` → 2.1 BM25 / 2.2 dense / 2.3 融合+重排 / 2.4 k-sweep / 2.5 SFT / 2.6 DPO / 2.7 推理
- `# 3.Testing and Evaluation` → 3.1 scorer 校验 / 3.2 ablation / 3.3 诊断切片 / 3.4 测试预测
- `## Object Oriented Programming codes here` → 重导入 `src/` 中的关键类

**每个 sub-section header 带状态标记**：
- ✅ verified locally —— 本地 dry-run 已跑过（Stage 0 全部 + 评估辅助）
- 🧪 stub-validated locally —— 单测 + 合成数据已验证 wiring（消融 harness）
- ⏳ requires Colab T4 —— 重计算（Stage 1 / 3 / 4 / 5 推理）

完整审计 trail：`outputs/dry_run_report.md`。

---

## 14. 复现与提交

### 14.1 三步复现

1. 准备数据 + 模板：`data/evidence.json` + `notebooks/GroupID__COMP90042_Project_2026.ipynb`
2. 本地：`python -m scripts.dry_run` → exit 0
3. Colab：`notebook.ipynb` 全 cell run → 产出 `outputs/test-claims-predictions.json`

### 14.2 提交清单

| 文件 | 内容 | 备注 |
|---|---|---|
| `COMP90042_<team>.pdf` | ACL 报告，≤7 页正文 | 团队贡献 + references 不计页 |
| `COMP90042_<team>_resource.zip` | notebook.ipynb + src/ + tests/ + scripts/ + design.md | **不**带 evidence.json / checkpoint / embeddings |

### 14.3 复现性自检

提交前 checklist：
- [ ] `notebooks/notebook.ipynb` 全 cell **clear all output → restart → run all** 不报错
- [ ] `python eval.py --predictions outputs/dev-claims-predictions.json --groundtruth data/dev-claims.json` 三个数与报告一致
- [ ] zip 大小 < 50 MB（无大文件）
- [ ] `python -m scripts.dry_run` exit 0
- [ ] `for t in tests/test_*.py; do python -m tests.$(basename $t .py); done` 全绿

---

## 15. 测试矩阵

| Suite | 用例数 | 覆盖模块 | 关键断言 |
|---|---|---|---|
| test_prompt | 8 | `prompt.py` | 模板生成 + parser 鲁棒性（含 garbage label / OOB index） |
| **test_eval_helpers** | **3** | `eval_helpers.py` | **复刻 eval.py 至 1e-15 精度** |
| test_sft_dataset | 3 | `sft_dataset.py` | 课程排序 / 硬负样本 / 标签格式 |
| test_fuse | 4 | `retrieval/fuse.py` | 加权归一化 / RRF 一致性 |
| test_query_rewrite | 7 | `query_rewrite.py` | WordNet 同义词 / 子声明 parse / HyDE prompt |
| test_dpo_pairs | 5 | `dpo_pairs.py` | 错预测才出对 / DISPUTED 对抗合成 |
| test_inference | 4 | `inference.py` | predict_all schema / 失败降级 |
| test_ablation | 3 | `ablation.py` | 主表渲染 / flagship 切片 / 全报告写盘 |
| **总计** | **37** | | **all green** |

加上端到端 dry-run（`scripts/dry_run.py`），**全部本地代码自检 ~3 秒**。

---

## 16. 风险登记册

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| ~~Qwen3.5-4B HF/ModelScope ID 未公开~~ | ~~中~~ | ~~阻塞 SFT~~ | **已解决**：`Qwen/Qwen3.5-4B`（base，无 -Instruct）已在 ModelScope 公开 |
| ms-swift CLI 参数名跨版本变（`tuner_type` vs `train_type` vs `sft_type`）| **中** | 训练命令直接报错 | 跑前 `swift sft --help` 自检；§8.6 的对照表；`debug_log.md` 问题 9 |
| Qwen3.5 GatedDeltaNet kernel 缺 → 训练慢 | **中** | 训练时间 +30% 但能跑 | 装 fla + causal-conv1d；T4 上若编译失败接受降级 |
| Qwen3.5 思考模式污染输出 | **中** | parse_response 解析失败率上升 | 思考三件套 (`--enable_thinking false / --add_non_thinking_prefix / --loss_scale ignore_empty_think`) 训推一致 |
| Qwen3.5-4B QLoRA OOM | 低（4080 SUPER）/ 中（T4） | 训练不能跑 | T4：max_length 768 + Liger + group_by_length；4080 SUPER：默认即可 |
| AutoDL 镜像 PyTorch/CUDA 不匹配 | **中** | 编译类包全部失败 | 创建实例时直接选标准镜像 `PyTorch 2.5.1 + CUDA 12.4 + Python 3.12`；不要用 cu130 |
| transformers 5.x 行为变化（`apply_chat_template` 返回 BatchEncoding） | **高** | 推理静默失败 + acc≈随机 | `if torch.is_tensor(x) else x["input_ids"]` 鸭子类型判断；`predict_all` 前 N 错打 traceback |
| Python 模块缓存让代码改动不生效 | **高** | 验证修复时被骗 | 改 `src/*.py` 后必须 reload 或重启 kernel；用 `hasattr(module, new_attr)` 一行确认 |
| DPO OOM | 中 | 损失新颖性 | β=0.1 + max_length=512；或砍 DPO |
| bge-m3 全语料 embedding 太久 | 中 | 阻塞 retrieval | 退 `bge-small-en-v1.5`（30 min 内可完成） |
| BM25 build OOM | 低 | retrieval 退化 | bm25s 已是 NumPy/SciPy 稀疏；分片建索引；或 Pyserini |
| 9B int4 推理太慢 | 中 | C1 ablation 没跑 | 报告诚实说明算力受限 |
| DISPUTED F1 ≈ 0 | 中 | 整体 accuracy | focal loss + 加大 hard 负采样 + DPO 对抗对手工增 |
| Colab session 12h 超时 | 高 | 训练中断 | 每 200 step 存 LoRA adapter 到 Drive；`save_total_limit=2`；`resume_from_checkpoint` |
| 评分时无 GPU 重跑 | 高 | 评分员看不到结果 | notebook 缓存所有重计算结果到 markdown；`outputs/dry_run_report.md` 留 audit trail |

**进度紧时砍的优先级**（从最先砍到最后砍）：
1. Stage 4 DPO（保 SFT 即可拿大头分）
2. Stage 1.5 HyDE（保留规则重排）
3. Stage 0.4 课程学习（改成 random shuffle）
4. Stage 5 自一致性（greedy 解码足够）

---

## 17. 设计决策记录

按时序记录关键决策和"为什么不选别的"。

### D-001：Hybrid RAG 而非纯 Transformer 分类
- **决策**：retrieval + LLM 推理 hybrid，不用 DeBERTa-v3-base 直接做 4 分类
- **替代**：纯分类器（曾在 plan v1 提过）
- **理由**：(1) 规范明确"鼓励 hybrid 系统"；(2) 1228 训练样本对 base 模型够，但 evidence retrieval 是任务核心评测的一半，纯分类器需要单独检索器；(3) 一个 LLM 同时输出 label + evidence 索引更优雅，且 SFT 信号一致

### D-002：模型选 Qwen3.5-4B 而非 9B
- **决策**：4B QLoRA SFT 主线；9B 仅作 int4 推理 ablation
- **替代**：9B 完整 SFT、租多卡服务器
- **理由**：§2.3/§2.7 复现性约束 → 必须 Colab T4 跑通；4B QLoRA 显存预算 11–13 GB 安全；9B QLoRA 4-bit 仍要 18–22 GB

### D-003：训练框架选 ms-swift
- **决策**：SFT + DPO 都用 ms-swift
- **替代**：Unsloth（更快但 VL 支持滞后）、LLaMA-Factory（多 30% 显存）、TRL（无 freeze_vit 一行）
- **理由**：ModelScope 原生适配 Qwen 系列；`--freeze_vit true` 一行；CLI + Python API 双形态

### D-004：融合权重 0.3/0.7（而非 swxy 的 0.05/0.95）
- **决策**：`score = 0.3*bm25 + 0.7*dense`
- **理由**：气候 claim 含大量实体 + 数值（1.5 °C / ppm / IPCC），lexical 信号比通用问答重要 → BM25 权重提升

### D-005：标签条件 k = {NEI:5, else:4}
- **决策**：推理时按预测标签调 final_k
- **替代**：固定 k；reranker 阈值 τ
- **理由**：EDA 发现 NEI 金标恒为 5。τ-based 也实现，作 ablation

### D-006：诊断 hold-out 切分（diag_test）
- **决策**：从 train 切 10% 作 `diag_test`，仅用于诊断切片表
- **替代**：直接在官方 dev 上做切片
- **理由**：dev 154 条 / 切到 8 个 domain × 3 个 difficulty = 每格 ≤ 5 条，统计噪声大；diag_test 121 条切片噪声同样大但**不污染汇报指标**

### D-007：DPO 而非完整 RL
- **决策**：SFT + DPO（无 reward model），不做 GSPO/PPO
- **替代**：GSPO + group-wise sampling
- **理由**：完整 RL 在 Colab 上风险大、收益不确定；DPO 一遍就出，且 ref_model=None 省显存

### D-008：Hash 切分用 md5 而非 train_test_split
- **决策**：`md5(salt || claim_id) % 10`
- **替代**：sklearn `train_test_split(random_state=42)`
- **理由**：跨 session 重跑需完全一致的分配；md5 + salt 给跨工具（Python / Bash / awk）一致结果

### D-009：本地优先，Colab 兜底
- **决策**：所有不依赖 GPU 的代码都在本地写完跑通，再上 Colab
- **替代**：直接在 Colab 上写
- **理由**：Colab session 12h 超时，频繁断连；本地 IDE 调试体验远好于 Colab；dry-run 在 ~3 秒内验证整条链路

### D-010：源代码 + notebook 双交付
- **决策**：notebook 仅作 orchestration，重逻辑在 `src/`；OOP section 重导入关键类
- **替代**：所有代码都写在 notebook 里
- **理由**：(1) `src/` 可单元测试；(2) notebook ≤45 cells 可读；(3) 评分员可单独看 `src/` 而非翻 notebook

### D-011：Qwen3.5-4B base 而非 -Instruct（被动决策）
- **决策**：用 `Qwen/Qwen3.5-4B`（base），不用 `-Instruct`
- **替代**：`Qwen/Qwen3.5-4B-Instruct`（**ModelScope 上不存在**）
- **理由**：ModelScope 只放出了 base 版本。Track 1/2 (zero-shot) 表现因此偏弱，但**这反而让 Track 3 (SFT) 的 delta 更显著**，对论证 SFT 价值有利。Smoke test 实测：base 在 4-class 选择任务上能输出干净 label（对 SUPPORTS / REFUTES），但对 NEI 类倾向给 REFUTES（base 模型缺"我不知道"概念）。
- **AutoDL 实测佐证（DISPUTED claim + 混合 evidence + SC 5×T=0.7）**：base 模型 5/5 都给 SUPPORTS，但其中 sample 5 的引用列表是 `##[1,2]##`（同时引用了支持 evidence 和反驳 evidence）。说明 base 模型**有"识别 evidence 相关性"能力，但缺"识别 evidence 矛盾 → 升级标签到 DISPUTED"能力**——正好是 `disputed_conflict` scenario + DPO 对抗对要教的核心技能。这条观察直接支撑"base 不够 + 必须 SFT/DPO"的论证。

### D-012：SFT/DPO 数据用 messages 标准格式（而非 query-response）
- **决策**：record schema = `{messages: [{system}, {user}, {assistant}], _meta}`
- **替代**：legacy `{system, query, response, _meta}`（ms-swift AutoPreprocessor 也接受）
- **理由**：(1) `materials/训练数据格式.docx` §2.1 明确 messages 是官方"无歧义标准"格式；(2) AutoPreprocessor 的隐式映射在 v3.6+ 过渡分支上行为变化频繁（与 §8.6 的 CLI 改名同源），messages 直接绕开映射层；(3) 多模态 / Agent / 工具调用未来扩展统一走 messages
- **影响**：`src/sft_dataset.py` + `src/dpo_pairs.py` + tests + notebook cell-1-sft 全部迁移；`id` / `_meta` 顶级字段保留（ms-swift 忽略未知 key，下游 curriculum / DPO 配对 / ablation 切片仍消费）

### D-013：T4 / Ampere+ 双硬件路径（运行时检测，不固化 dtype）
- **决策**：所有 dtype 选择走 `torch.cuda.is_bf16_supported()`；不硬编码 `bfloat16`
- **替代**：固定 bf16（design.md v1.0 的写法，T4 上变慢且不稳）；分两份 notebook
- **理由**：T4 (Turing 7.5) **无原生 bf16**，强行用会软件模拟；Ampere+ 上 bf16 数值范围大更稳。一份 notebook 跨硬件跑的关键是运行时检测。SFT CLI 不能跨硬件统一，所以 cell-2-sft-train 注释里明确两条路径的差异
- **辅助**：故意**不装 flash-attn 2.x**（要 SM ≥ 8.0），sdpa attention 在 T4 上是更稳的默认

### D-014：Standalone smoke-test 脚本（`scripts/test_qwen35_inference.py`）
- **决策**：建一个独立脚本，不依赖 notebook、不依赖 RAG 索引，验证 model + prompt + parse 三件事
- **替代**：所有验证都在 notebook cell 里做
- **理由**：(1) 调通新硬件 / 新模型时，notebook 大量上下文初始化分散注意力；(2) 脚本 4 个 section（env / model load / tokenizer probe / inference 4a-4d）能在 5 分钟内回答"模型本身能不能跑、推理路径有没有兼容性问题、SC 在易题/难题上分别什么表现"；(3) 加 `--with-real-rag` 可选开关，索引就绪后还能验证完整 Track-2 路径，无需开 notebook

### D-015：评估驱动、迭代式 SFT 数据扩充（方法论转向）
- **决策**：先做透 Track 1/2 的评估 + prompt 优化，然后**用诊断切片表识别 base 模型最弱的桶**，再针对性扩充 SFT 数据。**不一上来就 SFT 全量训练**。
- **替代（旧路线）**：按预设计划走 Stage 0 数据构造 → Stage 3 SFT 全量训练 → Stage 6 评估并归因。
- **理由**：
  1. **避免无效计算**：Smoke test 实测 base 模型已能输出干净的 `LABEL ##[..]##` 格式，所以 prompt 跟随能力不是瓶颈。瓶颈在**特定推理能力**（NEI 拒答、DISPUTED 矛盾识别）。如果不评估就 SFT，可能在 prompt 能修的部分浪费 LoRA 容量。
  2. **数据设计有的放矢**：旧路线靠 scenario × domain × difficulty 三维**先验**配比（合理但盲目）。新路线用 Track 2 的 per-bucket 诊断表反向倒推哪些桶 acc 最低（**后验证据**），SFT 数据按这个倾斜配比 → 同样训练量、对最差桶的提升更大。
  3. **报告叙事更强**：可以写"我们先充分压榨 prompt + RAG，识别出 base+RAG 在 X / Y 桶上 acc < 0.3 → 针对性合成 Z 条数据 → SFT 后这两桶提升到 0.6+"。这比"做了 SFT，整体 +5%"的论证强得多。
- **新工作流**：
  1. **评估 Phase 1**：完整跑 Track 1（base no-RAG）+ Track 2（base + RAG）on official dev + diag_test，产出**Track 2 的诊断切片表**（domain × scenario × difficulty）
  2. **Prompt 优化 Phase**：迭代多版 prompt（zero-shot / few-shot / chain-of-thought / 显式 NEI 触发条件），每版重新 Track 1/2 评估，取最优
  3. **评估 Phase 2**：用最优 prompt 重跑 Track 2 → 新诊断切片表 → **识别仍弱的桶**（典型预期：DISPUTED + NEI_topic_off + general_other 域）
  4. **SFT 数据扩充**：针对最弱桶，加大 hard-negative / DISPUTED 对抗 / domain-specific 样本配比（仍用原 build_sft_record 流水线，只调 sampler 权重）
  5. **SFT + 评估 Phase 3**：训练后再次诊断切片，**对比扩充前后的弱桶提升**
- **不变的部分**：Stage 0/1/3/4/5/6 的代码骨架、检索策略、评测脚本都不需要改；变的只是**执行顺序**和**SFT 数据配比策略**。
- **风险**：评估 + prompt 优化阶段会消耗一些 GPU 时间（推理 154 dev claims × N 个 prompt 变体），但相比 SFT 训练成本可忽略。
- **关联**：D-011 的 DISPUTED 实测已经预示了这套流程的产出形态——base 模型对"识别证据矛盾"无能，正是评估阶段会显式量化、SFT 阶段定向修补的典型弱点。

### D-016：本地离线转 .bin → .safetensors，不升级 torch（被动决策）

- **决策**：当 ModelScope 镜像只发 `pytorch_model.bin` 触发 transformers CVE-2025-32434 安全检查时，**保持 AutoDL 镜像 torch 2.5.1+cu124 不动**，改用 `scripts/convert_bin_to_safetensors.py` 离线把 .bin 转成 .safetensors。
- **替代（被否决）**：
  1. 升级 torch 到 ≥ 2.6 满足检查 → 破坏 flash-attn 2.x / bitsandbytes / flash-linear-attention 整条已编译依赖（D-013 + debug_log 问题 11、13 都建立在 torch 2.5 上）
  2. 从 HuggingFace 直接补下 safetensors → AutoDL 墙内 `huggingface.co` 不可达
  3. 用 `HF_ENDPOINT=https://hf-mirror.com` 走镜像 → 可行，但增加 2.27 GB 网络往返；本地已有 .bin，没必要
- **理由**：
  1. `torch.load` API 本身在 2.5 下能用；CVE 限制只在 transformers 的**包装层**。用户代码里直接调 `torch.load` 不受限。
  2. 转换是一次性操作（每个模型 30 s CPU），转完所有后续加载走 safetensors 路径，无副作用。
  3. 保留 `.bin.bak` 作为回滚后悔药；确认 retrieval 跑通后 `find models -name '*.bin.bak' -delete` 释放磁盘。
- **适用边界**：仅适用于 **single-file** `pytorch_model.bin` 布局（如 bge-m3、bge-reranker-base、bge-small-en-v1.5）。**Sharded** `pytorch_model-*-of-*.bin` 需要同步重生成 `pytorch_model.bin.index.json` 的 weight_map，脚本主动 warn 并跳过——这种情形请改走 hf-mirror.com 镜像重下。Qwen3.5-4B 是 sharded safetensors，无此问题。
- **关联**：debug_log 问题 16；optimization_plan §7.2 / §8。

### D-017：环境栈锁定（2026-05-12，Phase 5 SFT 启动前）

- **决策**：把 AutoDL 上跑 SFT 的整套依赖矩阵冻结成两份 requirements 文件 + 一个 patch 脚本，避免下次重建实例时再撞同一个连环坑（FSDP2 / qwen3_5 model_type / HybridCache）。

| 组件 | torch 2.5.1（当前 AutoDL 默认） | torch 2.6+（未来 / 高端集群） | 锁定原因 |
|---|---|---|---|
| `torch` | **2.5.1+cu124**（不动） | 2.6.x+ | D-013 + debug_log Issue 13：flash-attn 2.x / bitsandbytes / fla 全栈靠 2.5 编译 |
| `ms-swift` | **==4.2.0**（hard pin）+ FSDP2 patch | `>=4.2.0` 浮动 | 4.2.0 有 `--tuner_type` / `--enable_thinking` / `--add_non_thinking_prefix` / `--loss_scale ignore_empty_think` 四个 Qwen3.5 必备 CLI flag；4.1.x 老布局没有 |
| `transformers` | **==5.2.\***（two files 共同） | **==5.2.\*** | `qwen3_5` model_type 5.0 才注册；5.3.\* 视频 dataloader 坏（per `materials/qwen3.5_key_points.docx`） |
| `peft` | **`pip install -U peft`** (≥ 0.17, 自动) | 同左 | transformers 5.x 删了 `HybridCache`，peft < 0.17 硬编码引用会 ImportError（debug_log 复用经验 28） |
| `qwen_vl_utils` | **>=0.0.14** | 同 | Qwen3.5 是 mixed-thinking VL 模型，加载时 sanity check |
| `liger-kernel` | >=0.5.0 | 同 | `--use_liger_kernel true` 省 10-20% VRAM |
| `bitsandbytes` | >=0.45.0（实测 0.49.2） | 同 | QLoRA 4-bit |
| `flash-attn` | 可选（不在 r.txt） | **2.8.3** 推荐 | torch 2.6+ wheels 充足、Ampere+ 加速 |
| FSDP2 patch | **必跑** `scripts/patch_swift_fsdp2.py` | 不需要 | ms-swift 4.2 import `torch.distributed.fsdp.FSDPModule`（2.6+ API） |
| `nbstripout` | **必装** + `nbstripout --install` | 同 | 防 JupyterLab autosave vs git pull 死锁（复用经验 29） |

- **替代（被否决）**：
  1. **升 torch 到 2.6+，跟着 ms-swift / flash-attn 全栈升** → flash-attn 2.x 编译 wheel 不齐全 / bitsandbytes 不一定兼容；D-013 + Issue 13 已选 2.5 stable 路线，重新整套验证 1-2 天，价值不大
  2. **降 ms-swift 到 < 4.2** → 实测掉到 `swift/llm/` 老布局，所有 v3.6+ CLI flag 消失（debug_log Session 12 调研），等于回到 Issue 9 初态
  3. **不锁版本，每次按需 `pip install -U`** → 已被 Session 12 反复证伪：peft / transformers 不同步升级时 import chain 必爆
- **理由**：
  1. 这套版本组合在 Session 12 实测能让 ms-swift 4.2 + Qwen3.5-4B QLoRA SFT 跑起来（step 1 loss 0.1146 → step 35 loss 0.010 / VRAM 11.4 GB / ~19 s/step）。是 Phase 5 训练的可重现 baseline。
  2. `requirements-torch26.txt` 提前准备好的，等 AutoDL 镜像升 torch 2.6 后只需切文件，不用从零调研。
  3. 把 `nbstripout` 列为依赖而非"建议"，是因为 Session 12 这次 4-loop 死锁浪费了 30 分钟；自动化 0 维护成本，新人 onboarding 不会再撞。
- **关联**：`requirements.txt` / `requirements-torch26.txt` / `scripts/patch_swift_fsdp2.py`；debug_log 复用经验 27-29；optimization_plan §10 决策日志 2026-05-12 "Env stack locked" 一行。

### D-018：SFT 数据设计 v2 — 重平衡后定型（2026-05-12 PM，Phase 4 失败后重做）

- **决策**：Phase 4 SFT 训练数据 (`outputs/sft_data/sft_train_v2.jsonl`) 锁定为 1567 条记录、NEI 占比 38.7%，由以下 `build_dataset` 配置生成：

  ```python
  # src/build_stage0.py
  _TRAIN_WEAK_BUCKETS = {
      ("scenario", "nei_underspec"): 2,       # real NEI 适度 oversample
      ("scenario", "disputed_conflict"): 3,   # DISPUTED 仍弱，加强
      ("scenario", "refutes_clear"): 2,       # 不变
  }
  # train kwargs: k=20, pad_with_random=True, n_hard_neg=0, apply_curriculum=True
  ```

- **失败的替代方案 v2-cut-1（已被驳回）**：
  - 配置 `nei_underspec ×4` + `disputed_conflict ×2` + `n_hard_neg=1`
  - 实测训练数据 79.1% NEI（远超 gold 33%）
  - SFT 训完 Track 3 评估：HM 0.140 < Track 2 baseline 0.201（−6pp）；
    predicted NEI 占比 92.6%，non-NEI acc 0.062 —— **majority-class collapse**
  - 根因：`n_hard_neg=1` 在所有 claim 上加 synth NEI，weak_buckets 又把 nei_underspec ×4
    → 双重放大，合计 NEI 来自 1212 (real ×4) + 2083 (hard-neg × scale) = 3295 / 4166 = 79%
  - 详见 `debug_log.md` 复用经验 32

- **为什么去 `n_hard_neg`**：
  - 原意是教模型"看到 off-topic ev → NEI"（§0.5.3 hard constraint 1）
  - 但 `nei_underspec ×2`（606 条 real）已经提供这个信号
  - `n_hard_neg=1` 双倍 NEI 是无意识的代价，每个 real claim 强制配一个 NEI synth → 大类倾斜
  - 等 Phase 5 / 6 评估如果显示 NEI acc 不达标，**再考虑回到 `n_hard_neg=0.5`** 之类（每两条 real 配一条 hard-neg）

- **理由**：
  1. 重建后 label distribution 各 ratio 在 [0.63×, 1.89×] gold 区间，sanity check 静默通过
  2. 总记录 4166 → 1567 (×2.7 缩小)，训练时间 4h → ~1.5h，迭代成本降低
  3. 失败的 v2-cut-1 在 PROGRESS Session 13 留档，避免未来重复试错
- **复用 sanity check**：`src/build_stage0.py:_print_label_dist()` 在每个 split build 完打印 SFT vs gold 比对表 + ratio > 2× 或 < 0.5× warn。
- **关联**：debug_log 问题 32；optimization_plan §4.4 (v2 revision)、§10 决策日志 2026-05-12 PM 两行；`src/build_stage0.py`。

### D-019：retrieval-first 战略转向（2026-05-12 PM 晚，v3-rebalanced 也失败后）

- **决策**：**暂停 SFT 训练迭代，先把 retrieval recall@20 从 0.333 拉到 ≥ 0.50**，然后才回头训 SFT。Phase 5 SFT 工作流的执行顺序从 "数据 → 训练 → 评估" 改为 "**检索优化 → 数据 → 训练 → 评估**"。

- **触发事件**：v3-rebalanced SFT 实测 Track 3 HM **0.140**，跟 v2-cut-1 的 0.140 完全持平。把训练 NEI label 占比从 79% 砍到 38.7%（×2 大改）没改变 predicted NEI 94% 的塌缩行为，否决了"class imbalance 是根因"假设（D-018）。

- **新的根因诊断**（debug_log 复用经验 34）：

  ```
  pad_with_random=True 让非 NEI 训练样本 = "1-5 条 gold + 15-19 条 random ev"
                                       = ~75-95% 噪声 evidence

  RAG @ recall@20 = 0.333 推理时 = "~6.6 条 gold + 13.4 条 noise"
                              = ~67% 噪声 evidence

  → 推理时输入分布跟训练时 NEI 样本（5 条全无关 gold）更接近，
    跟非 NEI 训练样本（少数 gold + 多数 noise）也无法区分
  → 模型学到捷径："noise-heavy → output NEI"
  ```

  **representation alignment 问题，不是 label distribution 问题**。改 label 占比不影响 input space 的混叠。

- **被否决的替代**：
  1. **继续调 weak_buckets**（已试两次 79% / 39%，结论：class balance lever 已经 saturated，再调没用）
  2. **改 SFT 超参（lr / epochs / lora_rank 缩小）** —— 治标，但解决不了 representation alignment
  3. **接受 Track 2 baseline 不做 SFT** —— v1 HM 0.201 离 Phase 5 目标 ≥ 0.28 有距离；保留 SFT 路径价值大

- **新执行顺序**：

  ```
  Phase 3.5b retrieval 深度审计 (optimization_plan §3.5b)
       ↓
  scripts.retrieval_ceiling --mode retriever,fusion_w,synonym_expand
       ↓
  ┌─ recall@20 ≥ 0.50 ─→ 锁新 RetrievalConfig
  │                        ↓
  │                      build_stage0 --force (新 retrieval 重 build SFT data)
  │                        ↓
  │                      run_sft.py 重训 (~1.5h)
  │                        ↓
  │                      Track 3 评估
  │
  └─ recall@20 < 0.40 ─→ 写 scripts/rewrite_queries.py (HyDE + sub-claim)
                          ↓
                        RetrievalPipeline 加 multi-query
                          ↓
                        重 audit → 验证 recall 提升 → 同上路径
  ```

- **同时考虑 `pad_with_random=False`**（D-019 副推论）：build_dataset 训练时不用 random padding，仅用真实 gold ev。优点：训练分布更干净，模型不会学到"noise → NEI"。缺点：训练 evidence 数量 < 推理时 k=20，distribution 仍不一致但方向相反（训练少 noise，推理多 noise）。**等 retrieval 改进后再实验 toggle**，作为 Phase 5 ablation 一项。

- **报告价值**（README §307 "clear insight into why method works/fails"）：
  - 两次 SFT 失败 + retrieval-first 转向是论文 Results 章节的金牌 narrative
  - 比"SFT improved HM by +0.08"更有 publishable insight
  - 配合 debug_log 复用经验 22 (F-score ceiling) + 32 (class balance trap) + 34 (representation alignment) 三连
- **关联**：debug_log 复用经验 32 + 34；optimization_plan §3.5 / §4 / §10 决策日志 "战略转向 retrieval-first" 行；`src/build_stage0.py` 待加 `pad_with_random=False` 选项。

**D-019 进展 (2026-05-12 PM 晚)**：retrieval-first 第一步出乎意料 ——
`scripts.retrieval_ceiling --mode retriever` 发现 **bge-reranker-base 是
负贡献**（recall@5 0.119 with rerank vs 0.200 without，×1.68 worse）。
锁 `RetrievalConfig.use_rerank = False`（复用经验 35）。这是免费的
+0.02 HM 预期改进。但 recall@20 还只 0.360 < 0.50 目标 → 仍需要 LLM
rewrite (HyDE + sub-claim) 进一步提升。

**D-019 进展 (2026-05-13 AM)**：retrieval-side 全部尝试完毕，**ceiling
锁定在 recall@20 = 0.357**：

- no-rerank Track 2 实测：HM 0.201 → **0.213**（+0.030 over original k=5；
  Win 1 兑现）
- `scripts.retrieval_ceiling --mode llm_rewrite --no-rerank` 四档对比：
  - baseline (claim only) recall@20 = 0.357
  - HyDE only = 0.357（持平）
  - sub-claims only = 0.331（退步）
  - HyDE + sub-claims = 0.339（退步）
  - **recall@50/100 上 HyDE+sub 反超 +0.044 / +0.058**，但 SFT 用 final_k=20
    取不到这部分增益
- HyDE 在 climate domain 上贡献在 long-tail recall，不在 top-k precision
  （跟 MS MARCO 文献相反，原因是 domain shift + fused-no-rerank baseline
  已饱和 top-5）。详见 debug_log 复用经验 36。

**战略转向 retrieval-first → data-format-second（D-019 副推论现已主推论化）**：

retrieval ceiling 触底后，回到 `src/build_stage0.py` 改 train kwargs
`pad_with_random=True → False`：

- 旧：非 NEI 样本 1-5 gold + 15-19 random ev (~96% noise/sample)
- 新：非 NEI 样本仅 1-5 gold ev (0% noise)，**移除"noise → NEI"shortcut**

本地实测 n_shown 分布从 100% @ 20 → 1: 14% / 2: 18% / 3: 14% / 4: 9% /
5: 45%（NEI claims 自带 5 gold off-topic，比例最大）。Label 分布完全不变
（pad 不动 label）。

**等待 SFT v6 AutoDL 实测结果**：

| Track 3 v6 结果 | 含义 | 行动 |
|---|---|---|
| HM > 0.213 + non-NEI acc ≥ 0.45 | alignment fix 真救了 SFT | 进 DPO + Track 4 |
| HM 0.18-0.21 (持平 baseline) | 部分救场，提升不显著 | 可选进 DPO 看 +0.02 |
| HM < 0.16 (仍塌) | retrieval + alignment 都试过了，**真天花板** | **接受 Track 2 v1 HM 0.213 作 final** |

无论结果如何，**3 个 failure story 已经构成 ACL Results 章节主线**：
prompt → retrieval → SFT alignment。报告 publishable 无悬念。

---

## 18. 术语表

| 术语 | 含义 |
|---|---|
| RAG | Retrieval-Augmented Generation |
| BM25 | 经典稀疏检索算法（IDF + 词频） |
| Dense retrieval | 基于句嵌入的稠密检索 |
| Cross-encoder | 同时输入 (query, candidate) 做联合编码的 reranker |
| RRF | Reciprocal Rank Fusion，`1/(60+rank)` 求和 |
| HyDE | Hypothetical Document Embeddings，让 LLM 先写"假设证据"再去检索 |
| QLoRA | 4-bit base + LoRA adapter，PEFT 显存友好方案 |
| LoRA | Low-Rank Adaptation，往 attention/FFN 加低秩 adapter |
| ViT | Vision Transformer，多模态模型的视觉塔 |
| SFT | Supervised Fine-Tuning |
| DPO | Direct Preference Optimization，无 reward model 的偏好对齐 |
| RLHF | Reinforcement Learning from Human Feedback |
| NLI | Natural Language Inference（蕴含/矛盾/中立） |
| FEVER | Fact Extraction and VERification 数据集（**禁用**） |
| diag_test | 我们从 train 切出的诊断 hold-out（121 条） |
| dev_holdout | 我们从 train 切出的内部验证集（121 条），DPO 配对来源 |
| official_dev | 官方 `data/dev-claims.json`（154 条），最终汇报用 |
| 标签条件 k | 按预测标签动态调 retrieval top-k |
| ms-swift | ModelScope SWIFT，阿里官方 LLM 训练框架 |
| GatedDeltaNet | Qwen3.5 用的 linear attention 架构组件，需 fla + causal-conv1d 提速 |
| Liger kernel | 一组 fused MLP/RoPE 等 PyTorch kernel（HazyResearch），SFT 时省 ~10-20% 显存 |
| BatchEncoding | transformers 5.x `apply_chat_template(return_tensors="pt")` 的返回类型，dict-like，**不是 tensor** |
| messages 格式 | ms-swift 的标准数据 schema，`{"messages": [{"role": "...", "content": "..."}, ...]}`（per docx §2.1）|
| 思考模式三件套 | Qwen3.5 mixed-thinking 模型的训推一致开关：`--enable_thinking false` / `--add_non_thinking_prefix true` / `--loss_scale ignore_empty_think` |
| AutoDL | 国内 GPU 租用平台，本项目主用 4080 SUPER 实例 |

---

**文档版本**：v1.1（2026-05-11，session 4 收口）
- v1.0 → v1.1 主要增量：§4 系统总览图修正 Stage 3；§5.3 SFT 数据格式迁移到 messages；§8 Stage 3 完全重写（Qwen3.5 是 VL + GatedDeltaNet 模型；T4/Ampere+ 双路径；CLI 参数全更新；新增 §8.2/8.5/8.6）；§9.3 DPO CLI/schema 同步；§16 风险登记新增 4 条 + 调整 4 条；§17 新增 D-011/012/013/014；§18 术语表新增 7 条

**作者**：Group _<fill in>_
**关联文件**：
- 完整方案（plan-mode 双语版）：`~/.claude/plans/fancy-mapping-lemur.md`
- 调试日志（含 session 1 + session 2）：`debug_log.md`
- 进度日志：`outputs/PROGRESS.md`
- 自检报告：`outputs/dry_run_report.md`
- 参考材料：`materials/{swift_training,qwen3.5_key_points,训练数据格式}.docx`（gitignored）
