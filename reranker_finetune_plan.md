# Reranker 微调计划 / Reranker Fine-tuning Plan

> **目的**：把当前在 climate 域上**负贡献**的 `bge-reranker-base`
> 通过有监督微调改造成**正贡献**的 reranker，**主目标是把
> recall@20 从 0.357 拉到 ≥ 0.40**（理想 ≥ 0.50 解锁 SFT），次要
> 目标是端到端 Track 2 HM 从 0.213 → ≥ 0.235。
>
> **Goal**: turn the off-the-shelf `bge-reranker-base` (currently
> harming retrieval on climate, ×1.68 worse recall@5) into a net
> positive by supervised fine-tuning. **Primary target: lift
> recall@20 from 0.357 → ≥ 0.40** (ideal ≥ 0.50 to unlock SFT
> retraining). Secondary: end-to-end Track 2 v1 HM 0.213 → ≥ 0.235.
>
> Status: design document. Implementation tracked in
> `optimization_plan.md` §10 once execution starts.
>
> 最后更新 / Last update: 2026-05-16

---

## 0. TL;DR

- **数据 / Data**：986 claims × (gold ev as positives) + (top-50 fused,
  non-gold as hard negatives). InfoNCE list-wise loss.
- **模型 / Model**：bge-reranker-base + LoRA(r=8, α=16). 全 fp16, 4080
  SUPER 上 ~30 min 训完。
- **评估 / Eval**：`scripts/retrieval_ceiling.py --mode retriever` 跑
  diag_test 看 recall@{5,10,20,50,100} 全 k 单调 ≥ baseline 才锁。
  dev_holdout 只看 **1 次**（design.md D-006 配额）。
- **风险 / Risk**：986 claims 偏小，LoRA + 早停 + 3 seed 集成做兜底。
- **预算 / Budget**：~4.5h 单线（含数据 prep + 训练 + 评估 + 集成）。

---

## 1. 动机 / Motivation

`outputs/eval_phase1/retrieval_ceiling_diag_test.md` 关键发现 (2026-05-12 PM)：

| Mode | macro recall@5 | recall@20 |
|---|---|---|
| `fused (no rerank)` | **0.200** | 0.360 |
| `full (fused + rerank)` | 0.119 | 0.333 |

`bge-reranker-base` 在 climate 域**把 gold 证据从 top-k 推出去**，×1.68
worse on @5。当前对策是 `use_rerank=False` 锁死（pipeline.py:34）。

但 retrieval-side 其他路（HyDE/sub-claim, synonym expand, fusion-weight
sweep）全堵：

- HyDE+sub-claim recall@20 = 0.339 < 0.357 baseline；只在 @50/@100 有 +0.044/+0.058
- WordNet synonym expand: **−0.004**
- Best fusion w_bm25 = 0.7：recall@5 0.154 < no-rerank 0.200

**recall@20 当前 ceiling 0.357**，距离 D-019 写的 ≥ 0.50 解锁
SFT 还差 ~0.14 absolute。retriever 架构不动的前提下唯一剩下的杠
杆是 **rerank stage 本身**——把当前坏 reranker 改造成好的能不能拉 +0.05?
基于 BEIR / MS-MARCO 经验：domain-tuned cross-encoder vs zero-shot
cross-encoder 在 long-tail 域上 +0.05~0.10 nDCG@10 是常见现象。
climate 是典型 long-tail 域（XLM-R 预训练几乎不见），FT 上界乐观。

**Key finding**: `bge-reranker-base` pre-trained on MS-MARCO actively
demotes gold climate evidence. Other retrieval-side levers
exhausted; the rerank stage is the last knob. Domain-tuned
cross-encoders on long-tail domains typically lift recall +0.05-0.10
absolute, so FT'ing this specific stage has high expected ROI.

---

## 2. 数据设计 / Data Design

### 2.1 训练样本构造 / Training row construction

每条训练 row = **1 query + 1 positive + N hard negatives**（list-wise）。

```jsonl
{
  "claim_id": "claim-126",
  "claim_text": "El Niño drove record highs ...",
  "candidates": [
    {"ev_id": "evidence-338219", "text": "...", "label": 1},   // gold
    {"ev_id": "evidence-X1",     "text": "...", "label": 0},   // hard neg
    {"ev_id": "evidence-X2",     "text": "...", "label": 0},
    ...                                                          // total N=7 negs
  ]
}
```

- **Source**: `outputs/splits/train_split.jsonl` (986 claims, 已带 gold ev id)
- **Positives**: 每个 gold ev 各拆一条 row（gold ev 平均 ~2.1/claim，所以总 rows ≈ 986 × 2.1 ≈ **2070 lists**）
- **Hard negatives**: 用当前 production retriever（**fused, no rerank, no rule_reorder**）跑每个 claim 取 top-50，排除全部 gold ev → 取 top-7 作 hard negs
- **List size**: 1 pos + 7 negs = **8 candidates per list** → 总 pairs ≈ 16,560

Each row = 1 query + 1 positive + N=7 hard negatives, list-wise. Source
is the existing `train_split.jsonl`. Hard negatives come from the
**current production retriever in no-rerank mode**, top-50 minus gold
→ take top-7. About 2070 lists, ~16.5k pairs total.

### 2.2 Hard-negative quality 过滤 / Quality filter

要避免训练分布 vs 推理分布 mismatch（D-019 的核心教训），hard-neg
必须来自 **真正会出现在 inference top-50 的 evidence**，不能是随机
负样本：

| 规则 / Rule | 阈值 / Threshold | 理由 / Reason |
|---|---|---|
| 来自 `fused (no rerank)` top-50 | strict | reranker inference 也只看这 50 |
| 不在该 claim 的 gold ev 集 | strict | 防止 false negative |
| 每个 ev 跨全数据集出现 ≤ 5 次 | soft cap | 防止"climate" 这种通用句被反复用作负 |
| Claim 至少有 7 个 valid hard negs | strict | 不够就丢这条 claim（应该 < 1%） |

**Open question**：gold 集本身可能不完整（NLP 标注通病）。如果 hard-neg
里有"实际上也支持/反驳 claim 但没被标"的 ev，是一个 false negative。
缓解：抽 10 条 claim 人眼 spot check + 训练加 label smoothing 0.05。

To match the inference distribution (D-019 lesson), hard negatives must
come from the actual top-50 the reranker sees in production
(`fused, no rerank`). Gold-leak risk mitigated by spot-checking 10
random hard negs and label smoothing 0.05.

### 2.3 Split 边界 / Split boundaries

| Split | n_claims | 用途 / Use |
|---|---|---|
| `train_split` | 986 | 训练数据 / training |
| `diag_test` | 121 | 训练时 eval（每 200 steps）/ training-time eval |
| `dev_holdout` | 121 | **最终一次性确认 / final one-shot confirmation** |
| `official_dev` | 154 | Phase 6 才碰，不进 reranker FT 流程 |

`dev_holdout` 只在 reranker 锁版本后看 **1 次**，吃掉 D-006 配额的 1/3-1/4。

### 2.4 数据 prep 脚本 / Data prep script

```bash
# 新建 scripts/build_reranker_ft_data.py
# 输入：train_split + evidence corpus + 已建 BM25/dense index
# 输出：outputs/reranker_ft_data/{train,eval}.jsonl + meta.json
python -m scripts.build_reranker_ft_data \
    --train outputs/splits/train_split.jsonl \
    --eval outputs/splits/diag_test.jsonl \
    --top-k 50 --n-negs 7 --seed 42
```

预估时间 / Estimated wall-time:
- BM25+dense retrieval 121 × 986 = ~12 min on 4080 SUPER (4-bit dense)
- Hard-neg sampling + JSONL write: <30s
- **Total ~12 min**, cache 落 `outputs/reranker_ft_data/`

---

## 3. 模型架构 / Model Architecture

### 3.1 Base + LoRA 配置 / Base + LoRA config

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForSequenceClassification.from_pretrained(
    "models/bge-reranker-base",   # XLM-RoBERTa-base, num_labels=1
    torch_dtype=torch.float16,
)

lora_cfg = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["query", "key", "value", "dense"],  # XLM-R attn + intermediate
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    modules_to_save=["classifier"],   # 关键：classifier head 也要更新
)
model = get_peft_model(model, lora_cfg)
# Trainable params: ~1.5M (~0.5% of 278M base)
```

**为什么 LoRA r=8 而不是全量 / Why LoRA r=8 over full FT**:
1. 训练数据小（2070 lists），全量 FT 极易过拟合 / Small training set, full FT overfits easily.
2. LoRA adapter ~6 MB，提交友好 / Smaller artifact for submission zip.
3. 多 seed 集成时存 N 份 LoRA 而不是 N 份全模型 / Cheap to ensemble multiple seeds.
4. 出错时可 disable LoRA 直接退回 base，rollback 0 cost / Trivial rollback if FT regresses.

**classifier head 必须 train**：bge-reranker-base 的输出 head 是 `num_labels=1`
的 linear，没有 LoRA wrap 就完全冻住，FT 只调 attention 时 score 输出
还是预训练时的语义。`modules_to_save=["classifier"]` 让它跟着 LoRA 一起
更新但记录在 adapter checkpoint 里。

### 3.2 输入格式 / Input format

```python
# 每对 (claim, candidate) 的 tokenizer 输入：
tokenizer(claim_text, candidate_text,
          truncation="only_second", max_length=512,
          padding=False, return_tensors="pt")
# truncation="only_second" → claim 不被切，evidence 被切
```

`truncation="only_second"` 是 cross-encoder 的标准做法：claim 平均 ~15
tokens，evidence p95 < 200 tokens（见 `outputs/evidence_profile.md` 待跑），
512 上限足够。

---

## 4. 损失函数 / Loss Function

### 4.1 List-wise InfoNCE

每个 list（8 candidates，1 pos + 7 neg）按下式算损失：

```python
# scores: [batch_size, 8]   (model 输出每对 (claim, cand) 的 scalar score)
# 真值：每个 list 的第 0 个位置是 positive
labels = torch.zeros(batch_size, dtype=torch.long)  # all positives at index 0
loss = F.cross_entropy(scores / temperature, labels, label_smoothing=0.05)
```

- `temperature = 1.0` (默认；FlagEmbedding upstream 也是 1.0)
- `label_smoothing = 0.05` 兜底 gold incompleteness（§2.2）

This is the same training objective as FlagEmbedding's upstream
training scripts. Each batch row has 8 candidates, model produces 8
scores, cross-entropy targets index 0 = positive.

### 4.2 为什么不用 pairwise margin / Why not pairwise margin

Margin loss (`max(0, margin - s_pos + s_neg)`) 在小数据集上对 margin 值
非常敏感（0.1 vs 0.3 差很多），InfoNCE 自带温度归一化更稳。

### 4.3 不采用 / Skipped alternatives

- **KL distillation from larger reranker (bge-reranker-v2-m3)**: 推迟。
  baseline FT 先跑通；如果 recall@20 提升 < 0.03 再上 distill。
- **Listwise RankNet/LambdaRank**: 在 8 cand 的 list 上 InfoNCE 已经
  near-optimal，多余复杂度。

---

## 5. 训练超参 / Training Hyperparameters

| Param | Value | 理由 / Reason |
|---|---|---|
| Batch size (lists) | 8 | 1 list = 8 pairs → 64 forward/step；4080 SUPER fp16 ~12 GB VRAM |
| Effective batch | 16 (GA=2) | 平滑梯度 |
| LR | 2e-5 | LoRA 标准；XLM-R 全量 FT 通常 1e-5，LoRA 略激进 |
| Schedule | cosine, warmup 10% | 短训练用 cosine 比 linear 好 |
| Epochs | 3 | 2070 lists × 3 = 6210 steps；GA=2 → 3105 optimizer steps |
| Max seq | 512 | reranker 默认 |
| Eval every | 200 steps | recall@20 on diag_test |
| Early stop | patience 3 evals | 防过拟合（重要！数据小） |
| Weight decay | 0.01 | LoRA 标配 |
| Grad clip | 1.0 | 防梯度爆炸 |
| Seed | 42 / 1337 / 2024 | 3 seed 训完看是否需要集成 |

**预估时间**：4080 SUPER fp16，XLM-R-base @ 512 seq，BS=8 lists × 8 cands = 64 forward，
~0.25 s/step → 3000 steps × 0.25s ≈ **12 min/seed**，3 seed ≈ **36 min**.

Estimated wall-time: ~12 min per seed on RTX 4080 SUPER. 3 seeds = ~36 min total.

---

## 6. 训练脚本 / Training Script

新建 `scripts/finetune_reranker.py`（~250 行），结构：

```python
# Pseudo-code
def main():
    cfg = parse_args()
    
    # 1. Load data (cached from build_reranker_ft_data.py)
    train_ds = load_jsonl("outputs/reranker_ft_data/train.jsonl")
    eval_ds  = load_jsonl("outputs/reranker_ft_data/eval.jsonl")
    
    # 2. Load model + LoRA
    tokenizer = AutoTokenizer.from_pretrained("models/bge-reranker-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "models/bge-reranker-base", torch_dtype=torch.float16, num_labels=1,
    )
    model = get_peft_model(model, lora_cfg)
    
    # 3. Collator: flatten list-of-8 to flat batch, remember list boundaries
    collator = ListwiseCollator(tokenizer, max_length=512, n_cands=8)
    
    # 4. Custom Trainer with InfoNCE loss
    class RerankerTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            scores = model(**inputs).logits.squeeze(-1)         # [B*8]
            scores = scores.view(-1, 8)                          # [B, 8]
            labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
            loss = F.cross_entropy(scores, labels, label_smoothing=0.05)
            return (loss, scores) if return_outputs else loss
    
    # 5. Recall@k callback at every eval step
    class RecallAtKCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, model, **kwargs):
            recall = compute_recall_at_k(model, eval_ds, k_list=[5, 10, 20, 50])
            state.log_history.append({"step": state.global_step, **recall})
    
    # 6. Train
    trainer = RerankerTrainer(...)
    trainer.train()
    
    # 7. Save best ckpt
    trainer.save_model(f"models/bge-reranker-base-ft/lora-seed-{cfg.seed}")
    
    # 8. Merge LoRA into base for production
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(f"models/bge-reranker-base-ft/merged-seed-{cfg.seed}")
```

集成方式（如果单 seed 不够）/ Ensembling (if single seed underdelivers):
平均 3 seed 的 LoRA adapter weight（在 merged 之前），写
`scripts/avg_lora.py`。或推理时跑 3 model 取均值（昂贵 3×，仅在数字
显著靠近门槛但跨不过时用）。

---

## 7. 评估 / Evaluation

### 7.1 训练时 eval / Training-time eval

每 200 steps 在 `diag_test` 上跑：

```python
# 简化：直接用训好的中间 ckpt 在 (claim, top-50 cands) 上 rerank，
# 算 recall@{5,10,20}
def compute_recall_at_k(model, eval_data, k_list):
    """对每条 eval claim：
       1) 取已 cache 的 top-50 fused candidates
       2) reranker 重排
       3) 看 gold ev 在重排后 top-k 的 hit 率
    """
```

记入 `outputs/reranker_ft_log_seed-{N}.jsonl`，最后画 recall@20 曲线挑
best ckpt。

### 7.2 训练完整体 / Post-training audit

```bash
# 跑完后，用 retrieval_ceiling 复现 §1 表格、加新 reranker 列
python -m scripts.retrieval_ceiling \
    --dataset diag_test \
    --mode retriever \
    --reranker-path models/bge-reranker-base-ft/merged-best
```

期望产出 / Expected output: 一张 4 列对比表：
- baseline `fused (no rerank)`
- old `full (fused + bge-reranker-base)`
- **new `full (fused + bge-reranker-base-FT)`**

### 7.3 端到端 Track 2 eval / End-to-end

```bash
# 临时把 use_rerank 改回 True + reranker_path 指向新 ckpt
python -m scripts.phase1_eval \
    --tracks 2 --prompts v1 --dataset diag_test \
    --reranker-path models/bge-reranker-base-ft/merged-best \
    --use-rerank
```

跟 Track 2 v1 baseline HM 0.213 比。

### 7.4 锁定 / Lock criteria

**Success（→ 锁新 reranker as production default）**:
- recall@5 ≥ 0.20（不退化 vs no-rerank baseline）
- recall@20 ≥ 0.40（+0.043 over 0.357）
- 端到端 Track 2 v1 HM ≥ 0.235（+0.022）
- **且** 所有 k ∈ {5, 10, 20, 50, 100} 上 ≥ no-rerank baseline（不要按下葫芦浮起瓢）

**Partial success（→ enable rerank 但不重训 SFT）**:
- recall@20 ∈ [0.37, 0.40]，HM ∈ [0.215, 0.235]：值得开但不够解锁 SFT v6

**Failure（→ 留 ablation，保持 use_rerank=False）**:
- recall@20 < 0.37：FT 没救出来，记录在 design.md D-020（待加）作 negative result

### 7.5 dev_holdout 最终确认（只 1 次）/ One-shot dev_holdout confirmation

只在 §7.4 判定 success 后跑：
```bash
python -m scripts.retrieval_ceiling --dataset dev_holdout --mode retriever \
    --reranker-path models/bge-reranker-base-ft/merged-best
python -m scripts.phase1_eval --tracks 2 --prompts v1 --dataset dev_holdout \
    --reranker-path models/bge-reranker-base-ft/merged-best --use-rerank
```

如果 dev_holdout 上 recall@20 比 diag_test 落 > 0.05 absolute，说明 ft
过拟合了 diag_test → 回退。

This single dev_holdout check consumes 1 of the 3-4 allowed
peeks per D-006. Skip entirely if §7.4 fails.

---

## 8. 集成 / Integration

成功后修改 / On success, modify:

1. **`src/retrieval/rerank.py`**: `DEFAULT_RERANKER` 改成
   `"models/bge-reranker-base-ft/merged-best"`（相对路径 + `resolve_model_path` 兼容）
2. **`src/retrieval/pipeline.py`**: `RetrievalConfig.use_rerank` default
   `False → True`；新增 docstring 记录 FT 来源
3. **`scripts/phase1_eval.py`**: 加 `--reranker-path PATH` flag（已有
   `--use-rerank` 类似 flag 模式，沿用）
4. **`scripts/build_indexes.py`**: 不动（reranker 不进索引）
5. **`src/build_stage0.py`**: 如果 §7.4 success，重 build SFT 数据让训练
   分布跟新 retrieval 输出一致（`python -m src.build_stage0 --force`）
6. **`models/.gitignore`**: 加 `bge-reranker-base-ft/merged-*` 例外（LoRA
   adapter 文件 ~6 MB 可入仓）；merged 全模型留本地

---

## 9. 风险与缓解 / Risks & Mitigations

| 风险 / Risk | P | 缓解 / Mitigation |
|---|---|---|
| **986 claims 太小 FT 过拟** | high | LoRA r=8（限制容量）+ early stop（patience 3）+ 3 seed 集成 + label smoothing 0.05 |
| **Hard-neg leak**（gold incompleteness 把支持证据误标为 neg） | med | spot check 10 条 + label smoothing 0.05 + 训完看 train loss 是否异常低（<0.1 警示） |
| **FT 拉 @5 但塌 @20**（学到 over-promote top hits） | med | 训练时每 ckpt 看完整 recall@{5,10,20,50,100}，只锁全 k 单调 ≥ baseline 的 ckpt |
| **Train/inference 分布 mismatch**（D-019 教训重演） | low | Hard-neg 严格来自 `fused (no rerank)` top-50，就是 reranker 推理时看到的同一分布 |
| **diag_test 过拟合**（早停指标污染） | med | dev_holdout 留作最终确认；如果 dev 上 recall 落 > 0.05 → 回退 |
| **训练时间超预算**（3 seed × 12 min = 36 min 不算长，但加 prep + eval 容易膨胀） | low | 先单 seed 跑通，确认数字再上 3 seed |
| **AutoDL 实例丢失** | med | LoRA adapter + ft_data 全部 push 到 git（adapter ~6 MB，data ~10 MB 压缩后；如果太大 scp 到本地备份） |
| **bge-reranker-base 本身能力天花板低**（XLM-R-base 太小） | low-med | 若 FT 后 recall@20 < 0.37，作为 ablation 终结，不上 bge-reranker-v2-m3（568M, 4080 SUPER 训得动但调参成本高） |

---

## 10. 时间预算 / Time Budget

| Stage | 任务 | Time | Cumulative |
|---|---|---|---|
| 1 | 写 `build_reranker_ft_data.py` + cache hard negs | 1.0 h | 1:00 |
| 2 | 写 `finetune_reranker.py` + ListwiseCollator + RecallCallback | 2.0 h | 3:00 |
| 3 | Train seed 42 (smoke) + 看 loss/recall 曲线 | 0.5 h | 3:30 |
| 4 | Train seeds 1337 + 2024（如果 seed 42 OK） | 0.5 h | 4:00 |
| 5 | Eval (retrieval_ceiling + phase1_eval Track 2) | 0.5 h | 4:30 |
| 6 | 集成（pipeline.py / rerank.py / phase1_eval flag） | 0.5 h | 5:00 |
| 7 | dev_holdout final check（仅成功时） | 0.25 h | 5:15 |
| **Total** | | **~5 h** | |

并行机会 / Parallelism: Stage 2 写脚本时 Stage 1 数据 prep 已经在 AutoDL
跑，bash background。

---

## 11. 决策节点 / Decision Gates

### Gate A — Stage 3 后（seed 42 训完）

| 现象 | 行动 |
|---|---|
| diag_test recall@20 ≥ 0.40 | 进 Stage 4-5（3 seed 集成 + 端到端 eval）|
| recall@20 ∈ [0.36, 0.40] | 改 LR (2e-5 → 5e-5 或 1e-5) 重跑 seed 42 一次 |
| recall@20 < 0.36 OR train loss → 0 | 停！数据问题/过拟。回 Stage 1 看 hard-neg 质量 |

### Gate B — Stage 5 后（端到端）

| 现象 | 行动 |
|---|---|
| Track 2 HM ≥ 0.235 AND 所有 recall@k ≥ baseline | Lock，Gate C |
| HM ∈ [0.215, 0.235] | enable rerank as default，但 SFT 留 ablation 不重训 |
| HM < 0.215 OR 任意 k 退化 | 回退 `use_rerank=False`，FT 全留 ablation |

### Gate C — dev_holdout final（仅 Gate B success）

| 现象 | 行动 |
|---|---|
| dev recall@20 ≥ diag recall@20 − 0.03 | 真正 locked，写决策日志，进 §12 后续 |
| dev recall@20 比 diag 落 > 0.05 | 过拟合 diag_test。回退至 v1 reranker 关闭模式 |

---

## 12. 衍生 / Follow-ups (only if Gate C success)

如果 recall@20 实测达到 / If recall@20 actually reaches:

- **≥ 0.45** (unlocking SFT): 重建 SFT v6 数据用新检索 + 训 SFT (pad_with_random=False
  已经在 `src/build_stage0.py` 改好) → 目标 Track 3 HM ≥ 0.28
- **0.40-0.45** (marginal): 不重训 SFT；尝试 DPO 对 DISPUTED 对抗对训练 →
  目标 Track 4 HM ≥ 0.24
- **< 0.40 但 > baseline**: 接受新 baseline，直接进 Phase 6 official_dev / test

---

## 13. 关联文件 / Related files

- `design.md` D-006（dev 配额）, D-019（pad-alignment / train-inference distribution）
- `optimization_plan.md` §3.5（retrieval ceiling audit）, §10（decision log）
- `debug_log.md` 复用经验 34-36（retrieval-first 转向 + rerank 负贡献）
- `src/retrieval/{rerank,pipeline}.py` 待改文件
- `outputs/eval_phase1/retrieval_ceiling_diag_test.md` baseline 数字

---

## 14. 实施 checklist / Implementation checklist

执行时按序勾选 / Tick in order:

- [ ] 1. AutoDL 上 `source /etc/network_turbo && cd ~/autodl-tmp/NLP-A3 && git pull`
- [ ] 2. 写 `scripts/build_reranker_ft_data.py`，本地 unit test
- [ ] 3. AutoDL 跑数据 prep（~12 min），落 `outputs/reranker_ft_data/`
- [ ] 4. 写 `scripts/finetune_reranker.py` + 本地 import smoke test
- [ ] 5. AutoDL seed 42 训练（~12 min）→ Gate A
- [ ] 6. [if Gate A pass] seeds 1337, 2024 → 看是否需要集成
- [ ] 7. 跑 `scripts.retrieval_ceiling --mode retriever --reranker-path ...` → Gate B
- [ ] 8. 跑 `scripts.phase1_eval --tracks 2 ... --use-rerank --reranker-path ...` → Gate B end-to-end
- [ ] 9. [if Gate B pass] 改 `pipeline.py` + `rerank.py` defaults，commit
- [ ] 10. [if Gate B pass] dev_holdout final check → Gate C
- [ ] 11. 写决策日志到 `optimization_plan.md` §10 + `debug_log.md` 新一条复用经验
- [ ] 12. [if Gate C pass and recall@20 ≥ 0.45] 重建 SFT v6 + 重训 → 进 §12
