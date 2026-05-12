# 优化计划 / Optimization Plan

> **本文是 design.md D-015（评估驱动的迭代式 SFT 数据扩充）的可执行落地版**。
> design.md 说"为什么这么做"；本文说"具体怎么一步步做"。
>
> **This document is the executable counterpart to design.md D-015** (eval-driven
> iterative SFT data design). design.md states the *why*; this file states the
> *what* and *when*, phase by phase.

---

## 0. 总览 / Overview

```
评估驱动的六阶段闭环 / Six-phase evaluation-driven loop:

  Phase 1  Baseline 评估     ─→  Phase 2  Prompt 迭代
  Baseline eval (Track 1+2)        Prompt iteration (v2-v4)
                                          ↓
  Phase 6  最终汇报          ←─  Phase 3  最优 Prompt + RAG 重测
  Final report (eval.py)           Locked prompt re-eval
       ↑                                  ↓
  Phase 5  SFT/DPO 训练 + 评估  ←   Phase 4  弱桶定位 + SFT 数据扩充
  Train + eval                       Weak-bucket → tilted SFT data
```

**核心原则 / Core principles**:
1. **评估先行 / Eval first** — 在 SFT 之前先压榨 prompt 和 RAG 的能力（D-015）
2. **缓存优先 / Cache first** — 所有重计算结果落 Drive / 本地，notebook 自动检测复用（§9）
3. **后验驱动 / Posterior-driven data design** — SFT 数据配比由诊断切片表决定，不靠先验
4. **不污染 dev / Never look at dev** — Phase 1-5 全部在 `diag_test`；只有 Phase 6 才碰 `official_dev`

---

## 0.5. Base 模型能力探针（先决条件）/ Base-model Capability Probe (Precondition)

> 进入 Phase 1 前已用 `scripts/test_qwen35_inference.py` 在 AutoDL
> 4080 SUPER（4-bit NF4）对 base **Qwen3.5-4B** 做了 4 项探针测试，
> 结果作为 **§4.3 SFT 数据扩充策略的硬约束** 来源。原始记录见
> `debug_log.md` 会话 2 "实测数据" 段。
>
> Before Phase 1 we ran `scripts/test_qwen35_inference.py` on AutoDL
> (RTX 4080 SUPER, 4-bit NF4) to probe four capability dimensions of
> the base **Qwen3.5-4B**. Findings are the source of the **hard
> constraints** that drive §4.3 SFT data-augmentation strategy. Raw
> log: `debug_log.md` Session 2 "实测数据".

### 0.5.1 测试条件 / Test setup

| 维度 / Dimension | 值 / Value |
|---|---|
| 硬件 / Hardware | RTX 4080 SUPER 31.5 GB VRAM |
| 模型 / Model | Qwen3.5-4B base, NF4 4-bit + double-quant |
| dtype | bf16（Ampere+ 自检 / auto-detected） |
| VRAM 占用 / VRAM footprint | **2.9 GB**（4-bit 加载后 / post 4-bit load） |
| 首次加载 / First load | 7.9 s（首次从 ModelScope 下载 ~9 min / 8 GB） |
| 样本 / Sample claims | 3 manual (SUPPORTS / REFUTES / NEI) + 1 DISPUTED w/ mixed ev |

### 0.5.2 关键发现 / Key findings

| 区段 / Section | 设置 / Setup | 发现 / Finding |
|---|---|---|
| **4a no-RAG, greedy** | base prompt + 3 manual claims | SUPPORTS ✓ / REFUTES ✓ / **NEI → REFUTES ✗**（base 模型无 "我不知道" 概念） |
| **4b RAG fake-ev, greedy** | system prompt + 3 numbered evidences | 严格输出 `LABEL ##[1,2]##` 格式 — prompt 语法 base 模型已会跟 |
| **4c SC, easy SUPPORTS** | 5 samples @ T=0.7, top_p=0.9 | **5/5 一致** → SC 在易题上零增益 |
| **4c SC, DISPUTED + mixed ev** | 同上 + 1 supporting / 1 refuting / 1 off-topic | **待 Phase 1 后重跑确认** — 期望看到 ≥2 个不同 label |

### 0.5.3 推导出的 SFT 数据三条硬约束 / Three hard constraints on SFT data

源自 §0.5.2，覆盖 §4.3 augmentation strategy / Derived from §0.5.2,
binding on §4.3:

1. **NEI 类必须 oversample —— 但要避免双重放大**（v2-revision 2026-05-12 PM）：
   base 模型对 NEI claim 强行裁决（4a 实测），SFT 训练分布需要 NEI 信号
   高于 gold 33%，但**不能超过 ~50%，否则模型塌缩到 majority class**
   （debug_log 复用经验 32：v2-cut-1 NEI 79% → Track 3 HM 0.140 < Track 2
   baseline 0.201，predicted NEI 92.6% / non-NEI acc 0.062）。

   **当前锁定（v2-revision）**：
   - `nei_underspec ×2`（real NEI 适度 oversample，606 条）
   - `n_hard_neg=0`（**禁用 hard-neg 同义放大**：之前 `n_hard_neg=1` 在
     每条 real claim 后加一条 synth NEI，被 weak_buckets factor 一起 scale，
     2083 条 synth NEI 占总数 50%，主导 majority class）
   - 实测 NEI 占比 38.7%（gold 33.1% × 1.17）

   **训练前 sanity check**：`src/build_stage0.py:_print_label_dist()`
   每个 split build 完打印 SFT vs gold ratio + warn >2× / <0.5×。

   *NEI must be oversampled — but avoid double amplification*: v2-cut-1
   used `n_hard_neg=1 + nei_underspec ×4` → 79% NEI → SFT collapsed to
   majority class. Locked v2-revision: `nei_underspec ×2`, `n_hard_neg=0`,
   resulting in NEI share 38.7% (1.17× gold). Build-time sanity check
   prevents future collapses silently.
2. **Citation 格式样本可保持稀疏**：base 模型已能严格输出
   `LABEL ##[i,j]##`（4b 实测），SFT 不需要为格式做样本增强，每条记
   录一次 demo 即够。
   *Citation-format examples stay lean* — base model already follows
   `LABEL ##[i,j]##` strictly (4b finding); no need for format-only
   augmentation, one demo per record suffices.
3. **DISPUTED / 高分歧桶优先扩充**：4c 在易题上 5/5 一致 → SC 仅在
   模糊样本上有价值（DISPUTED, scenario ∈ {refutes_partial,
   nei_topic_off}）。Phase 4 §4.3 数据扩充倍率 r：高分歧桶取上限
   r=2.0，易桶取 r=1.0。同时 Phase 5 DPO 务必包含
   `synthesise_disputed_contrast` 生成的 DISPUTED-vs-SUPPORTS 对抗对。
   *Prioritize DISPUTED-style augmentation* — SC is value-add only on
   ambiguous claims; apply r=2.0 multiplier on those buckets, keep
   r=1.0 on supports_clear-type buckets. Phase 5 DPO must include the
   DISPUTED-vs-SUPPORTS contrast pairs.

### 0.5.4 未探针的、留给 Phase 1 解决的 / Open questions for Phase 1

- 模型在 **Track 2 真 RAG** 上的端到端 HM —— Phase 1.4 输出 `track2_v1_diag_test.md` 给答案
- **DISPUTED claim 上 SC 是否拉开分歧** —— 改进版 4c 排在 Phase 1 之后立刻补一次（在 AutoDL 上跑改 prompt 或重试 base）
- **4-bit 量化对 logit 噪声**是否压低了 SC 多样性 —— 仅当 Phase 4 SC 仍无效时再排查（A/B：4-bit vs fp16）

---

## 1. Phase 1 — Baseline 评估 / Baseline Evaluation

### 1.1 目的 / Purpose

测量 **base 模型 + 当前 prompt（v1）** 在两条路径上的表现，作为后续所有改进的基线。

Measure base model + current prompt (v1) baselines on Track 1 (no-RAG) and
Track 2 (RAG), so all later improvements have a comparison anchor.

### 1.2 前置条件 / Prerequisites

| 资源 / Resource | 路径 / Path | 缺失则跑 / If missing, run |
|---|---|---|
| **所有模型权重 / all model weights** | `models/{Qwen3.5-4B,bge-m3,bge-reranker-base,bge-small-en-v1.5}/` | `python -m scripts.download_models`（一次性 ~11 GB / one-time） |
| BM25 索引 / index | `outputs/bm25_index/` | `python -m scripts.build_indexes --skip-dense` |
| Dense 索引 / index | `outputs/dense_index/faiss.index` | `python -m scripts.build_indexes` |
| `diag_test.jsonl` | `outputs/splits/diag_test.jsonl` | notebook cell 1.3 (run_splits) 或 / or `python -m scripts.dry_run` |

### 1.3 执行 / Execute

```bash
# AutoDL（4080 SUPER, ~10 分钟 / minutes）
python -m scripts.phase1_eval \
    --tracks 1,2 --prompts v1 --dataset diag_test
    # phase1_eval auto-detects models/Qwen3.5-4B/ (cache-first), no need
    # to pass --model-dir. Override only to point at a different snapshot.
```

### 1.4 产出 / Outputs

- `outputs/eval_phase1/track1_v1_diag_test.{json,md}`
- `outputs/eval_phase1/track2_v1_diag_test.{json,md}`
- `outputs/eval_phase1/summary_diag_test.md`

### 1.5 看什么 / What to read

打开 `track2_v1_diag_test.md`，per-bucket 表已按 HM 升序——**最差的桶在最上面**，那就是 Phase 4 要瞄准的目标。

Open `track2_v1_diag_test.md`; per-bucket tables are sorted by HM ascending,
so worst buckets surface at top — those are the Phase 4 targets.

### 1.6 决策点 / Decision criteria

| 现象 / Observation | 含义 / Meaning | 行动 / Action |
|---|---|---|
| Track 1 acc < 0.25 | 模型/parser 失效 / model or parser broken | 回 smoke test 排查 |
| Track 2 F < 0.05 | RAG 检索没接通 / retrieval not wired | 回 cell 2.3 检查 `pipeline.retrieve` |
| Track 2 acc > Track 1 acc + 0.10 | RAG 显著提升 / RAG helps a lot | 进 Phase 2 |
| Track 2 acc < Track 1 acc | RAG 上下文挤注意力 / RAG distracts | 仍进 Phase 2，看 prompt 能不能修 |

---

## 2. Phase 2 — Prompt 迭代 / Prompt Iteration

### 2.1 目的 / Purpose

把 **prompt 的能力压榨到极致**，看哪些"看似 SFT 才能修"的问题其实 prompt 就能修。
**Push prompts to their limit** to identify problems that look like SFT-needed but are actually fixable by prompting.

### 2.2 候选变体 / Candidates（已实现 in `src/prompt.py`）

| 版本 / Version | 动机 / Motivation |
|---|---|
| **v1** baseline | 当前 production prompt / current production |
| **v2** nei_explicit | 显式 NEI 触发条件（解决 base 把 NEI 错判 REFUTES） / explicit NEI trigger |
| **v3** disputed_explicit | v2 + 显式 DISPUTED 触发条件（解决 sample 5"识别矛盾仍 SUPPORTS"） / explicit DISPUTED trigger |
| **v4** few_shot | v3 + 4 shot 示范（每类 1 条）/ v3 + 4-shot demos |
| ~~v5 chain-of-thought~~ | 推迟 / deferred — 需先改 `parse_response` 取最后匹配 |

### 2.3 执行 / Execute

```bash
# 仅 Track 2，全部变体 / Track 2 only, all variants（~15 min on 4080）
python -m scripts.phase1_eval \
    --tracks 2 --prompts v2,v3,v4 --dataset diag_test
# Base model 由 cache-first 自动定位（models/Qwen3.5-4B/），无需 --model-dir。
```

### 2.4 决策 / Decision

打开 `summary_diag_test.md`：
- **取 Track 2 上 HM 最高的 prompt** 作为锁定版本
- 如果 v4（few-shot）只比 v3 高 < 0.005，**优先选 v3**（context 短、推理快）
- 把锁定版本写入 `optimization_plan.md` 的 §10 决策日志

### 2.5 风险 / Risks

| 风险 / Risk | 缓解 / Mitigation |
|---|---|
| Few-shot 把 prompt 撑长 → 显存/延迟 / few-shot bloats prompt | v4 实测 ~1500 字符 ≈ 400 tokens，可接受 |
| 不同 prompt 在不同桶上各有所长 / different prompts win different buckets | 看 per-bucket 表，按"最弱桶提升幅度"加权而不是只看总 HM |
| Prompt 锁定后 SFT 数据也得跟着用同一版本 / SFT data must use same prompt | `src/sft_dataset.py:build_user_query` 默认 v1，需改 default 或在 `build_dataset` 调用处显式传 version |

---

## 3. Phase 3 — 最优 Prompt + RAG 重测 / Locked-prompt Re-eval

### 3.1 目的 / Purpose

用锁定的 prompt 在 `diag_test` 上重新生成完整诊断切片，作为 Phase 4 的"靶子"。

Re-evaluate with the locked prompt to produce the diagnostic slices that
Phase 4 uses as targets.

### 3.2 执行 / Execute

如果 Phase 2 已经跑过最优 prompt 在 diag_test 上的 Track 2，**直接复用**那次的 `track2_<prompt>_diag_test.md`，**不必再跑**。

If Phase 2 already produced track2_<best>_diag_test.md, **skip — reuse it**.

### 3.3 把当前最优数字记下来 / Record the headline numbers

在本文件的 §10（决策日志）记录：
- 锁定 prompt 版本号
- Track 2 整体 (F, Acc, HM)
- 最弱的 3-5 个 bucket（domain × scenario × difficulty 任一维度）

---

## 3.5. Phase 3.5 — 检索天花板审计 / Retrieval Ceiling Audit

> **新增于 2026-05-12**：Phase 2 prompt sweep 后用 `diagnose_phase1.py`
> 发现 Track 2 v1-v4 **evidence recall 全部锁在 0.10 附近**（macro 0.10-0.11,
> micro 0.09-0.10）。F-score 的天花板由检索决定，与 prompt / SFT 无关。
> 若 retrieval recall=0.11，即使 label 100% 正确也只能拿到 F ≈ 0.12, HM ≈ 0.21。
> 在花 GPU-hour 做 SFT 之前必须先排查检索瓶颈。
>
> **Added 2026-05-12**: post-Phase-2 diagnosis showed evidence recall is
> locked at ~0.11 across all four prompt variants — the F-score ceiling
> is bound by retrieval, not by label classification. Even with perfect
> labels we cap at HM ≈ 0.21. Must audit retrieval before spending
> GPU-hours on SFT.

### 3.5.1 目的 / Purpose

定位检索瓶颈：是 (i) `final_k=5` 太紧、(ii) BM25/dense 融合权重错、
(iii) 单一 query 表征不足，还是 (iv) 检索器本身能力不够。每一项的修
复成本和上限不同，必须分别量化。

Locate the retrieval bottleneck: (i) `final_k=5` too tight, (ii)
fusion weights miscalibrated, (iii) single-query representation
insufficient, or (iv) retriever capacity ceiling. Each has different
fix cost; quantify them separately.

### 3.5.2 实验设计 / Experiments

实现在 `scripts/retrieval_ceiling.py`（无 LLM、纯检索 + recall@k 测量，
~3 min on AutoDL 4080 SUPER）。

Implemented as `scripts/retrieval_ceiling.py` (no LLM, retrieval-only +
recall@k measurement, ~3 min on AutoDL).

| 模式 / Mode | 调节维度 / Knob | 假设 / Hypothesis |
|---|---|---|
| `final_k` | 5 → 10 → 20 → 50 → 100 | 当前 5 太紧；扩到 20 可能让 recall 翻倍 |
| `retriever` | BM25-only / dense-only / fused / +rerank | 看哪个组件贡献最大 |
| `fusion_w` | w_bm25 ∈ {0.1, 0.3, 0.5, 0.7, 0.9} | 当前 0.3/0.7 偏 dense，可能 BM25 多给点效果好（claim 文本里关键词更具区分度） |
| `synonym_expand` | 单 query vs claim + WordNet 同义词 | `src/query_rewrite.synonym_expand` 已实现，多 query union 可能在词面差异大的 claim 上拉 recall |

### 3.5.3 评估指标 / Metrics

- **macro evidence recall@k**：每条 claim 算一次 `|pred ∩ gold| / |gold|`，对全部 claim 取均值
- **micro evidence recall@k**：所有 claim 的 hit 总数 / gold 总数
- **per-label recall**：分别按 SUPPORTS / REFUTES / NEI / DISPUTED gold label 看 recall（NEI 有 5 gold，其他 1-3，分开看）
- **recall@k 曲线**：k 从 5 到 100，看是否在某个 k 处饱和

### 3.5.4 决策准则 / Decision criteria

| 现象 / Observation | 含义 / Meaning | 行动 / Action |
|---|---|---|
| recall@100 仍 < 0.30 | 检索器本身就不行 | 严重：考虑加 query rewrite (HyDE/sub-claim, 需 LLM) 或换 ColBERT |
| recall@5 = 0.11 但 recall@20 = 0.35 | `final_k` 太紧 | 加 `--final-k` CLI flag 到 `phase1_eval`，重测 |
| BM25-only recall ≫ dense-only | 当前 0.3/0.7 fusion 偏错 | 调到 0.6/0.4 或更高 BM25 权重 |
| synonym expand 提升 ≥ +0.05 recall | 词面表征不足是瓶颈 | 把 multi-query 接进 `RetrievalPipeline` |
| 上面都不行 | LLM-driven 重写（HyDE / sub-claim）值得试 | Phase 3.5b：上 Qwen 跑 HyDE/sub-claim 重写 |

### 3.5.5 执行 / Execute

```bash
# AutoDL，~3 min
python -m scripts.retrieval_ceiling --dataset diag_test --mode all

# 单独跑某一项
python -m scripts.retrieval_ceiling --dataset diag_test --mode final_k
python -m scripts.retrieval_ceiling --dataset diag_test --mode synonym_expand
```

### 3.5.6 产出 / Output

- `outputs/eval_phase1/retrieval_ceiling_<dataset>.md` — 全模式对比表 + 最佳配置 callout
- 决策日志填回本文件 §10，把锁定的 `RetrievalConfig` 写进 `optimization_plan.md` 供 Phase 4 SFT 数据重新构造时使用（SFT 数据里的 retrieved evidence 必须用最终的检索配置生成，否则 train/inference 不一致）

### 3.5.7 风险 / Risks

| 风险 / Risk | 缓解 / Mitigation |
|---|---|
| 改了 `final_k` 后 prompt context 长度增加 → 推理慢 / OOM | `final_k=20` 时 evidence 拼起来 ~2000 tokens，4080 SUPER 4-bit Qwen3.5-4B 还远没到 OOM，再大才要降级 |
| Phase 4 SFT 数据要按新检索配置重建 | `python -m src.build_stage0 --force` 重跑 ~5 s，但 `outputs/sft_data/sft_train_v1.jsonl` 会被覆盖，先备份 |
| BM25/dense recall@100 都低 → 真要 LLM rewrite | HyDE/sub-claim 用 base Qwen3.5-4B 已经够好，加 ~5 min inference cost 但能跨语义鸿沟 |

---

## 4. Phase 4 — 弱桶定位 + SFT 数据扩充 / Weak-bucket Targeting

### 4.1 目的 / Purpose

把 SFT 训练样本从"按先验配比"改成"按后验弱点配比"——同样训练量，对最差桶提升最大。

Tilt SFT sampler from heuristic-prior to data-driven posterior — same
training budget, biggest lift on weakest buckets.

### 4.2 弱桶判定标准 / Weak-bucket criteria

按以下任一条件入选 / pick if any of:
- HM < 0.30（绝对低）
- HM < 整体 HM × 0.60（相对低）
- n ≥ 5（样本数够，结论稳）

### 4.3 数据扩充策略 / Augmentation strategy

> **前置约束 / Hard prerequisites** — 必须先满足 §0.5.3 的三条硬约束
> （NEI oversample + citation 格式样本稀疏 + DISPUTED 优先扩充）。下表
> 是在硬约束之上的桶级精调。
>
> The three hard constraints from §0.5.3 must hold first; the table
> below adds bucket-level fine-tuning on top of them.

按弱桶类型分别处理 / Different tactics per bucket type:

| 弱桶维度 / Bucket axis | 典型弱点 / Typical weakness | 扩充手段 / Augmentation tactic |
|---|---|---|
| **scenario** = `nei_topic_off` | base 把无关证据看成相关 / treats off-topic as relevant | 加大 `n_hard_neg` 配比（当前 1 → 3）；hard-neg 已经天然是 nei_topic_off |
| **scenario** = `disputed_conflict` | 看到矛盾仍 SUPPORTS / labels SUPPORTS despite conflict | 在 DPO 阶段（Phase 5）合成 SUPPORTS-vs-DISPUTED 对抗对（已在 `dpo_pairs.synthesise_disputed_contrast`） |
| **domain** = `general_other` (启发式覆盖率不足 / heuristic miss) | 训练分布外 / out-of-distribution at train time | 重新跑 §1.2 三维打标，用 sentence-transformer 聚类细化 `general_other` 子类（当前 519 / 1228 = 42% 兜底） |
| **difficulty** = `hard` | 多证据需聚合 / aggregation required | 在 `build_sft_record` 里给 hard 样本设 `k=8`（看更多 evidence） |

### 4.4 实施 / Implementation (v2 ✅ 2026-05-12，revised PM 同日)

`src/sft_dataset.py:build_dataset` 已加 `weak_buckets` 参数，签名：

```python
def build_dataset(
    tagged_rows, evidence_corpus, *,
    retrieval=None, k=20, pad_with_random=False, n_hard_neg=0,
    seed=42, apply_curriculum=True,
    weak_buckets: dict[tuple[str, str], int] | None = None,
) -> list[dict]:
```

匹配多个 (axis, bucket) 的 row 取**最大** factor（非乘积）以避免失控；
对真实 SFT 记录和 hard-neg 记录同时缩放；`rng` 跨重复样本共享 → 不同
副本的 random padding / hard-neg 噪声 evidence 不同（非简单克隆）。

`src/build_stage0.step_sft` 内置 Phase 4 配比（**v2 revision，2026-05-12 PM**，
根因 debug_log 复用经验 32）：

```python
# v2 first cut (broken): n_hard_neg=1, nei_underspec ×4 → 79.1% NEI in train
# → SFT collapsed to "always predict NEI" (Track 3 HM 0.140 < Track 2 0.201)
_TRAIN_WEAK_BUCKETS = {
    ("scenario", "nei_underspec"): 2,       # 4 → 2 (real NEI 减半)
    ("scenario", "disputed_conflict"): 3,   # 2 → 3 (DISPUTED 仍弱，加强)
    ("scenario", "refutes_clear"): 2,       # 不变
}
# train config 也改：n_hard_neg=0（去掉 hard-neg 同义重复 → 2083 条 synth NEI）
```

实测重建后 label 分布（vs gold 33% NEI / 31% S / 18% R / 17% D）：

| label | v2-broken | **v2-rebalanced** | gold | ratio |
|---|---|---|---|---|
| SUPPORTS | 10.4% | 27.6% | 31.4% | 0.88 |
| REFUTES | 6.2% | 16.5% | 18.2% | 0.91 |
| **NEI** | **79.1%** ⚠️ | **38.7%** | 33.1% | 1.17 |
| DISPUTED | 4.3% | 17.2% | 17.4% | 0.99 |

总记录 4166 → 1567 (×2.7 缩小)，训练时间预估 4h → ~1.5h。

**Sanity check 落在 `src/build_stage0.py:_print_label_dist`**：每个 split build
完打印 vs gold label distribution + ratio，**ratio > 2× 或 < 0.5× warn**，
指向 debug_log 复用经验 32。再也不会盲跑 4h 然后发现 class-collapse。

`tests/test_sft_dataset.py` 覆盖：(1) factor 同时缩放 real + hard-neg，
(2) 多 (axis, bucket) 匹配取 max 而非 product，(3) `weak_buckets={}`
或 None 为 no-op。

### 4.5 产出 / Output

- `outputs/sft_data/sft_train_v2.jsonl`（k=20 + weak_buckets 倾斜版 /
  k=20 with weak-bucket tilt）
- `outputs/sft_data/sft_{dev_holdout,diag_test}_v2.jsonl`（k=20，不做
  oversample，用于评估 / k=20, no oversample, used for eval）
- v1 文件保留 / v1 files preserved（k=5 baseline，ablation 用）

跑：`python -m src.build_stage0 --force` 重建（~5 s on AutoDL）。

---

## 5. Phase 5 — SFT/DPO 训练 + 评估 / Train + Eval

### 5.1 SFT 训练（v2 数据）/ SFT (v2 data)

```bash
# 修改 cell-2-sft-train 的 DATA_PATH 指向 sft_train_v2.jsonl
python -m scripts.run_sft   # 或直接在 notebook 取消注释 !{cmd}
```

**显存预算**已在 design.md §8.4 列出。AutoDL 4080 SUPER 上 ~25-35 min。

### 5.2 DPO 训练（dev_holdout 错样本 + DISPUTED 对抗对）/ DPO

```bash
python -m scripts.run_dpo  # 待实现 / TBD
```

### 5.3 评估同样的 prompt + 训练后的模型 / Re-eval

**LoRA adapter 加载坑（2026-05-12 PM 实测，debug_log 复用经验 31）**：

ms-swift 训练 Qwen3.5 SFT 时使用 `Qwen3_5VLForConditionalGeneration` (VL
wrapper)，adapter 的 `target_modules` regex 和 state_dict keys 都带
`model.language_model.` 前缀。`phase1_eval.py:load_model_and_tokenizer` 用
`AutoModelForCausalLM` 加载会得到 `Qwen3_5ForCausalLM`（纯 LM，结构
`model.layers.X`），peft 注入失败 / state_dict 不匹配 → LoRA 静默不生效，
Track 3 输出跟 Track 2 字节级相同。

**正确路径：用 `swift export --merge_lora true` 把 LoRA 烤进 base，然后
phase1_eval 用 `--sft-merged-dir` 加载合并后的模型（走标准 AutoModelForCausalLM
路径，无 adapter）**：

```bash
# 1. 合并 LoRA 到 base（~30s，产物 ~8 GB）
swift export --adapters outputs/sft-out/checkpoint-final \
    --merge_lora true --output_dir outputs/sft-out/merged

# 2. eval 用 --sft-merged-dir（不是 --sft-adapter）
#    Base model 由 cache-first 自动定位（models/Qwen3.5-4B/），无需 --model-dir。
python -m scripts.phase1_eval \
    --tracks 2,3 --prompts <locked> --dataset diag_test \
    --sft-merged-dir outputs/sft-out/merged

# 3. Track 4 (DPO) 待实现 — DPO 训练完后同样 merge_lora 再 eval
```

**警告信号 - adapter 静默不生效**：
- `phase1_eval` stdout 显示 `SFT adapter loaded (0.0X M LoRA params)` 或低于 1M
- Track 3 vs Track 2 数字字节级相同（F/Acc/HM 完全一样）
- 看到这些立刻停掉，走 `--sft-merged-dir` 路径

**`--sft-adapter` flag 保留**给将来的非 VL 模型用，但 Qwen3.5 不行。

### 5.4 验证 / Validation

对比 Phase 3 的 per-bucket 表 vs Phase 5 的 per-bucket 表：
- **每个被瞄准的弱桶 HM 提升 ≥ +0.10** → 数据扩充策略生效
- **未瞄准的桶 HM 不掉** → 没有"按下葫芦浮起瓢"
- **总 HM 提升 ≥ +0.05** → 可以进 Phase 6

如果 (1) 不达标，回 Phase 4 重新设计 weak_buckets 配比。

Compare Phase 3 vs Phase 5 per-bucket tables; targeted buckets should
each gain ≥ +0.10 HM, untargeted buckets shouldn't regress, total HM
should gain ≥ +0.05. If targeted buckets don't improve, return to
Phase 4 and rethink the augmentation tactic.

---

## 6. Phase 6 — 最终汇报 / Final Report

### 6.1 在 official dev 上跑一次最终 4-track / Run 4-track on official dev once

```bash
# 这是消耗"看 dev"额度的最终一次 / consumes the "look at dev" budget
python -m scripts.phase1_eval \
    --tracks 1,2 --prompts <locked> --dataset official_dev ...
# 加上 SFT/DPO track（Phase 5 拿到 checkpoint 后）
```

### 6.2 跑 test 集 / Run test set

`notebook.ipynb` cell 3.4 (`cell-3-test-code`)：
- 读 `data/test-claims-unlabelled.json`
- 跑同一 pipeline → `outputs/test-claims-predictions.json`

### 6.3 写报告 / Write report

`design.md` §11.3 列出的 6-9 张图表；本 plan 的 §10 决策日志直接搬进报告 Discussion。

---

## 7. 持久化策略 / Persistence Strategy

### 7.1 Cache-first 模式 / Cache-first pattern

每个 cell / 脚本的统一模板 / Uniform template:

```python
if (CACHE_PATH / "<artifact>").exists():
    obj = load_from_cache(CACHE_PATH)
    print("[cache] loaded")
else:
    obj = expensive_build()
    save_to_cache(obj, CACHE_PATH)
    print(f"[wrote] {CACHE_PATH}")
```

### 7.2 Artifact 清单 / Artifact inventory

| Artifact | 路径 / Path | 大小 / Size | 来源 / Source | Cache 检测点 / Cache detection |
|---|---|---|---|---|
| Evidence corpus | `data/evidence.json` | 174 MB | 课程提供 / provided | (no build) |
| 三维打标 / 3-axis tags | `outputs/splits/claims_tagged.jsonl` | < 1 MB | `src.stage0_tag.run()` | cell 1.2 检测 |
| Hash 切分 / splits | `outputs/splits/{train_split,dev_holdout,diag_test,official_dev}.jsonl` | < 1 MB | `src.splits.run()` | cell 1.3 |
| **SFT 训练数据 / SFT data** | `outputs/sft_data/sft_{train,dev_holdout,diag_test}_v1.jsonl` | ~5 MB | `src.sft_dataset.build_dataset()` | **cell 1.4 已升级 / upgraded ✓** |
| BM25 索引 / index | `outputs/bm25_index/` | ~200 MB | `BM25Retriever.build()` | cell 2.1 ✓ |
| Dense 索引 / index | `outputs/dense_index/{faiss.index,chunks/}` | ~5 GB | `DenseRetriever.build()` | cell 2.2 ✓（带 faiss.index 二次校验） |
| **模型权重（统一）/ all model weights** | `models/{Qwen3.5-4B,bge-m3,bge-reranker-base,bge-small-en-v1.5}/` | ~11 GB total | `scripts.download_models` | `paths.resolve_model_path()` ✓ — DenseRetriever / CrossEncoderReranker / phase1_eval / smoke_test 全部自动检测 |
| **safetensors 转换（bge-* 系列）/ bin→safetensors conversion** | `models/<bge-*>/model.safetensors` + `pytorch_model.bin.bak` | ~2 GB / model | `scripts.convert_bin_to_safetensors` | 跳过已有 `*.safetensors` 的目录；见 §8 风险表 + debug_log 问题 16 |
| SFT checkpoint | `outputs/sft-out/checkpoint-*/` | ~100-500 MB | ms-swift sft | cell 3.5b: `if SFT_CKPT.exists()` ✓ |
| DPO checkpoint | `outputs/dpo-out/checkpoint-*/` | ~100-500 MB | ms-swift rlhf | cell 3.5b: `if DPO_CKPT.exists()` ✓ |
| 推理预测 / predictions | `outputs/predictions/track*.json` | < 1 MB | `evaluate_track()` | (覆盖式 / overwritten each run) |
| 评估报告 / eval reports | `outputs/eval_phase1/*.{json,md}` | < 1 MB | `phase1_eval` | (覆盖式) |

### 7.3 提交边界 / Submission boundary（per design.md §2.8）

**禁止提交 / Cannot ship**：
- `data/evidence.json`（174 MB）
- `outputs/dense_index/`（5 GB）
- `outputs/model_cache/`（8 GB）
- `outputs/sft-out/`、`outputs/dpo-out/`（checkpoints）

**可以提交 / Can ship**：
- `notebooks/notebook.ipynb`（含已执行结果 / with cell outputs）
- `src/`、`tests/`、`scripts/`
- `design.md`、`debug_log.md`、`optimization_plan.md`、`requirements.txt`、`README.md`
- `outputs/sft_data/*.jsonl`（小，演示数据格式 / small, shows format）
- `outputs/eval_phase*/*.md`（评估报告 markdown / eval markdown reports）
- `outputs/dry_run_report.md`、`outputs/PROGRESS.md`

`.gitignore` 已经把大件全部排除，本地开发 / AutoDL 上保留完整 cache，提交 zip 时仅打包小件。

`.gitignore` already excludes large artifacts; full cache lives on dev /
AutoDL boxes; submission zip carries only small artifacts.

### 7.4 跨机器 cache 复用 / Cross-machine cache reuse

```bash
# 本地建好 BM25 → 上传到 AutoDL（10 分钟 vs 重建 3 分钟，没必要 / not worth it）
# 本地建好 Dense → 上传到 AutoDL（5 GB 上传慢，重建 15-20 分钟 / rebuild faster）

# 真正值得跨机复用的是 model weights：
# AutoDL 下好 → tar + scp 到本地（避免本地再下 8 GB）
# 或本地用 modelscope 下好 → scp 到 AutoDL outputs/model_cache/
```

实测：dense 索引重建比 SCP 5GB 快，所以**每台机器各自建**就行。模型权重值得跨机复用。

---

## 8. 风险与回退 / Risks and Fallbacks

| 风险 / Risk | 概率 / P | 缓解 / Mitigation |
|---|---|---|
| Phase 4 数据扩充后总 HM 反而下降 / total HM drops after Phase 4 | 中 / med | 保留 v1 数据；按 ablation 双数据集训两个 LoRA 对比 |
| Few-shot prompt 在某些样本上 OOM / few-shot OOMs on some claims | 低 / low | `--max_length` 从 1024 → 1280 给 prompt 留余量 |
| Phase 5 训练 8h 超时 / training timeout | 中 / med | `--save_steps 200` + `resume_from_checkpoint`（已配置 / configured） |
| Phase 6 official_dev 数字比 diag_test 差很多 / official_dev underperforms | 中 / med | 说明 diag_test 不能完全代理 official_dev 分布；只在 final report 诚实说明 |
| 部分弱桶 n < 5（统计噪声）/ small-n buckets are noisy | 高 / high | 弱桶判定加 `n >= 5` 门槛（§4.2 已规定） |
| ModelScope 镜像缺 safetensors → transformers CVE-2025-32434 拒绝 .bin / mirror gap blocks .bin load on torch<2.6 | 已发生 / observed | `python -m scripts.convert_bin_to_safetensors` 本地离线转换；**不**升级 torch（会破坏 flash-attn 2.x 编译）；见 debug_log 问题 16 + design.md D-016 |
| AutoDL 墙内 HF 直连不通（`[Errno 99]`） / HF unreachable from AutoDL | 已发生 / observed | `export HF_ENDPOINT=https://hf-mirror.com` 走镜像；优先 ModelScope，HF 仅作 fallback |
| Colab Free 12.7 GB RAM 跑 inference 余量薄 (~1.2 GB) / Colab Free RAM headroom thin for inference | 已审计 / audited | **当前 AutoDL 主线无影响**（31.5 GB 充裕）；若回 Colab Free，启 `faiss.IO_FLAG_MMAP` 改 mmap 加载（−3 GB RAM）和/或换 `bge-small-en-v1.5`（−3 GB 额外，recall 掉 5-10pp）。详见 §8.1 推理 RAM 优化清单。 |
| Colab 90 min 空闲断连 + 4h SFT 训练 / Colab idle-disconnect 90 min vs 4h SFT | 已审计 | 主线在 AutoDL 训，Colab 备份场景需保活脚本 / Colab Pro |

### 8.1 推理 RAM 优化清单（deferred）/ Inference RAM optimization backlog

**触发条件 / When**：仅当主线从 AutoDL 切回 Colab Free（12.7 GB RAM / 14.5 GB T4）或其他低 RAM 环境时启用。AutoDL 31.5 GB 余量大不需要。

**推理高峰 RAM 拆解**（121 claims × Track 2 RAG，实测预估）：

| 组件 | RAM | 占比 | 优化空间 |
|---|---|---|---|
| Python + torch + transformers + ms-swift | 3.0 GB | 26% | 难压缩 |
| `evidence` dict (1.2M passages) | 1.5 GB | 13% | mmap-based lazy lookup |
| **FAISS `IndexFlatIP`（全 load）** | **5.0 GB** | **43%** | **`IO_FLAG_MMAP` 直接 −3 GB** |
| BM25 sparse arrays | 0.5 GB | 4% | 难压缩 |
| Pipeline + reranker buffers | 0.5 GB | 4% | — |
| Python overhead | 1.0 GB | 9% | — |
| **峰值 / Peak** | **~11.5 GB** | — | — |

**优化选项**（按 ROI 排序）：

1. **`faiss.IO_FLAG_MMAP`**（首选，−3 GB，0 精度损失）：
   修改 `src/retrieval/dense.py:DenseRetriever.load` 加 `mmap: bool = True` 参数，
   传 `faiss.IO_FLAG_MMAP` 给 `faiss.read_index`。AutoDL 上 mmap 也无副作用
   （Linux 文件缓存自动管理）。
2. **替换 dense 模型 `bge-m3` (1024-d) → `bge-small-en-v1.5` (384-d)**（−3 GB 额外，−5 to −10pp recall）：
   notebook cell-2-2-code 把 `DEFAULT_MODEL` 换成 `LIGHT_MODEL`。索引重 build (~5 min)。
   降级路径，正式数字保留 bge-m3。
3. **Evidence dict lazy lookup**（−1.5 GB，+5ms/claim）：
   实现 `EvidenceStore` 用 byte-offset 索引，按需 `pread()` 读 evidence.json。
   代码量中等，节省有限，优先级低。
4. **FAISS `IndexIVFFlat` 量化版**（−3.5 GB，−2 to −3pp recall）：
   重 build 索引为 IVF + PQ 量化。精度损失可接受但需要重训 IVF 中心，目前不值得做。

**测试方式**（未来真要做时）：

```bash
# 在 Colab Free 实例上
python -c "
import psutil, faiss
mem_before = psutil.virtual_memory().used / 1e9
idx = faiss.read_index('outputs/dense_index/faiss.index', faiss.IO_FLAG_MMAP)
mem_after = psutil.virtual_memory().used / 1e9
print(f'mmap load: {(mem_after - mem_before):.2f} GB resident')
# 期望 < 2 GB vs 默认 ~5 GB
"
```

**审计记录**：2026-05-12 PM Session 13 末 user 询问 Colab 兼容性后捕获。当前 SFT 训练 + 评估全在 AutoDL（不阻塞主线）。等 Phase 6 final report 阶段如果要重现性测试，再评估是否需要 mmap + bge-small 降级在 Colab 跑一遍 baseline。

---

## 9. 当前状态 / Current Status (2026-05-11)

### 已完成 / Done
- [x] D-015 方法论决策固化到 design.md
- [x] `src/prompt.py` 加 v1-v4 变体
- [x] `src/inference.py` 加 `prompt_version` 参数
- [x] `scripts/build_indexes.py`（独立索引构建）
- [x] `scripts/phase1_eval.py`（Phase 1 评估 harness）
- [x] cell-1-sft-code 升级为 cache-first（同时建三份数据）
- [x] `requirements.txt` 加 AutoDL Quick Start
- [x] AutoDL 4080 SUPER 实例 + 模型 + smoke test 通过
- [x] 本地 BM25 索引（`outputs/bm25_index/`, ~200 MB）
- [x] 本地 SFT 数据迁移到 messages 格式（重跑 `src.build_stage0`）
- [x] §0.5 base 模型能力探针写入 plan（4a-4c 实测 → SFT 数据三条硬约束）
- [x] `scripts/convert_bin_to_safetensors.py`（应对 ModelScope 缺 safetensors + transformers CVE-2025-32434）
- [x] AutoDL 上 dense 索引建好（9.2 GB）+ Phase 1 第一次评估跑通（Track 1+2 × v1 on diag_test）
- [x] `scripts/diagnose_phase1.py`（应对 Phase 1 数字异常 — 见 debug_log 问题 17）
- [x] **Phase 1 诊断完成**：base 模型 NEI acc=0.025 / DISPUTED acc=0.000（量化证实 §0.5.2 4a）；
      非 parser fallback。Track 2 v1 per-label acc: SUPPORTS 0.526 / REFUTES 0.500 /
      NEI 0.350 / DISPUTED 0.286。
- [x] **Phase 2 prompt sweep 完成**：v1 (HM=0.1830) 锁定为生产 prompt。v2 NEI 指令拉起 NEI acc
      +20pp 但 over-correct 偷 REFUTES (−27pp)；v3 DISPUTED 指令把 REFUTES 干到 0；v4 few-shot
      未能修 v3 的问题。验证 §0.5.3 硬约束 1+3（NEI/DISPUTED 是能力问题，prompt 教不会）。
- [x] **检索天花板发现（关键）**：Track 2 v1-v4 evidence recall 全部 ≈ 0.11，与 prompt 无关。
      F-score 当前架构硬上限 ≈ 0.12，HM ≈ 0.21。在 SFT 之前必须先做 Phase 3.5。
- [x] **Phase 3.5 检索审计完成**：`final_k` mode 显示 recall@5=0.119 / recall@20=0.333 /
      recall@100=0.579，gold ev 多在 rank 6-20。端到端验证 k=5/10/20 在 Track 2 v1 上跑：
      HM 0.183 / 0.196 / 0.203 单调上升。**锁定 `final_k=20`**（`RetrievalConfig` default 已改）。
      副作用：NEI acc 0.350 → 0.025（k=20 让模型几乎不输出 NEI，正好是 Phase 4 要修的）。
- [x] **Phase 4 `weak_buckets` 实现**：`src/sft_dataset.py:build_dataset` 加参数；
      `src/build_stage0.py` 内置 `{nei_underspec:4, disputed_conflict:2, refutes_clear:2}` 配比；
      `tests/test_sft_dataset.py` 覆盖。
- [x] **v2 SFT 数据本地构建 + 分布验证**（2026-05-12）：`python -m src.build_stage0 --force`
      产生 sft_train_v2.jsonl 4166 records（×2.11 over v1）。weak_buckets 完全按预期触发：
      nei_underspec 303→1212 (×4.00), disputed_conflict 90→180 (×2.00),
      refutes_clear 98→196 (×2.00). 全局 NEI 占比 65.4% → 79.1%
      （含 hard-neg），hard difficulty 占比 14.3% → 24.1%。
- [x] **AutoDL SFT 训练 cell 修复**（notebook_autodl.ipynb 9 cells patched）：
      cache-first model 加载（sec2-5-download/sec3-5a），ms-swift v3.6+ CLI
      命名（`--tuner_type` / `--quant_bits` / thinking trio），DATA_PATH→v2，
      MAX_LEN 1024→1536 容纳 k=20 prompt。
      *Fixes ModelScope re-download + Issue 9 CLI rename + Phase 4 v2 path*.
- [x] **清理仓库内残留 `Qwen3___5-4B` 路径**（5 处）→ 统一用 `Qwen3.5-4B`。
      详见 `debug_log.md` 复用经验 26。

- [x] **AutoDL SFT v2 训练启动**（2026-05-12 ~04:05 UTC）：env stack 三连碰
      （FSDP2 / transformers 5.2 / peft HybridCache，详见 debug_log 复用经验
      27-28）全部解决；nbstripout 装好挡 jupyter autosave（复用经验 29）；
      `scripts/patch_swift_fsdp2.py` 落地。训练开跑：step 1 loss 0.1146 →
      step 25 loss 0.019，grad_norm 健康，VRAM 11.4 GB / 31.5 GB，预估 ~4h。
- [x] **phase1_eval Track 3 支持**：`--sft-adapter PATH` flag 实现 + 校验。
      SFT 训完直接 `python -m scripts.phase1_eval --tracks 2,3 --sft-adapter ...`
      端到端拿到 Track 2 vs 3 对比。

### 已完成（2026-05-13 早段）/ Done
- [x] **Phase 3.5b 检索深度审计完成**：4 个 mode (retriever / fusion_w /
      synonym_expand / llm_rewrite) 全部跑完。**关键发现**：
      - bge-reranker-base 在 climate 域是负贡献（×1.68 worse on recall@5）
        → 锁 `use_rerank=False`，免费 +0.030 HM
      - HyDE + sub-claims recall@20 = 0.339 < baseline 0.357，但
        recall@50/100 +0.044/+0.058
      - **recall@20 锁死在 0.357**，retrieval-side 路全堵
- [x] **战术转向 data-format**：D-019 副推论 `pad_with_random=False`
      改 `src/build_stage0.py`；本地 rebuild 验证 n_shown 从全 20 →
      1-5 ev 分布；label 分布不变。详见 debug_log 复用经验 36。
- [x] **Track 2 v1 baseline 升级到 HM 0.213**（no-rerank 实测）

### 已完成 + 留档作为 ablation / Done as ablation evidence
- [x] **SFT v2-cut-1 + v3-rebalanced 两次失败**：HM ≈ 0.140 < Track 2
      baseline 0.213。报告 Results 章节用这两次 + retrieval audit + pad
      pivot 作 narrative 主线（README §307 "insight into why method
      fails"）。debug_log 复用经验 32 / 34 / 36 三连。

### 进行中 / In progress
- [ ] **SFT v6 重训（pad_with_random=False）**：AutoDL 上 `git pull` +
      `build_stage0 --force` + `run_sft.py`（~1.5h）→ `swift export
      --merge_lora` → `phase1_eval --tracks 2,3 --sft-merged-dir
      merged_v6` → diagnose。这是 SFT 救场的最后一次机会。
      *SFT v6 retrain with pad-alignment fix — final shot at making SFT
      beat Track 2 baseline.*

### 待做 / Todo（取决于 v6 结果）
- [ ] **v6 HM > 0.213** → DPO 训练（dev_holdout 错样本 + DISPUTED 对抗对）
      → Track 4 SC 评估 → 4-track 完整对比 → Phase 6 official_dev 终评
- [ ] **v6 HM ≤ 0.213** → **接受 Track 2 v1 (HM 0.213) 作 final 提交**，
      SFT 全留 ablation；直接进 Phase 6（test 集预测用 Track 2 配置）
- [ ] **Phase 6 二选一**：
      - SFT 路径成功：sft-merged-v6 / 或 + DPO 跑 official_dev + test
      - SFT 失败：Track 2 v1 配置（fused no-rerank, final_k=20, no SFT）跑
- [ ] **报告 LATEX 撰写**（design.md §11.3 列了 6-9 张图表 + 3 failure
      story narrative 已就绪）

---

## 10. 决策日志 / Decision Log

每次 Phase 完成后在此追加一条 / Append after each phase completion.

| Date | Phase | Decision | Source data |
|---|---|---|---|
| 2026-05-11 | (pre-Phase 1) | smoke test 显示 base 模型能输出 `LABEL ##[..]##` 干净格式，5/5 SC 同票 (SUPPORTS) 但 sample 5 同时引 ev-pro 和 ev-con → base 缺"识别证据矛盾"能力 | `materials/qwen3.5_key_points.docx` + AutoDL smoke test logs |
| 2026-05-11 | Phase 1 (raw) | Track 1 v1 F=0.0000 Acc=0.3223 HM=0.0000；Track 2 v1 F=0.1169 Acc=0.4215 HM=0.1830。Track 1 Acc 几乎=NEI 占比 0.3306 → 触发 NEI-default 诊断（debug_log 问题 17 + diagnose_phase1.py） | `outputs/eval_phase1/summary_diag_test.md` |
| 2026-05-12 | Phase 1 diagnosed | 非 parser fallback。Track 1 NEI acc=0.025 / DISPUTED acc=0.000（base 模型缺这两类能力，量化重复 §0.5.2 4a）。Track 2 v1 per-label: S 0.526 / R 0.500 / NEI 0.350 / D 0.286 — RAG 补了 NEI/DISPUTED 但磨钝 S/R。 | `outputs/eval_phase1/diagnose_diag_test.md` |
| 2026-05-12 | Phase 2 done | 锁定 prompt **v1 baseline**（HM=0.1830）。v2 NEI 指令 over-correct 偷 REFUTES (−27pp)；v3 DISPUTED 把 REFUTES 干到 0；v4 few-shot 未修。验证 §0.5.3 硬约束 1+3。 | `outputs/eval_phase1/summary_diag_test.md` |
| 2026-05-12 | 检索天花板发现 / Retrieval ceiling found | Track 2 v1-v4 evidence recall 全部 ≈ 0.11（与 prompt 无关）→ F-score 当前架构硬上限 ≈ 0.12, HM ≈ 0.21。SFT 之前必须先做 Phase 3.5。 | `outputs/eval_phase1/diagnose_diag_test.md` |
| 2026-05-12 | Phase 3.5 audit (final_k mode) | macro recall@k: 5→0.119 / 10→0.210 / 20→0.333 / 50→0.485 / 100→0.579. Gold ev 集中在 rank 6-20。 | `outputs/eval_phase1/retrieval_ceiling_diag_test.md` |
| 2026-05-12 | Phase 3.5 done — final_k=20 锁定 | k=5/10/20 端到端实测：HM 0.183 / 0.196 / 0.203（单调）。k=20 NEI acc 0.025（崩塌），SUPPORTS 0.737 / REFUTES 0.682（暴涨）。`RetrievalConfig.final_k` 默认改为 20；`phase1_eval --final-k` 默认 20，非默认值加 `_kN` 后缀防覆盖。 | `outputs/eval_phase1/diagnose_diag_test_k20.md` |
| 2026-05-12 | Phase 4 implementation | `build_dataset` 加 `weak_buckets` 参数；`build_stage0` 内置 `{nei_underspec:4, disputed_conflict:2, refutes_clear:2}`；tests 全绿。SFT 数据待重建。 | `src/sft_dataset.py`, `src/build_stage0.py`, `tests/test_sft_dataset.py` |
| 2026-05-12 | SFT data v2 built (本地) | `sft_train_v2.jsonl` 4166 records（×2.11 over v1）。weak_buckets 100% 按预期触发：nei_underspec 303→1212, disputed_conflict 90→180, refutes_clear 98→196。NEI label 占比 65.4%→79.1%，hard difficulty 占比 14.3%→24.1%。n_shown ≈ k=20。 | `outputs/sft_data/sft_train_v2.jsonl` |
| 2026-05-12 | AutoDL SFT 启动准备 | notebook_autodl.ipynb 9 cell 修补：cache-first 模型加载（解决 8 GB 重复下载）+ ms-swift v3.6+ CLI 命名（`--tuner_type` / `--quant_bits` / thinking trio + liger + group_by_length + save_total_limit）+ DATA_PATH→v2 + MAX_LEN 1024→1536（容纳 k=20 prompt）。清掉 `Qwen3___5-4B` 残留 5 处。AutoDL pull 用 `source /etc/network_turbo` 走官方加速通道。 | `notebooks/notebook_autodl.ipynb`, `debug_log.md 复用经验 24-26` |
| 2026-05-12 | Env stack locked | torch 2.5.1+cu124 + ms-swift 4.2.0 + transformers 5.2.0 + peft (latest, post-HybridCache-removal) + qwen_vl_utils 0.0.14+ + liger-kernel + bitsandbytes 0.49+。FSDP2 patch (`scripts/patch_swift_fsdp2.py`) 把 ms-swift 4.2 对 torch 2.6 API 的硬依赖打成可选。`requirements.txt` 拆 torch 2.5（默认）+ `requirements-torch26.txt`（未来）。`nbstripout` 加入 dev deps 防 jupyter autosave 与 git pull 死锁（debug_log 复用经验 27/28/29）。 | `requirements.txt`, `requirements-torch26.txt`, `scripts/patch_swift_fsdp2.py`, `debug_log.md` 复用经验 27-29 |
| 2026-05-12 | SFT v2 kickoff | 训练参数：LoRA r=16 / α=32，QLoRA 4-bit + bf16，BS=2 ×GA=8 (eff 16)，lr 2e-4 + warmup 0.03，max_len 1536，liger_kernel + group_by_length + save_total_limit=3，thinking 三件套 (--enable_thinking false / --add_non_thinking_prefix true / --loss_scale ignore_empty_think)，3 epochs / 783 steps。Step 1 loss 0.1146 / step 25 loss 0.019 / VRAM 11.4 GB / ~19 s/step (4080 SUPER) / 预估 ~4h 总。Loss 低是 prompt-masking 正常表现（debug_log 复用经验 30），真信号看 Track 3 vs 2 metric delta。 | SFT stdout log，checkpoint 路径 `nlp_a3_cache/sft-out/v4-20260512-040505/` |
| 2026-05-12 | phase1_eval Track 3 wired | 新增 `--sft-adapter PATH` flag → `PeftModel.from_pretrained(base, path)` 原地包装。`--tracks 3` 强制 require adapter。Track 3 = base + SFT + RAG (k=20)，复用 ZeroShotInferer。诊断脚本 / write_summary 自动支持 track3_* 文件 prefix。 | `scripts/phase1_eval.py` |
| 2026-05-12 PM | SFT v2 first cut **失败** (class-collapse) | merged-LoRA Track 3 数字 HM 0.140 < Track 2 baseline 0.201（−6pp）。诊断揭示：predicted NEI 占比 **92.6%**（gold 33%），NEI acc 0.975 但 non-NEI acc **0.062** —— 训练数据 79.1% NEI 让模型塌缩到 majority class。根因：`n_hard_neg=1` × `nei_underspec ×4` 双重 NEI 放大。详见 debug_log 复用经验 32。 | `outputs/eval_phase1/diagnose_diag_test.md` (track3_v1 行) |
| 2026-05-12 PM | v2 revision (rebalanced) | `n_hard_neg: 1→0` + `nei_underspec: 4→2` + `disputed: 2→3`。重建后 NEI label 占比 79.1% → 38.7%（接近 gold 33%），总记录 4166→1567 (×2.7 缩小)。所有 ratio 在 0.63-1.89× gold 区间。`build_stage0` 加 distribution sanity check 每个 split 打印 + warn >2×/<0.5× 偏差。 | `src/build_stage0.py` + label-dist 输出 |
| 2026-05-12 PM | SFT v3-rebalanced **也失败** | 重平衡 (NEI 79%→38.7%) 不解决问题：Track 3 HM 0.140 与 v2-cut-1 完全相同；predicted NEI 92.6% → **94.2%** 略恶化；non-NEI acc 0.062 → **0.049**。**class-imbalance 不是根因**。详见 debug_log 复用经验 34。 | `v5-20260512-180014/checkpoint-294` merged_v3 |
| 2026-05-12 PM | 战略转向 retrieval-first | v3-rebalanced 失败定位根因为 **train-inference distribution mismatch**：`pad_with_random=True` 让训练时非 NEI 样本看起来跟低 recall RAG 输出一样（少数 gold + 多数 random ev），模型学到 "evidence 模糊 → NEI" 捷径。把 SFT 暂停，**先把 retrieval recall@20 从 0.333 拉到 ≥ 0.50** 才有意义训 SFT。design.md D-019。 | debug_log 复用经验 34 |
| 2026-05-12 PM | Retrieval audit done (3 modes) | **意外发现：reranker 是负贡献**。`fused (no rerank)` recall@5 = 0.200 vs `full (fused + rerank)` 0.119，×1.68 improvement。fusion_w 扫 0.1-0.9 最佳 w_bm25=0.7 (recall@5 0.154)；synonym expand WordNet 同义词 −0.004 没用。**Decision**: 锁 `use_rerank=False` default（复用经验 35），但 recall@20 仍只 0.360 < 0.50 target → 还需要 LLM rewrite。 | `outputs/eval_phase1/retrieval_ceiling_diag_test.md` |
| 2026-05-12 PM | 锁 use_rerank=False | `RetrievalConfig.use_rerank` default True→False；`phase1_eval --rerank` opt-in for ablation。期望 Track 2 baseline HM 0.201 → ~0.22-0.24 免费拿 +0.02。 | `src/retrieval/pipeline.py`, `scripts/phase1_eval.py` |
| 2026-05-12 PM | Track 2 v1 (no-rerank) 实测 | HM 0.201 → **0.213** (+0.030)，F 0.135 → 0.151，免费胜利兑现。non-NEI acc 0.531 (中间值)，predicted NEI 1.7%。 | `outputs/eval_phase1/diagnose_diag_test.md` (Track 2 v1 行) |
| 2026-05-13 | HyDE/sub-claim audit done | `llm_rewrite` mode 4 configs × 121 claims: baseline recall@20=0.357, HyDE only=0.357, sub-claims only=0.331, HyDE+sub=0.339。**top-20 没救**。但 recall@50 baseline 0.467 → HyDE+sub 0.511 (+0.044)，recall@100 +0.058。HyDE 价值在 long-tail，不在 SFT context。 | `outputs/eval_phase1/retrieval_ceiling_diag_test.md` |
| 2026-05-13 | 战术转向 pad_with_random=False | retrieval-side 路全堵了 (recall@20 = 0.357 上限)。换战术：保留 retrieval baseline，改 `src/build_stage0.py` train kwargs `pad_with_random=True → False`。训练样本变成 1-5 真实 gold ev (无 random noise)，跟推理 RAG 输出分布差距缩小。n_shown 分布 100% 20 → 14%/18%/14%/9%/45% (1-5 ev)。详见 debug_log 复用经验 36 + design D-019 副推论。 | `src/build_stage0.py` |
| TBD | SFT v6 (no padding) Track 3 | 重训 + 重 Track 3 评估。HM > 0.213 (Track 2 baseline) → SFT 真正生效；不到 → 接受 baseline 作 final 提交，SFT 全留 ablation。 | (待跑) |
| TBD | SFT v4 (post-retrieval-fix) | retrieval 锁好后重 build SFT 数据（自带新检索的 retrieved evidence）+ 重训。Δ HM target ≥ +0.05 vs Track 2 baseline 0.201 | (待跑) |
| TBD | Phase 2 done | 锁定 prompt: v?  Track 2 HM 提升 +? | `outputs/eval_phase1/summary_diag_test.md` |
| TBD | Phase 4 done | weak_buckets 配比: {...} → sft_train_v2.jsonl | `outputs/sft_data/sft_train_v2_meta.json` |
| TBD | Phase 5 done | SFT/DPO HM = ?, 最弱桶提升: ... | Phase 5 eval reports |
| TBD | Phase 6 done | Official dev: F=?, Acc=?, HM=?  Test predictions: outputs/test-claims-predictions.json | eval.py output |

---

**关联文件 / Related files**:
- `design.md` — 系统架构 + 决策记录 / system design + decision records
- `debug_log.md` — 问题排查日志 / problem-solving log
- `requirements.txt` — 依赖 + AutoDL Quick Start / deps + AutoDL onboarding
- `scripts/{build_indexes,phase1_eval,test_qwen35_inference}.py` — 阶段执行脚本 / phase execution scripts
