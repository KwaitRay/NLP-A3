# Evidence Chunking 优化计划 / Evidence Chunking Plan

> **目的**：把 evidence 长文段按 chunk 切分后再索引，让超过 dense encoder
> max_seq_length=256 的段落不再被静默截断。**主目标 recall@20 从 0.360
> 提到 ≥ 0.42**，次要目标 Track 2 HM 从 0.213 → ≥ 0.235。
>
> **Goal**: chunk long evidence passages before indexing so bge-m3
> (max_seq_length=256) stops silently truncating. Primary target:
> recall@20 0.360 → ≥ 0.42. Secondary: Track 2 v1 HM 0.213 → ≥ 0.235.
>
> Status: design document. Implementation guards in §11 checklist.
>
> 最后更新 / Last update: 2026-05-16

---

## 0. TL;DR

- **Profile-first**：先跑 `scripts/profile_evidence.py` 看 verdict。
  - verdict ∈ {"Strongly recommended", "Worth trying"} → 执行本 plan
  - verdict ∈ {"Marginal", "Unlikely to help"} → 跳过 chunking 直接进 Phase 6
- **3 个 chunking 策略**：fixed-token (256/overlap 64) / sentence-group / paragraph-aware
- **Chunk → passage 映射**：retrieval 返回 chunk_ids，aggregate 到 passage_id 用 **MAX score**
- **Index 重建成本**：BM25 ~10 min；Dense ~3-4h on 4080 SUPER（3× passages）
- **预算 ~6h 单线**（profile + impl + 3 strategy bake-off + lock）
- **核心约束**：gold 是 passage-level，submission 是 passage_id，**chunking 纯内部检索优化**

---

## 1. 动机 / Motivation

### 1.1 为什么 chunking 可能有效

bge-m3 默认 `max_seq_length=256`。当 evidence 段落超过 256 tokens 时：

- **静默截断**：encoder 把 token 257 后丢掉，向量只代表前 256 tokens
- **后半内容不可检索**：如果 gold 信息在 token 300 处，本段绝对不会被检索到（除非前 256 tokens 也跟 claim 高度相关）
- **bge-m3 BM25 不受影响**（BM25 是 sparse 词袋），但 dense 部分受影响 → fusion 后整体仍弱

`outputs/eval_phase1/retrieval_ceiling_diag_test.md` 实测：

| Mode | recall@5 | recall@20 |
|---|---|---|
| BM25 only | 0.136 | 0.263 |
| dense only | 0.170 | 0.319 |
| **fused (no rerank)** | **0.200** | **0.360** |

dense 比 BM25 强（0.319 vs 0.263 @20），所以 dense 的提升空间大。**如果 dense 在长段落上漏召回，chunking 直接救 dense**。

### 1.2 失败的兄弟方向 vs chunking 的不同

| 方向 | 问题 | chunking 不同点 |
|---|---|---|
| Reranker FT (Plan §15) | retrieval-relevance vs fact-checking task mismatch | chunking 不需要训练，**纯索引重建**，不依赖任务对齐 |
| HyDE/sub-claim (Plan §10 row) | only helps recall@50/100, not @20 | chunking 改变**段落本身**的可检索性，不依赖 query 改写 |
| SFT v2/v3 (D-019) | train/inference distribution mismatch | chunking 跟 train 无关 |

**关键判断**：reranker / SFT / query rewrite 全部失败的根因都是 **task signal 不足**；
chunking 修的是 **encoder truncation**，是个完全不同的物理瓶颈。这是为什么值得最后一搏。

---

## 2. Profile-first Gate / 先 profile 再决定做不做

### 2.1 先跑 / Run first

```bash
source /etc/network_turbo
cd ~/autodl-tmp/NLP-A3
python -m scripts.profile_evidence 2>&1 | tee outputs/evidence_profile.log
head -60 outputs/evidence_profile.md   # 看 Verdict 段
```

预计 ~5-10 min on 4080 SUPER（1.2M passages × bge-m3 tokenizer count）。

### 2.2 Verdict → 行动 / Verdict → action

`scripts/profile_evidence.py:171-194` 的 verdict 逻辑：

| 条件 | Verdict | 行动 |
|---|---|---|
| `pct_over_256 < 1%` AND `median < 100 tokens` | Chunking unlikely to help | **不做 chunking**，直接 Phase 6 |
| `pct_over_256 < 5%` | Marginal | 评估：如果 deadline 还宽松（4+ days）执行；紧（< 3 days）skip |
| `5% ≤ pct_over_256 < 20%` | Worth trying | **执行本 plan** |
| `pct_over_256 ≥ 20%` | Strongly recommended | **执行本 plan** + 优先级最高 |

**期望 verdict**：气候 fact-checking corpus 经常混合维基片段 + 新闻段 + 论文摘要，长尾不短。**主观先验：verdict 落在 "Worth trying" 概率最高**。但必须先看实测数字再 commit。

If profile says "Unlikely" / "Marginal" with tight deadline → 直接进 Phase 6 ablation 路径。

---

## 3. Chunking 策略 / Strategies

### 3.1 三种候选 / Three candidates

| 策略 | 实现 | 适用场景 | 风险 |
|---|---|---|---|
| **A. Fixed-token + overlap** | tokenize 全段 → 滑窗切 (size=256, stride=192, overlap=64) | 段落无清晰内部结构；纯长尾切割 | overlap 让索引膨胀 1.3-1.5× |
| **B. Sentence grouping** | spaCy/regex 切句 → 贪心聚成 ≤ 256 tokens 的 chunk | 段落是良构句子序列（科普文/论文） | 句界不准时（缩写、列表）切糟 |
| **C. Paragraph-aware** | 先按 `\n\n` 切段，每段内再用 A | newline counts > 1 (p90 看 evidence_profile) | 单段超过 256 还得回退 A |

**默认推荐**：先跑 **A (fixed-token, 256/64)** —— 最简单、最稳定、cache 容易。如果 evidence 有显著段落结构（p90 newlines ≥ 2），再跑 C 看是否有增益。

### 3.2 关键参数 / Key parameters

| 参数 | 默认 | 调节范围 | 影响 |
|---|---|---|---|
| `chunk_size` | 256 | 128 / 256 / 384 | 跟 bge-m3 max_seq_length 对齐；不要超过 |
| `overlap` | 64 | 0 / 32 / 64 / 128 | 大 = 索引膨胀 + recall 略升；小 = 边界 chunk 丢信息 |
| `stride` | 192 | = chunk_size - overlap | 滑窗步长 |
| `min_chunk_tokens` | 50 | 30 / 50 / 80 | 末尾过短 chunk 丢弃（语义不全） |

**默认锁 256/64/192/50**，先验证整体方向能不能跑通。后期再微调。

### 3.3 Chunk ID schema

每个 passage 切成多 chunk 后，chunk_id 命名规则：

```
原 passage:  evidence-12345
切完:       evidence-12345-c0, evidence-12345-c1, evidence-12345-c2
```

`-c{N}` 后缀。chunk_id 可以无歧义 split 回 passage_id：

```python
def chunk_to_passage_id(chunk_id: str) -> str:
    return chunk_id.rsplit("-c", 1)[0]
```

**关键约束**：保证原 passage_id 里**没有** "-c{digit}" 模式，否则 split 失败。`evidence.json` 实测命名都是 `evidence-{num}` 形式，安全。

---

## 4. Index Management / 索引管理

### 4.1 写哪 / Where indexes live

新策略产物**与现有索引并存**（不覆盖 baseline）：

```
outputs/
├── bm25_index/                    # 现有 baseline（passage-level）
├── dense_index/                    # 现有 baseline（passage-level）
├── bm25_index_chunked/             # chunking 实验产物
│   └── strategy_{A,B,C}/
└── dense_index_chunked/
    └── strategy_{A,B,C}/
```

每个 strategy 单独 sub-dir，方便 ablation 切换。

### 4.2 重建成本 / Rebuild cost

| 工作 | 时间（4080 SUPER） | 磁盘 |
|---|---|---|
| Chunking 1.2M passages → ~2-3M chunks | ~5 min (CPU) | ~250 MB JSONL |
| BM25 re-index | ~10-15 min | ~300 MB |
| Dense re-encode (bge-m3, fp16, bs=64) | **~3-4 h** | ~15 GB FAISS + chunks dir |
| Total per strategy | **~4 h** | ~15 GB |

如果跑 3 个 strategy 全 bake-off → 12h GPU + 45 GB disk。**预算紧，先只跑 A，看效果再决定要不要 C**。

### 4.3 Retrieval 接入 / Pipeline integration

`RetrievalPipeline.retrieve()` 返回 chunk-level 结果 → 必须 aggregate 到 passage：

```python
# pseudo-code in pipeline.py
def retrieve(self, claim):
    raw = ... # 现有逻辑，但 evidence_corpus = chunks_corpus
    # raw = [(chunk_id, text, score), ...]
    
    # Aggregate to passage-level (MAX score):
    by_passage: dict[str, tuple[str, float]] = {}
    for chunk_id, text, score in raw:
        pid = chunk_to_passage_id(chunk_id)
        if pid not in by_passage or score > by_passage[pid][1]:
            by_passage[pid] = (chunk_id, score, text)
    
    # Sort by score desc, return top-final_k passages
    sorted_passages = sorted(by_passage.items(), key=lambda kv: -kv[1][1])
    top_k = sorted_passages[:cfg.final_k]
    return [(pid, evidence_corpus[pid]) for pid, _ in top_k]
```

**Aggregate 用 MAX score**（不是 sum / avg）：
- MAX = "如果 passage 任一 chunk 跟 claim 强相关，passage 就相关" → recall-friendly
- SUM 会偏好长 passage（chunks 多）
- AVG 会被弱 chunks 稀释

### 4.4 Pool 深度 / Pool depth adjustment

因为 chunk-level pool 后要 dedup 到 passage-level，**必须把 `bm25_top` / `dense_top` 加大** 才能保证 unique passage 数足够。

经验值：
- 现有 `bm25_top=200, dense_top=200, fuse_top=150`
- Chunked 版：**`bm25_top=600, dense_top=600, fuse_top=400`**（3× 系数，按 chunks-per-passage 平均 ~2-3 估）
- `final_k=20` 不变（passage-level 输出）

---

## 5. Implementation / 实现

### 5.1 新文件 / New files

| 文件 | 用途 | 行数 估 |
|---|---|---|
| `src/retrieval/chunking.py` | 策略 A/B/C 实现 + `chunk_passages()` driver + `chunk_to_passage_id()` 工具 | ~150 |
| `scripts/build_indexes_chunked.py` | 调 `chunking.py` 切分 → BM25 build → Dense build；分 strategy sub-dir | ~120 |
| `scripts/eval_chunking.py` | 类似 `retrieval_ceiling.py` 但加 chunk → passage aggregate；对比 baseline vs 各 strategy 的 recall@k | ~180 |

### 5.2 改的文件 / Modified files

| 文件 | 改动 |
|---|---|
| `src/retrieval/pipeline.py` | `RetrievalConfig` 加 `chunk_aggregate: str = "max"`；`retrieve()` 加 aggregate 路径（cfg.chunked=True 时启用） |
| `src/retrieval/bm25.py` | 无需改（已支持 generic corpus 加载） |
| `src/retrieval/dense.py` | 无需改（chunks 也是 dict[id, text]） |
| `scripts/phase1_eval.py` | 加 `--chunked-index-dir PATH` flag，指向 `outputs/{bm25,dense}_index_chunked/strategy_X` |

### 5.3 API skeleton — `src/retrieval/chunking.py`

```python
from typing import Iterator
from transformers import AutoTokenizer

class FixedTokenChunker:
    def __init__(self, tokenizer, chunk_size=256, overlap=64, min_tokens=50):
        self.tok = tokenizer
        self.chunk_size = chunk_size
        self.stride = chunk_size - overlap
        self.min_tokens = min_tokens
    
    def chunk(self, passage_id: str, text: str) -> list[tuple[str, str]]:
        """Returns [(chunk_id, chunk_text), ...]. Empty list if text too short to chunk."""
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) <= self.chunk_size:
            return [(passage_id, text)]  # passthrough, no chunking
        chunks = []
        for start in range(0, len(ids), self.stride):
            window = ids[start : start + self.chunk_size]
            if len(window) < self.min_tokens:
                break
            chunk_text = self.tok.decode(window, skip_special_tokens=True)
            chunks.append((f"{passage_id}-c{len(chunks)}", chunk_text))
        return chunks


def chunk_passages(evidence: dict[str, str], strategy: str = "fixed_token",
                   **kwargs) -> Iterator[tuple[str, str]]:
    """Yield (chunk_id, chunk_text). Use bge-m3 tokenizer to count."""
    from src.paths import MODELS_DIR
    tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR / "bge-m3"))
    if strategy == "fixed_token":
        chunker = FixedTokenChunker(tokenizer, **kwargs)
    elif strategy == "sentence_group":
        chunker = SentenceGroupChunker(tokenizer, **kwargs)
    elif strategy == "paragraph_aware":
        chunker = ParagraphAwareChunker(tokenizer, **kwargs)
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    
    for pid, text in evidence.items():
        yield from chunker.chunk(pid, text)


def chunk_to_passage_id(chunk_id: str) -> str:
    """evidence-12345-c0 → evidence-12345. Idempotent: no suffix → returns as-is."""
    if "-c" in chunk_id:
        base, suffix = chunk_id.rsplit("-c", 1)
        if suffix.isdigit():
            return base
    return chunk_id
```

### 5.4 Driver — `scripts/build_indexes_chunked.py`

```bash
# Build A (fixed-token):
python -m scripts.build_indexes_chunked --strategy fixed_token \
    --chunk-size 256 --overlap 64 --out-dir outputs/bm25_index_chunked/A
# 自动用同 strategy 也建 dense_index_chunked/A
```

CLI flags:
- `--strategy {fixed_token, sentence_group, paragraph_aware}`
- `--chunk-size N` (default 256)
- `--overlap N` (default 64)
- `--out-dir PATH` (auto-derive bm25 + dense sub-dirs)
- `--skip-bm25` / `--skip-dense` (incremental)

---

## 6. Evaluation / 评估

### 6.1 Bake-off — `scripts/eval_chunking.py`

类似 `retrieval_ceiling.py`，但对比维度是 **baseline (passage-level) vs strategy A/B/C (chunked)**：

```bash
python -m scripts.eval_chunking --dataset diag_test \
    --strategies fixed_token --out outputs/eval_phase1/chunking_audit.md
```

产出表（重点看 r@20）：

| Config | Index | r@5 | r@10 | r@20 | r@50 | r@100 | Δ vs baseline |
|---|---|---|---|---|---|---|---|
| baseline (no chunking) | passage-level | 0.200 | 0.273 | 0.360 | 0.485 | 0.579 | - |
| chunking A (fixed 256/64) | chunk-level → max-agg | ? | ? | **?** | ? | ? | ? |

### 6.2 端到端 / End-to-end

如果 r@20 提升 > 0.03 absolute，跑 Track 2：

```bash
python -m scripts.phase1_eval --tracks 2 --prompts v1 --dataset diag_test \
    --chunked-index-dir outputs/bm25_index_chunked/A  # 加新 flag
```

### 6.3 dev_holdout 最终确认 / Final dev_holdout check

仅当 chunking bake-off + Track 2 都 pass 时跑（吃 D-006 配额最后 1 次）：

```bash
python -m scripts.eval_chunking --dataset dev_holdout --strategies fixed_token
python -m scripts.phase1_eval --tracks 2 --prompts v1 --dataset dev_holdout \
    --chunked-index-dir outputs/bm25_index_chunked/A
```

---

## 7. Decision Gates / 决策节点

### Gate A — chunking bake-off (`eval_chunking.py` on diag_test)

| 现象 | 行动 |
|---|---|
| ✅ recall@20 ≥ 0.40 (+0.04 vs baseline 0.360) | 进 Gate B 端到端 eval |
| ⚠️ recall@20 ∈ [0.37, 0.40] | marginal；可选进 Gate B，但 SFT 解锁无望 |
| ❌ recall@20 < 0.37 OR 任意 r@k 退化 | 放弃 chunking，进 Phase 6 ablation |

### Gate B — end-to-end (Track 2 on diag_test)

| 现象 | 行动 |
|---|---|
| ✅ HM ≥ 0.235 (+0.022 vs baseline 0.213) | lock 新 default，进 Gate C |
| ⚠️ HM ∈ [0.215, 0.235] | enable chunking 但不重训 SFT |
| ❌ HM < 0.215 | 回退；chunking 留 ablation |

### Gate C — dev_holdout final（仅 Gate B success）

| 现象 | 行动 |
|---|---|
| dev recall@20 ≥ diag recall@20 − 0.03 | 真正 locked，写决策日志 |
| dev recall@20 落 > 0.05 | 过拟合 diag → 回退 baseline |

---

## 8. Risks / 风险

| 风险 / Risk | P | 缓解 / Mitigation |
|---|---|---|
| **Profile verdict 是 Marginal/Unlikely** → chunking 无效 | high (40%) | 即停，6 day deadline 不冒险，直接 Phase 6 |
| Dense re-encode 3-4h 超预算 | med | 先跑 BM25-only 验证方向（10 min）；BM25 r@20 不动 → dense 也不会动 |
| Chunk → passage MAX aggregate 不是最优 | low | eval_chunking 支持 `--aggregate {max, sum, mean}` 三档对比 |
| 索引磁盘 15 GB 装不下 | low | AutoDL 实例 ~50 GB free，3 strategy = 45 GB 也撑得住；超了删一个 |
| 现有 SFT 数据基于 passage-level retrieval 构建 → chunking 后 train/inference mismatch | med | SFT 已经全部失败 (D-019)，不再训，不存在此问题 |
| **chunked submission 误把 chunk_id 写出去** | med | `chunk_to_passage_id()` 在 pipeline 出口强制调用；测试加 invariant check |
| 长尾 chunks (size < min_tokens 50) 被丢 → 漏召回 | low | min_tokens=50 已经很激进；后期 audit 失败再调 |

---

## 9. Time Budget / 时间预算

| Stage | 任务 | Time | Cumulative |
|---|---|---|---|
| 1 | Profile evidence + verdict 判定 | 0.5 h | 0:30 |
| 2 | Write `src/retrieval/chunking.py` + unit tests | 1.5 h | 2:00 |
| 3 | Write `scripts/build_indexes_chunked.py` | 1.0 h | 3:00 |
| 4 | Build A (BM25 + dense) on AutoDL | **4.0 h** GPU (mostly idle wall-time) | 7:00 |
| 5 | Write `scripts/eval_chunking.py` + run on diag_test → Gate A | 1.0 h | 8:00 |
| 6 | [if Gate A pass] Track 2 end-to-end → Gate B | 0.5 h | 8:30 |
| 7 | [if Gate B pass] dev_holdout final → Gate C | 0.5 h | 9:00 |
| 8 | 集成 (pipeline.py / phase1_eval flag) + commit | 0.5 h | 9:30 |
| **Total** | | **~9-10 h** | 单线（dense build 期间可并行写脚本）|

**并行机会**：Stage 4 dense build 期间，Stage 5/3 写脚本可同时进行 → 实际人工时间 ~5-6h。

---

## 10. 未来扩展 / Future extensions（仅当 chunking 成功）

如果 chunking + 现有 baseline 把 recall@20 拉到 ≥ 0.45：

1. **重启 reranker FT** —— chunk-level reranking 让 reranker 看到更小、更聚焦的语义单元，pos/neg 判别可能更容易（消解 plan §15.3 的 task mismatch 部分诱因）
2. **重启 SFT** —— chunk-level retrieval 给 SFT 训练数据提供更聚焦的 evidence context，可能解 D-019 distribution mismatch

但这些是「假如 chunking 成功」的下一步，不是现在的承诺。先做完 chunking 看结果。

---

## 11. Implementation Checklist / 实施 checklist

按序勾选 / Tick in order:

- [ ] 0. AutoDL: `source /etc/network_turbo && cd ~/autodl-tmp/NLP-A3 && git pull`
- [ ] 1. 跑 `python -m scripts.profile_evidence`，看 verdict
  - [ ] verdict ≥ "Worth trying" → 继续
  - [ ] verdict < "Worth trying" → 跳到 Phase 6 路径，本 plan 留 ablation
- [ ] 2. 写 `src/retrieval/chunking.py`（FixedTokenChunker + chunk_to_passage_id）
- [ ] 3. 本地 unit test `tests/test_chunking.py`（passage roundtrip + tokenizer accuracy）
- [ ] 4. 写 `scripts/build_indexes_chunked.py`
- [ ] 5. AutoDL: build strategy A（BM25 ~15 min + dense ~3-4h）
- [ ] 6. 写 `scripts/eval_chunking.py`
- [ ] 7. 跑 `scripts.eval_chunking --dataset diag_test --strategies fixed_token` → Gate A
- [ ] 8. [if Gate A pass] 改 `pipeline.py` `RetrievalConfig` 加 chunk_aggregate + chunked-mode flag
- [ ] 9. [if Gate A pass] 改 `phase1_eval.py` 加 `--chunked-index-dir`
- [ ] 10. [if Gate A pass] 跑 `phase1_eval --tracks 2 ... --chunked-index-dir ...` → Gate B
- [ ] 11. [if Gate B pass] 写决策日志到 `optimization_plan.md` §10 + `debug_log.md` 新一条复用经验
- [ ] 12. [if Gate B pass] dev_holdout final check → Gate C
- [ ] 13. [if Gate C pass] 锁 chunking 当生产；可选重建 SFT 数据走 §10 后续

---

## 12. 关联文件 / Related files

- `optimization_plan.md` §10（决策日志）
- `reranker_finetune_plan.md` §15（reranker FT 失败 ablation，本 plan 是 pivot 后继任者）
- `scripts/profile_evidence.py` —— 决定要不要做 chunking 的 gatekeeper
- `outputs/evidence_profile.md`（待生成）—— Profile 产出 + verdict
- `src/retrieval/{bm25,dense,pipeline}.py` —— 现有 retrieval stack 接口
- `debug_log.md` 复用经验 39 —— reranker 失败的 task mismatch 诊断（chunking 不在此根因下，是不同 lever）
