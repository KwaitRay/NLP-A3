# Implementation Progress Log

> Live status tracker. Update when crossing milestones. Plan in `~/.claude/plans/fancy-mapping-lemur.md`.

## 2026-05-13 — Session 15 (Phase 3.5b retrieval audit + HyDE + pad-alignment pivot)

Three concrete wins + one negative result + one new strategic pivot.
The retrieval-first strategy (D-019) ran its course; we now know exactly
where the retrieval ceiling is and have moved on to fixing training data
alignment.

### Win 1 — no-rerank Track 2 baseline lifted +0.030 HM

After locking `RetrievalConfig.use_rerank=False` (Phase 3.5b mode=retriever
audit, debug_log 复用经验 35), reran Track 2 v1:

  - F:  0.135 → **0.151**  (+0.016)
  - Acc: 0.397 → 0.364     (−0.033)
  - HM:  0.201 → **0.213**  (+0.012, +0.030 over original k=5 baseline)
  - predicted NEI: 2.5% → 1.7% (slightly less NEI)
  - non-NEI acc: 0.580 → 0.531

Trade-off: F gains more than Acc loses because removing the (negative-
contribution) reranker improves recall at the cost of slightly noisier
top-20 ordering, which the LM has to disentangle. Net HM positive.

### Win 2 — Phase 3.5b LLM rewrite cache built

`scripts.rewrite_queries --splits diag_test` produced 121 cached
{claim, HyDE, sub_claims} records in ~24 min on 4080 SUPER. Sample
inspection shows HyDE introduces domain-specific keywords absent from
the original claim (e.g. "Eocene", "Holocene", "pre-industrial baseline
of 280 ppm" for a stomata/ice-core CO2 claim). Quality good enough to
proceed to retrieval audit.

### Negative result — HyDE/sub-claims don't help in SFT context

`scripts.retrieval_ceiling --mode llm_rewrite --no-rerank` four-way
audit (baseline / HyDE only / sub-claims only / HyDE+sub-claims):

  recall@k          k=5    k=10   k=20   k=50   k=100
  ----------------  -----  -----  -----  -----  -----
  baseline          0.201  0.300  0.357  0.467  0.558
  HyDE only         0.164  0.270  0.357  0.506  0.602
  sub-claims only   0.151  0.224  0.331  0.479  0.583
  HyDE + sub-claims 0.193  0.248  0.339  0.511  0.616

**recall@20 (SFT context length) ceiling is locked at 0.357**. HyDE +
sub-claims actively HURT top-20 (0.339), only winning at k=50/100 with
+0.044 / +0.058 over baseline. This differs from MS MARCO/NQ HyDE
literature (which reports top-5 gains) — explained by domain shift +
already-saturated fused-no-rerank top-5 (which can't be improved).

debug_log 复用经验 36 captures: HyDE's marginal benefit is in long-tail
recall, not top-k where SFT lives; for in-context RAG, plain fused
beats multi-query.

### Strategic pivot — pad_with_random=False (D-019 副推论)

Retrieval-side is now exhausted (rerank, fusion_w, synonym, HyDE all
tried). The recall@20 = 0.357 IS the ceiling.

But the SFT-collapse hypothesis (复用经验 34) says the model shortcuts
"high-noise input → output NEI" because training has ~96% noise per
non-NEI sample (1-5 gold + 15-19 random pad) while inference has ~67%
noise. The two distributions overlap, letting the model use noise ratio
as a label proxy.

Fix: `src/build_stage0.py` train kwargs `pad_with_random: True → False`.
Now non-NEI training samples contain ONLY the 1-5 real gold ev with
zero random padding. Model is forced to learn semantic discrimination,
not noise-ratio shortcuts. Local rebuild verified:

  Records: 1567 (unchanged)
  Label distribution: identical to v3-rebalanced (sanity OK)
  n_shown was 100% at 20, now spreads:
    1: 14.0%, 2: 18.4%, 3: 13.6%, 4: 9.1%, 5: 44.9%  (NEI=5 gold)

### Next session

Run on AutoDL:

    python -m src.build_stage0 --force         # rebuild v2 with pad=False
    python -m scripts.run_sft 2>&1 | tee ...   # ~1.5h SFT v6
    swift export --adapters .../checkpoint-final --merge_lora true \\
        --output_dir .../merged_v6
    python -m scripts.phase1_eval --tracks 2,3 --prompts v1 \\
        --dataset diag_test --sft-merged-dir .../merged_v6
    python -m scripts.diagnose_phase1 --dataset diag_test

Track 3 v6 target: HM > 0.213 (current Track 2 baseline). If still
collapses, retrieval + alignment both exhausted → accept Track 2 v1
HM 0.213 as final submission. Three-failure-story ACL report:

  1. prompt engineering can't teach base what it doesn't know
     (复用经验 21)
  2. retrieval ceiling caps F-score under any rerank/fusion/HyDE config
     (复用经验 22, 35, 36)
  3. SFT collapses on train/inference distribution mismatch despite
     class rebalance (复用经验 32, 34) — pad alignment is the last
     attempted fix

That spine is publishable per README §307 regardless of v6 outcome.

## 2026-05-12 PM — Session 14 (SFT v3-rebalanced also collapses → retrieval-first pivot)

The rebalanced SFT data didn't fix anything. After v2-cut-1 (NEI 79% →
Track 3 HM 0.140) we cut NEI to 38.7% and retrained. Result: Track 3
HM **0.140 again** (identical), predicted NEI **94.2%** (slightly worse
than 92.6%), non-NEI acc **0.049** (worse than 0.062). The trained model
on rebalanced data is basically the same as the broken one — class
imbalance was not the root cause.

This forced a real diagnosis. The training data v3 layout:

  - 606 nei_underspec records: 5 real gold ev each, all off-topic
  - 961 SUPPORTS/REFUTES/DISPUTED records: 1-5 real gold ev each, padded
    to k=20 with random non-gold ev (`pad_with_random=True` in
    build_dataset)
  - At training time the model sees ~96% irrelevant evidence on non-NEI
    samples too

At inference time RAG returns top-20 with recall@20 = 0.333 (Phase 3.5
audit). That's ~6.6 gold ev + 13.4 noise = ~67% noise per sample. From
the model's perspective:

  - During training, "high noise ratio" → either NEI (if all noise) or
    non-NEI (if a few gold mixed in)
  - During inference, RAG noise ratio ~67% is ABOVE the training non-NEI
    threshold (non-NEI samples had 1-5 gold / 20 ≈ 5-25% gold) and
    closer to the NEI pattern
  - So model shortcuts: noise-heavy retrieval → output NEI

**Root cause = train-inference distribution mismatch driven by retrieval
recall ceiling**. Rebalancing NEI label proportion doesn't help because
the input representation gap is preserved.

### Strategic pivot (design D-019)

SFT is paused. Phase 3.5b: optimize retrieval first.

  - Current: full pipeline @ k=20 → macro recall@20 = 0.333
  - Target: recall@20 ≥ 0.50
  - If 3 remaining retrieval_ceiling.py modes (retriever / fusion_w /
    synonym_expand) can reach it via parameter tuning, lock new
    RetrievalConfig and retry SFT
  - If not, escalate to LLM-driven query rewrite (HyDE + sub-claim
    decomposition via base Qwen3.5-4B, ~half day to implement)
  - After retrieval is fixed, retry SFT — possibly with
    `pad_with_random=False` to keep training distribution clean

### debug_log 复用经验 34

Captures:
  - The "rebalance doesn't help" surprise + numerical evidence
  - The pad_with_random / RAG recall coupling theory
  - General lesson: class balance is a diagnostic signal, not always the
    root cause; check input representation alignment too
  - Reusable rule: SFT on retrieved RAG context requires recall@k ≥ 0.5
    or accept that training will collapse to majority class

### v2-cut-1 + v3-rebalanced kept as ablation evidence

Per README §307 ("clear insight into why the method works or fails"),
both SFT failures are now project narrative gold:

  - Story 1: prompt engineering can't teach base model what it doesn't
    know (Phase 2, 复用经验 21)
  - Story 2: retrieval ceiling caps F-score below 0.12 (Phase 3.5, 复用
    经验 22)
  - Story 3: SFT on small data + low recall = collapse to majority
    (Phase 4 v2-cut-1 + v3-rebalanced, 复用经验 32 + 34)

These three failure-then-diagnosis trails are the Soundness + Results
maxima for the final ACL report.

### Next session

Run retrieval audit's 3 remaining modes. Based on whether recall@20
reaches 0.50, decide:
  - 9a (no LLM): lock new config, rebuild SFT data with retrieval fix +
    `pad_with_random=False`, retry SFT
  - 9b (LLM rewrite): implement `scripts/rewrite_queries.py`, multi-query
    pipeline, re-audit, then 9a

## 2026-05-12 PM — Session 13 (SFT v2 first cut class-collapse + rebalance)

SFT training finished overnight; Track 3 evaluation revealed catastrophic
class-collapse — model learned to predict NEI for 92.6% of claims, killing
non-NEI acc to 0.062. Took the rest of the day to diagnose, fix, and add
a sanity check so this can't happen again silently.

- **Track 3 eval failure**: merged SFT model via `swift export --merge_lora
  true` (after AutoModelForImageTextToText loader breakdown, debug_log 复用
  经验 31). Track 3 HM 0.140 vs Track 2 baseline 0.201 — SFT made things
  worse by 6pp. diagnose_phase1 cross-run summary table showed the smoking
  gun: Track 3 predicted NEI share 92.6% (vs gold 33%), non-NEI acc 0.062.
  Model achieved 97.5% NEI acc but at the cost of 81 non-NEI claims now
  classified as NEI.
- **Root cause** (debug_log 复用经验 32): training labels 79.1% NEI from
  two interacting sources:
  - `nei_underspec ×4` (weak_buckets) → 1212 real NEI samples (29% of v2)
  - `n_hard_neg=1` (every claim gets one synthetic NEI hard-neg, also
    scaled by weak_buckets factors) → **2083 synth NEI samples (50% of v2)**
  - Total: 3295 / 4166 = 79.1% NEI. With SGD on a heavily imbalanced
    dataset, the loss-minimizing strategy IS "always predict majority class".
- **v2 rebalance**:
  - `nei_underspec ×4 → ×2` (real NEI halved)
  - `disputed_conflict ×2 → ×3` (DISPUTED still weak, push harder)
  - `n_hard_neg = 1 → 0` (drops the 2083-sample dominant NEI source)
  - Result: 1567 records (vs 4166), NEI label share 79.1% → **38.7%**
    (close to gold 33%, with light +5pp oversample). Training will take
    ~1.5h instead of 4h (×2.7 fewer records, same eff_bs).
- **Sanity check landed in `build_stage0.py`**: `_print_label_dist()` prints
  SFT label distribution vs gold for every split it builds. Warns when any
  label's `sft_share / gold_share` ratio is > 2× or < 0.5×. The first cut
  would have fired at NEI 2.39× — we'd have caught it before a 4h SFT run
  wasted itself.
- **What's next**: AutoDL `git pull` → `python -m src.build_stage0 --force`
  → verify the distribution table shows NEI ~1.17× gold → retrain SFT
  (~1.5h) → merge_lora → retry Track 3 evaluation. Target: NEI acc in
  [0.40, 0.60] (lifted from base 0.025, not collapsed to 0.97); non-NEI
  acc ≥ 0.45 (preserved from base 0.58, not catastrophic 0.06); HM ≥ 0.25
  (vs Track 2 baseline 0.201).
- **Lesson for future SFT iterations**: any time `n_hard_neg > 0` is on
  AND `weak_buckets` has a label-coherent boost (e.g. NEI-scenario
  oversample), check the additive effect on global label share. The two
  mechanisms compound, and 50%+ majority share will collapse the model.

## 2026-05-12 — Session 12 (SFT training kickoff — env-stack 三连碰 + Track 3 wiring)

From "v2 data built, ready to train" to "training actually started" took
three independent env-stack discoveries + a 4-loop JupyterLab autosave war,
all logged into debug_log 复用经验 27-30. Training is now running; this
entry records the full path so future-me / collaborators don't redo it.

### Env-stack 三连碰

1. **ms-swift 4.2.0 × torch 2.5.1 = FSDP2 ImportError** (复用经验 27):
   `swift/callbacks/activation_cpu_offload.py` 在 module-load 期 import
   `torch.distributed.fsdp.FSDPModule` (torch 2.6+ API). AutoDL 镜像锁
   torch 2.5.1+cu124 (debug_log Issue 13 历史决策)，不能升 torch (会破
   flash-attn 2.x / bitsandbytes / fla 全栈). Decision: **不降 ms-swift**
   (会丢 `--tuner_type` + thinking 三件套), **不升 torch** (破栈), 而是
   写 `scripts/patch_swift_fsdp2.py` 把那一行 import 包成 try/except + stub.
   Idempotent, callback 用不到所以 stub 永远不被实例化. Wired 到
   `cell-autodl-setup-2` 自动调用，新建 AutoDL 实例不再撞.

2. **transformers 5.2 model_type 'qwen3_5' KeyError** (复用经验 11 已知，
   但 setup-2 没 pin 落地):
   `requirements.txt` 之前没 hard-pin transformers，pip 装到 4.x →
   AutoConfig 不认识 qwen3_5. Fix: pin `transformers==5.2.*` per
   `materials/qwen3.5_key_points.docx` (5.1 缺 qwen3_5, 5.3 视频坏).
   重装后 model_type 识别 OK.

3. **peft × transformers 5.2 = HybridCache ImportError** (复用经验 28，新):
   transformers 5.x 删了 `HybridCache`，但 peft 0.14 还硬编码
   `from transformers import HybridCache`. Diagnosis 被误导一次—
   `patch_swift_fsdp2.py` 的 `try: import swift` 笼统 catch 报 "ms-swift
   not installed"，真正错误在 traceback 倒数第 2 行. Fix: `pip install -U peft`
   到 0.17+ (post-HybridCache-removal). 已记 TODO 改 patch 脚本的 error UX.

### JupyterLab autosave vs git pull 4-loop deadlock (复用经验 29)

用户撞了 4 次完全相同的错：

```
error: Your local changes to the following files would be overwritten by merge:
        notebooks/notebook_autodl.ipynb
```

试过 `git checkout --`、`git stash` (×2)、`rm + git checkout HEAD --` (×2)，
全部失败。根因：JupyterLab 那个 notebook 还在浏览器开着，每 ~30s autosave
把内存 cell outputs 写回磁盘 — git pull 永远追不上 autosave 节奏.

Resolution recipe:
1. JupyterLab UI: `File → Close and Shut Down Notebook` (必须 Shut Down)
2. 终端: `git fetch origin main && git reset --hard origin/main`
3. JupyterLab 重开 + Restart Kernel

Long-term fix: `nbstripout --install` 装 git filter 自动剥 outputs. 已
加进 `requirements.txt` (torch 2.5 / 2.6 两份).

### Requirements 拆分 (2 files)

- `requirements.txt` (torch 2.5.1 default): `ms-swift==4.2.0` hard pin +
  必跑 `python -m scripts.patch_swift_fsdp2` 后处理 + `nbstripout`.
- `requirements-torch26.txt` (future torch 2.6+): `ms-swift>=4.2.0`
  unpinned (FSDP2 native), 推荐 flash-attn 2.8.3.

两份都 pin `transformers==5.2.*` + `qwen_vl_utils>=0.0.14` (Qwen3.5 必需，
与 torch 无关). 跨场景部署不用猜.

### SFT 启动 + loss 读法 (复用经验 30)

Training kicked off at 2026-05-12 ~04:05 UTC, target run dir
`/root/autodl-tmp/nlp_a3_cache/sft-out/v4-20260512-040505/`. 配置:

```
QLoRA 4-bit (NF4 + bf16 compute)
LoRA r=16 / α=32 / dropout 0.05 / target_modules=all-linear
BS=2 × GA=8 (eff_bs=16), lr 2e-4 + warmup 0.03
max_len=1536, gradient_checkpointing on, liger_kernel on, group_by_length on
thinking 三件套: --enable_thinking false / --add_non_thinking_prefix true
                 / --loss_scale ignore_empty_think
3 epochs × 4166 records / 16 = 783 steps
save_steps=200, eval_steps=200, save_total_limit=3
```

实测 step 1 loss 0.1146 → step 25 loss 0.019 → step 35 loss 0.010.
**Loss 低不是 bug** — prompt masking + 短响应 ('LABEL ##[..]##' ≈ 8 tokens)
+ base 模型已认识 4 个 LABEL 词. 真信号看 Track 3 vs 2 acc/HM delta.
VRAM 11.4 GB / 31.5 GB (余量 20 GB), ~19 s/step (稳态), 预估 ~4h 总.

### Track 3 wired into phase1_eval

`scripts/phase1_eval.py` 加 `--sft-adapter PATH` flag:
- `PeftModel.from_pretrained(base_model, adapter_path)` 原地包装 (不开第二
  份 base 占 VRAM)
- `--tracks 3` 强制 require adapter，校验在加载模型之前 fire
- Track 3 复用 run_track2 (同 RAG pipeline, 同 ZeroShotInferer, 只换 model)
- 输出 `track3_{prompt}_{dataset}.{json,md}`, diagnose_phase1 自动 discover

训完一行命令直接拿对比:
```
python -m scripts.phase1_eval --tracks 2,3 --prompts v1 --dataset diag_test \
    --sft-adapter /root/autodl-tmp/nlp_a3_cache/sft-out/v4-20260512-040505/checkpoint-final
python -m scripts.diagnose_phase1 --dataset diag_test
```

### 关键决策回顾

| 决策 | 选择 | 替代方案 (rejected) | Why |
|---|---|---|---|
| ms-swift 4.2.0 vs older | 4.2.0 + patch | 降到 < 4.2 (掉到 swift/llm/ 老布局) | 老布局没 `--tuner_type` / thinking 三件套，等于回到 Issue 9 初态 |
| torch 2.5 vs 2.6 | 留 2.5 | 升到 2.6+ | 升 torch 破 flash-attn 2.x / bitsandbytes / fla 全栈 (Issue 13 教训) |
| FSDP2 callback 处理 | patch import 为 optional + stub | 删整个 callback / 改 ms-swift fork | callback 用不到，patch 1 行最干净 |
| transformers 5.x | pin 5.2.* | 留 4.x | qwen3.5 model_type 在 5.0 才注册，4.x 不识别 |
| jupyter notebook git 卫生 | nbstripout | 手动 strip / 不 commit notebook | 自动化 0 维护成本 |
| Track 3 实现 | phase1_eval --sft-adapter | 在 notebook 里手写 | 命令行 + 自动 diagnose 比 notebook 灵活，无 kernel restart 摩擦 |

### 下次 session 第一句话

AutoDL 上 `ls /root/autodl-tmp/nlp_a3_cache/sft-out/v4-20260512-040505/`
确认 checkpoint-final 存在 → `phase1_eval --tracks 2,3 --sft-adapter ...`
→ 把 `diagnose_diag_test.md` 的 cross-run summary 表（含 Track 2 / 3 对比）
+ Track 3 per-label correctness 表贴回来。**关键指标：NEI acc 是否 ≥ 0.40，
HM 是否 ≥ 0.28**。

## 2026-05-12 — Session 11 (v2 SFT data built + notebook patched for AutoDL SFT kickoff)

The path from "Phase 4 implementation pushed" to "SFT is actually about to
train on AutoDL" had several surprise blockers — each documented in
debug_log.md as 复用经验 24-26.

- **v2 SFT data built locally + verified** (`python -m src.build_stage0 --force`):
  4166 records (×2.11 over v1's 1972). weak_buckets fired exactly as
  designed: nei_underspec 303→**1212 (×4.00)**, disputed_conflict 90→**180
  (×2.00)**, refutes_clear 98→**196 (×2.00)**. Untargeted buckets unchanged.
  Label distribution shifted: NEI 65.4% → **79.1%** (heavy NEI signal for
  base model's NEI capability gap); hard difficulty 14.3% → **24.1%**.
  n_shown stable at 19-20 (rejection-sampling edge case noted, harmless).
- **AutoDL SFT kickoff blocked by ModelScope re-download** (debug_log 复用经验 24):
  notebook's `sec2-5-download` unconditionally called `snapshot_download
  (cache_dir=CACHE_ROOT/model_cache/)`, which doesn't see the existing
  `models/Qwen3.5-4B/` placed by `scripts.download_models`. User caught
  it mid-download (13% / 21% on two safetensors shards) and asked why.
- **AutoDL `git pull` blocked by GFW** (debug_log 复用经验 25):
  `Failed to connect to github.com port 443 after 130930 ms`. Fix:
  `source /etc/network_turbo` (AutoDL official whitelist proxy for
  github / huggingface / pytorch / conda; per-shell scope).
- **Notebook patched — 9 cells, one commit**:
  - sec2-5-download: cache-first (prefer models/Qwen3.5-4B/ → fallback
    snapshot_download). Same pattern as `phase1_eval.py:load_model_and_tokenizer`.
  - sec3-5a: identical cache-first + auto bf16/fp16 by hardware.
  - sec2-5-train: full ms-swift v3.6+ flag refresh (`--train_type` →
    `--tuner_type`, `--quant_bits` not `--quantization_bit`, +thinking
    trio +liger +group_by_length +save_total_limit), DATA_PATH→v2.
  - sec2-6-code (DPO): same flag rename, +thinking flags.
  - sec1-4-code: k=20 + weak_buckets, writes v2.
  - sec2-3-code: drops explicit `final_k=5` (uses new default 20).
  - sec2-5-md / autodl-reconnect: doc references updated to v2.
  - autodl-setup-gpu: MAX_LEN 1024→1536 (k=20 evidence prompts can hit
    1024); QUANT_FLAG uses --quant_bits.
- **Stale `Qwen3___5-4B` triple-underscore paths cleaned** (debug_log
  复用经验 26): old ModelScope used `___` to escape `.` in folder names;
  current ModelScope writes `Qwen3.5-4B` directly (confirmed in user's
  download stdout). Replaced 5 lingering references (optimization_plan,
  TODO, test_qwen35_inference docstring) with `models/Qwen3.5-4B/` or
  removed `--model-dir` entirely now that cache-first auto-locates.
- **AutoDL pull conflict pattern observed** (informational): pushing a
  new notebook version into a directory where JupyterLab autosaved
  execution counts / cell outputs causes `git pull` to refuse with
  "untracked working tree files would be overwritten" or "local changes
  would be overwritten". Resolution recipe (also in TODO.md):
  `git checkout -- notebooks/*.ipynb` (lose autosave noise) or
  `git stash` + inspect-then-drop (preserve manual edits cautiously).
- **Repo state going into Session 12**: code + docs + notebook ready;
  v2 SFT data needs to be rebuilt on AutoDL (gitignored, doesn't pull);
  SFT training kickoff blocked only on running 5 cells.

## 2026-05-12 — Session 10 (Phase 3.5 final_k locked + Phase 4 weak_buckets implemented)

End-to-end retrieval audit + Phase 4 implementation. Repo state moves
from "knows retrieval is the bottleneck" to "production config locked,
SFT data design done, ready to train".

- **Phase 3.5 audit (final_k mode)** on AutoDL via
  `scripts.retrieval_ceiling --mode final_k`: macro recall@k curve
  k=5→0.119, k=10→0.210, k=20→0.333, k=50→0.485, k=100→0.579.
  Gold evidence sits at ranks 6-20 — current production k=5 was
  throwing it away.
- **End-to-end k=5/10/20 sweep on Track 2 v1**:
  - k=5  (current): F 0.117  Acc 0.422  HM 0.183  (baseline)
  - k=10:           F 0.131  Acc 0.388  HM 0.196
  - k=20:           F 0.136  Acc 0.397  **HM 0.203**  (winner)
  - Surprise: at k=20 the model almost never predicts NEI (0.025
    acc, 3 predicted out of 121). non-NEI acc shoots up to 0.580
    (vs k=5's 0.457). Trade-off net positive for HM but pushes
    100% of error onto the NEI bucket — exactly what Phase 4
    weak_buckets is designed to fix.
- **Phase 3.5 lock: `final_k=20`**.
  - `RetrievalConfig.final_k` default 5 → 20 with inline comment.
  - `phase1_eval --final-k` default also 20; non-default k auto-suffixes
    output filenames as `_kN` to preserve baseline tables.
  - `diagnose_phase1._strip_k_suffix` strips the suffix when looking up
    the underlying gold split.
- **Phase 4 weak_buckets implemented**.
  - `src/sft_dataset.py:build_dataset` adds
    `weak_buckets: dict[tuple[str, str], int] | None` parameter; per row,
    the max factor across matching (axis, bucket) keys is applied to
    both the real record and any hard-negative augmentations.
    `tests/test_sft_dataset.py` covers (a) factor scaling, (b) max-not-
    product semantics on overlapping matches, (c) empty/None no-op.
  - `src/build_stage0.step_sft` switched to k=20 across all three splits
    and bakes the Phase 4 config:
    `{nei_underspec:4, disputed_conflict:2, refutes_clear:2}`. Output
    files versioned `sft_*_v2.jsonl`; v1 kept on disk for ablation.
- **debug_log 复用经验 23**: the "information-density paradox" —
  larger k makes the base model assume an answer exists in the evidence
  set, so it stops predicting NEI. Quantified across 3 k values. Locks
  the rationale for k=20 + Phase 4 NEI oversample as a coupled pair.
- **optimization_plan §3.5 / §4.4 finalised**: §3.5 now records the
  audit result + Phase 3.5 lock; §4.4 replaces pseudocode with the
  actual implementation reference (signature + Phase 4 dict).
- **What's next**: AutoDL runs `python -m src.build_stage0 --force` to
  produce the v2 SFT files (~5s), then kicks off ms-swift LoRA training
  with `DATA_PATH=outputs/sft_data/sft_train_v2.jsonl`
  (~25-35 min on 4080 SUPER).

## 2026-05-12 — Session 9 (Phase 1 diagnosed + Phase 2 prompt sweep + retrieval ceiling discovered)

Three major findings in one session; the third reshapes the rest of the plan.

- **Phase 1 diagnosis closed (debug_log Issue 17)**. `diagnose_phase1.py`
  on `track{1,2}_v1_diag_test.json` ruled out the parser-fallback
  hypothesis. Track 1 confusion matrix shows the base model is
  actively classifying (SUPPORTS acc 0.658, REFUTES 0.591) but has
  ~zero capability on NEI (1/40 = 0.025) and DISPUTED (0/21 = 0.000).
  Track 1 Acc=0.3223 ≈ NEI gold fraction 0.3306 was a coincidence:
  the model nearly always picks SUPPORTS or REFUTES, accidentally
  matching 1 gold NEI claim. Track 2 (RAG) recovers NEI to 0.350 and
  DISPUTED to 0.286, at the cost of SUPPORTS (0.526) and REFUTES
  (0.500) which RAG mildly distracts. This is a **quantified replay
  of `optimization_plan.md §0.5.2 4a`** at n=121.
- **Phase 2 prompt sweep complete; v1 locked**. Ran v2/v3/v4 on Track 2:
  - v1 baseline: F=0.1169 Acc=0.4215 **HM=0.1830** (winner)
  - v2 nei_explicit: F=0.1087 Acc=0.4132 HM=0.1722. NEI rule fires
    (NEI predicted 25% → 42%, NEI acc 0.350 → 0.550) but over-applies:
    REFUTES acc 0.500 → 0.227.
  - v3 disputed_explicit: F=0.1108 **Acc=0.2562** HM=0.1547. DISPUTED
    over-predicted (24 → 73 / 121); REFUTES collapses to 0/22.
  - v4 v3 + 4 few-shot: F=0.1035 Acc=0.2893 HM=0.1524. Few-shot tempers
    DISPUTED slightly but kills NEI (0.350 → 0.050). Single NEI
    demo overfit.
  - Confirms §0.5.3 hard constraints 1+3 (NEI/DISPUTED are capability
    gaps, not prompt gaps — must SFT).
- **Retrieval ceiling discovered (key finding)**. All four prompt
  variants in Track 2 report **evidence recall ≈ 0.11** (macro 0.10-
  0.11, micro 0.09-0.10), invariant in prompt. The F-score ceiling
  under the current `RetrievalConfig(final_k=5, w_bm25=0.3,
  w_dense=0.7)` is **F ≈ 0.12, HM ≈ 0.21** even with perfect labels.
  This means Phase 4 SFT can lift label acc but won't move HM unless
  retrieval improves first.
- **Phase 3.5 inserted into optimization_plan.md** to audit retrieval
  before spending GPU-hours on SFT. Plan: sweep `final_k` (5→100),
  retriever ablation (BM25-only / dense-only / fused / +rerank),
  fusion weight (w_bm25 ∈ {0.1..0.9}), and synonym multi-query
  (via `src/query_rewrite.synonym_expand`). Implemented as
  `scripts/retrieval_ceiling.py` (pure retrieval, no LLM, ~3 min on
  4080 SUPER).
- **debug_log.md 复用经验 21+22** added:
  - 21: prompt can't teach concepts the base model lacks — quantified
    via v2 NEI +20pp / v3 DISPUTED +19pp counter-examples where the
    "explicit trigger" instruction over-applies.
  - 22: F-score ceiling = retrieval recall — every prompt/model sweep
    should include `evidence_recall` in the report; if it's invariant
    across variants, switch to retrieval optimization.

## 2026-05-11 — Session 8 (Phase 1 baseline executed; Track 1 numbers trigger NEI-default diagnosis)

First end-to-end Phase 1 run on AutoDL after all cache prerequisites
(BM25, dense, models, messages-format SFT data) landed. Headline numbers
came back fast but Track 1 Acc looked suspicious; spawned a diagnostic
tool rather than guessing.

- **Phase 1 eval ran clean** (`python -m scripts.phase1_eval --tracks
  1,2 --prompts v1 --dataset diag_test`, ~4.5 min total on 4080 SUPER):
  - Track 1 (no-RAG, greedy):  F=0.0000  Acc=**0.3223**  HM=0.0000  (69.7s / 121 claims)
  - Track 2 (RAG, greedy):     F=0.1169  Acc=0.4215   HM=0.1830  (199.3s / 121 claims)
  - Outputs at `outputs/eval_phase1/track{1,2}_v1_diag_test.{json,md}` +
    `summary_diag_test.md`.
- **Suspicion**: Track 1 Acc=0.3223 = 39/121 is one claim shy of
  `gold_NEI/n = 40/121 = 0.3306`. Two competing hypotheses:
  - **(a)** Parser fallback dominates: model output doesn't carry
    `LABEL ##[..]##`, `parse_response` returns `default_label='NOT_ENOUGH_INFO'`
    for ~all claims, accidentally hitting the NEI majority.
  - **(b)** Base model genuinely predicts a mix of labels but is biased
    toward NEI on the non-NEI claims.
  - Fix paths diverge sharply: (a) → prompt v2 / parser tweak; (b) →
    Phase 4 SFT data tilt. Must distinguish before acting.
- **`scripts/diagnose_phase1.py` added** — pure analysis on saved JSONs
  (no model loading). Per (track, prompt) it reports predicted-vs-gold
  label distribution, 4×4 confusion matrix, per-gold-label correctness,
  evidence recall (Track 2+), a defaulting-to-NEI heuristic flag, and
  sample mispredictions for eyeballing. CLI mirrors `phase1_eval`:
  `python -m scripts.diagnose_phase1 --dataset diag_test`. Writes
  `outputs/eval_phase1/diagnose_<dataset>.md` + stdout summary.
- **debug_log.md Session 3 + Issue 17** capture the full hypothesis split
  and the 复用经验 19-20 (don't trust majority-class-looking Acc;
  consider adding `--save-raw` to `predict_all` later).
- **TODO.md** gets a new Step 2.5 inserting the diagnostic before
  Step 3 (Phase 2 prompt sweep).
- **optimization_plan.md** §9 progress flips Phase 1 from "in progress"
  to "data collected, diagnosing"; §10 decision log gets the headline
  numbers as a new row.

## 2026-05-11 — Session 7 (data-format migration + AutoDL .bin→.safetensors workaround)

Same calendar day as Session 6 but distinct enough to log separately. Covers
the post-BM25 work and the AutoDL workflow gaps that surfaced when the user
tried to build the dense index.

- **Local SFT data regenerated** into ms-swift messages format
  (`outputs/sft_data/sft_{train,dev_holdout,diag_test}_v1.jsonl`):
  prior on-disk files were still in the old `{system, query, response}`
  schema even though `src/sft_dataset.py` had already been migrated. Ran
  `python -m src.build_stage0`; existence checks correctly skipped EDA /
  tagging / splits and rebuilt only SFT (~1.5 s evidence load, 0.1 s build
  per split, 1972 / 121 / 121 records).
- **`optimization_plan.md` §0.5 added** — bilingual "Base-model capability
  probe (precondition)" capturing the AutoDL smoke-test findings from
  `debug_log.md` Session 2 实测数据:
  - 4a no-RAG greedy: SUPPORTS/REFUTES correct, NEI→REFUTES (base lacks
    "I don't know" concept) → drives **NEI must be oversampled** hard
    constraint.
  - 4b RAG fake-ev greedy: `LABEL ##[i,j]##` format strict → **citation
    examples stay lean** (no format-only augmentation).
  - 4c SC easy SUPPORTS: 5/5 agreement → **prioritize DISPUTED/ambiguous
    augmentation** with r=2.0 multiplier in §4.3.
  - §4.3 augmentation strategy now cross-references §0.5.3 as precondition.
- **AutoDL build_indexes hit transformers CVE-2025-32434 mitigation**:
  ModelScope mirror serves `BAAI/bge-m3` as `pytorch_model.bin` only (no
  safetensors), and transformers >= some-version refuses torch.load unless
  torch >= 2.6. AutoDL is pinned at torch 2.5.1+cu124 for flash-attn /
  bitsandbytes / flash-linear-attention compatibility (debug_log Issue 13)
  — upgrading torch is high-risk.
- **Resolution**: `scripts/convert_bin_to_safetensors.py` — walks
  `models/*/`, skips dirs already carrying `*.safetensors`, converts
  single-file `pytorch_model.bin` via direct `torch.load` +
  `safetensors.save_file` (user-code torch.load is unrestricted; only the
  transformers wrapper enforces the CVE check). Renames `.bin` → `.bin.bak`
  for rollback. Sharded layouts get warn-and-skip.
- **TODO.md Step 1 expanded**: pipeline now runs `git pull` →
  `pip install -U modelscope huggingface_hub` → `python -m src.build_stage0`
  → `python -m scripts.download_models` → `python -m
  scripts.convert_bin_to_safetensors` → `python -m scripts.build_indexes`.
- **debug_log.md Issue 16 + 复用经验 19**: full root-cause writeup of the
  CVE / mirror-gap / firewall triple-bind plus the `HF_ENDPOINT=
  https://hf-mirror.com` fallback for future single-file補下.
- **design.md D-016**: codifies "convert offline locally, do **not**
  upgrade torch" as a binding decision; rationale ties back to D-013
  (T4/Ampere dual-path) and Issue 13 (instance rebuild for torch 2.5).
- **Commits pushed**: `2617051` doc alignment → `45b1dc2` build_stage0 step
  → `f78865a` safetensors converter + §0.5 + §4.3 prerequisite link. All
  on `origin/main`; AutoDL just needs `git pull`.

## 2026-05-11 — Session 6 (local prep complete, ready for AutoDL Phase 1)

Local prerequisites for Phase 1 evaluation now fully satisfied. Next session is AutoDL.

- **BM25 index built locally** (`outputs/bm25_index/bm25/`):
  `data.csc.index.npy`, `indices.csc.index.npy`, `indptr.csc.index.npy`,
  `params.index.json`, `vocab.index.json` (~200 MB total) +
  `outputs/bm25_index/ev_ids.txt`. Validates retrieval path can dry-run on
  Windows without needing AutoDL.
- **Doc sync**: `TODO.md` rewritten — Step 1 (local BM25) moved to "已完成";
  remaining AutoDL steps renumbered 1→4. Bottom guidance updated to reflect
  the new step numbers.
- **What remains pending (AutoDL only)**: dense index build (bge-m3 on 4080
  SUPER, ~15 min), Phase 1 baseline eval on `diag_test` (v1 prompt, ~10
  min), Phase 2 prompt sweep (v2/v3/v4, ~15 min). See `TODO.md` Steps 1-3.

## 2026-05-10/11 — Session 5 (AutoDL boot + Phase 1 scaffolding + bilingual plan)

Catches up the period between Session 4 and today; pushed across commits
`9465f9b` → `003122a` to `origin/main`.

- **AutoDL instance up**: PyTorch 2.5.1+cu124, RTX 4080 SUPER 31.5 GB VRAM,
  bf16 + flash-attn 2.x both supported. Smoke test
  (`scripts/test_qwen35_inference.py`) passes end-to-end with Qwen3.5-4B.
- **Phase 1 scaffolding** (all green-tested):
  - `src/prompt.py` — added `PROMPT_VARIANTS` dict with v1 (current baseline)
    through v4 (each layering one more constraint), all consumed by the new
    eval harness via `--prompts vN[,vM,...]`.
  - `scripts/build_indexes.py` — standalone BM25 + dense index builder,
    `--skip-dense` runs BM25-only (used locally today).
  - `scripts/phase1_eval.py` — Track 1 (no-RAG) / Track 2 (RAG) × prompt
    variant sweep harness. Writes `outputs/eval_phase1/track{1,2}_v{1..4}_
    {dataset}.{json,md}` plus `summary_{dataset}.md`. Per-bucket tables in
    Track 2 are sorted by HM ascending so weakest buckets surface for
    Phase 4 targeting.
  - `scripts/download_models.py` — one-shot fetch of Qwen3.5-4B + bge-m3 +
    bge-reranker-base + bge-small-en-v1.5 into `models/` (~11 GB).
- **Persistence refactor**: notebook `cell-1-sft-code` switched to
  cache-first; all model paths now flow through `MODELS_DIR` +
  `resolve_model_path()` so `models/` is authoritative on both local and
  AutoDL.
- **SFT / DPO data migration**: train + dev_holdout + diag_test rewritten
  into ms-swift `messages` standard format; all 8 unit-test suites green.
- **Documentation**:
  - `design.md` bumped to v1.1 (records D-011 through D-015, where D-015
    formalises the eval-driven SFT-data-design loop).
  - `optimization_plan.md` — new 6-phase bilingual (中文 + English) plan,
    executable counterpart to D-015.
  - `debug_log.md` Session 2 — Qwen3.5 / AutoDL pitfalls captured (mixed-
    thinking VL handling, `enable_thinking=False` + thinking-trio, T4 vs
    4080 dtype gating, transformers 5.x `apply_chat_template` returning
    `BatchEncoding` not tensor).
  - `TODO.md` — bilingual single-page recovery doc for tomorrow-self.
- **Models on disk**: `models/{Qwen3.5-4B,bge-m3,bge-reranker-base,bge-
  small-en-v1.5}/` — 4 directories, ~11 GB combined.

## 2026-04-30 — Session 4 (notebook annotated + design.md)

- **Notebook section status badges**: every sub-section header in
  `notebooks/notebook.ipynb` carries one of three markers (✅ verified
  locally / 🧪 stub-validated / ⏳ requires Colab). 15 headers tagged.
  Marker added to README cell explaining the legend; `outputs/dry_run_report.md`
  is the audit trail.
- **`design.md`** at project root — 809 lines, 18 sections covering data
  model, all 7 stages, code organisation, notebook layout, reproducibility,
  37-test matrix, risks, decision records, glossary. Chinese-primary,
  technical identifiers preserved verbatim. Cross-references plan,
  PROGRESS.md, dry_run_report.md.

## 2026-04-30 — Session 3 (dry-run wired)

`scripts/dry_run.py` validates the entire local pipeline in one command (~3 s):
env survey → Stage 0 idempotent re-run → artifact existence checks → Stage 1
class smoke imports → Stage 5+6 stub run with 275 synthetic predictions → all 8
unit-test suites. Writes `outputs/dry_run_report.md` summarising what's
verified vs what still needs Colab. Run `python -m scripts.dry_run` before
each Colab push.

## 2026-04-30 — Session 2 (Stage 5/6 added)

### Added since session 1

- **Stage 5 inference** (`src/inference.py`)
  - `ModelInferer` — self-consistency sampling on top of any retriever (5 samples @ T=0.7, top_p=0.9, majority vote on label, max-confidence sample's evidence list).
  - `ZeroShotInferer` — same shape but greedy decoding for ablation rows A1-A4.
  - `RetrievalOnlyInferer` — no LLM; predicts SUPPORTS (or arbitrary label) and emits retrieved evidences. Lets us measure retrieval F-score in isolation.
  - `predict_all` — batch driver, tqdm-aware progress, writes JSON validated against `eval.py` schema, gracefully degrades to NEI on per-claim failure.
- **Stage 6 ablation harness** (`src/ablation.py`)
  - `AblationConfig` dataclass (declarative pipeline toggles + `flagship` flag).
  - `AblationHarness` — model-agnostic; takes (config, predictions_dict_or_path) pairs; renders main table on official dev + diagnostic slice tables on `diag_test` (domain × 8, scenario × 7, difficulty × 3) + per-label slice on dev.
  - `DEFAULT_CONFIGS` — the nine A1-C2 configurations from Plan §6.1.
  - End-to-end demo confirmed renders all 4 tables from a single `predict()` dict spanning dev + diag_test.

### Tests

| Suite | Cases | Status |
|---|---|---|
| test_prompt | 8 | green |
| test_eval_helpers | 3 | green |
| test_sft_dataset | 3 | green |
| test_fuse | 4 | green |
| test_query_rewrite | 7 | green |
| test_dpo_pairs | 5 | green |
| test_inference | 4 | green |
| test_ablation | 3 | green |
| **total** | **37** | **all green** |

### Code surface

`src/` 14 modules, ~2400 lines. `tests/` 8 suites. Covered modules:
`data_io paths eda tagging splits prompt sft_dataset query_rewrite dpo_pairs eval_helpers retrieval/{bm25,dense,fuse,rerank,pipeline} inference ablation build_stage0`.

### Demo artifact

`outputs/ablation/ablation_report.md` — synthesised from baseline + 70%-correct flagship simulation. Confirms diagnostic tables surface the expected DISPUTED-hardest / supports_clear-easiest pattern.

---

## Session 1 (2026-04-30) — Stage 0 + Stage 1 scaffolding

### What's done

- **Project skeleton**: `src/` (11 modules), `tests/` (6 suites, 30 cases all green), `notebooks/`, `outputs/{eda,splits,sft_data}/`. `.gitignore` excludes evidence.json, checkpoints, embeddings, predictions.
- **Notebook ported to official template** (`notebooks/notebook.ipynb`, 45 cells). The 3 mandatory section headers (`1.DataSet Processing`, `2.Model Implementation`, `3.Testing and Evaluation`) untouched per assignment rule. Sub-sections fill them. OOP section at bottom re-imports key classes for grading visibility.
- **Stage 0 fully runnable locally** (`python -m src.build_stage0`, ~2 s force rebuild):
  - EDA report (key prior: NEI claims always have exactly 5 gold evidences)
  - Three-axis tagging: scenario × climate-domain × difficulty
  - Hash split: train_split 986 / dev_holdout 121 / diag_test 121 / official_dev 154
  - Six pairwise leakage assertions all pass
  - SFT data: train 1972 (with hard-neg ×1) / dev_holdout 121 / diag_test 121 in ms-swift format
- **Stage 1 retrieval scaffolding** (Colab-targeted but interface-tested locally):
  - `bm25.py` — `bm25s` wrapper with on-disk caching
  - `dense.py` — sentence-transformers (`bge-m3` default, `bge-small-en-v1.5` fallback) + FAISS, chunked encoding
  - `fuse.py` — weighted-sum (0.3 BM25 + 0.7 dense) + RRF
  - `rerank.py` — cross-encoder (`bge-reranker-base`) + rule-based reorder (NER boost, near-dup suppress, diversity cap)
  - `pipeline.py` — composable end-to-end with label-conditioned-k toggle
- **Stage 2 query rewriting** (`query_rewrite.py`): WordNet synonym expansion + sub-claim decomposition prompt + HyDE prompt + claim/hypothesis text/embedding blending
- **Stage 4 DPO pair builder** (`dpo_pairs.py`): mines errors from `dev_holdout` (never dev), supports DISPUTED-vs-SUPPORTS contrast augmentation
- **Eval helpers** (`eval_helpers.py`): bit-for-bit match with `eval.py` (verified to 1e-15 on baseline), plus per-bucket slicer + recall@k

### Performance fix

`build_dataset` had O(N×n_claims) blowup: rebuilt 1.2M-id pool per claim during random padding. Replaced with index-cached rejection sampling. **376 s → 0.1 s** for 1972-record build.

### Outputs on disk

```
outputs/
  eda/eda_report.md
  splits/{train_split,dev_holdout,diag_test,official_dev}.jsonl + split_summary.md
  sft_data/claims_tagged.jsonl + tag_distribution.md
  sft_data/sft_{train,dev_holdout,diag_test}_v1.jsonl
```

### Tests

| Suite | Cases | Status |
|---|---|---|
| test_prompt | 8 | green |
| test_eval_helpers | 3 | green (matches eval.py to 1e-15) |
| test_sft_dataset | 3 | green |
| test_fuse | 4 | green |
| test_query_rewrite | 7 | green |
| test_dpo_pairs | 5 | green |
| **total** | **30** | **all green** |

### What's blocked / pending

- `data/evidence.json` ✓ downloaded (174 MB, 1,208,827 passages)
- `notebooks/GroupID__COMP90042_Project_2026.ipynb` ✓ official template at hand
- **Needs Colab T4** (not local):
  - BM25 index build (~2-4 min)
  - bge-m3 full-corpus embedding (~30-60 min, cached to Drive)
  - Qwen3.5-4B download from ModelScope
  - ms-swift SFT 3 epochs (~75-105 min)
  - DPO 1 epoch (~25 min)
  - Inference on dev + test

### Decisions deferred until first Colab run

- Confirm `Qwen/Qwen3.5-4B-Instruct` exists on ModelScope. Fallback: `Qwen/Qwen2.5-VL-3B-Instruct`.
- Confirm ms-swift's `--model_type` slug for Qwen3.5-VL. Fallback: Unsloth.
- Pick final retrieval weights (0.3/0.7 default) by k-sweep on dev.
