# TODO — 续跑指南

> 单页"明天打开就知道做什么"的快速恢复文档。
> 完整计划见 `optimization_plan.md`，本文只列**接下来一步要做什么**。
>
> 最后更新: 2026-05-12 晚（SFT v2/v3 两次都塌缩 → 战略转向 retrieval-first，design D-019）

---

## ✅ 已完成（截至今天）

- [x] AutoDL 实例就绪（PyTorch 2.5.1+cu124, 4080 SUPER 31.5GB VRAM）
- [x] Smoke test on AutoDL 通过（`scripts/test_qwen35_inference.py`）
- [x] Phase 1 scaffolding 全部到位:
  - `src/prompt.py` 加 v1-v4 变体
  - `scripts/build_indexes.py` 独立索引构建
  - `scripts/phase1_eval.py` Track 1/2 × prompt 扫描 harness
  - `scripts/download_models.py` 一键下所有第三方权重到 `models/`
  - `scripts/diagnose_phase1.py` 预测分布 / confusion matrix / NEI-default 检测
- [x] SFT/DPO 数据迁移到 ms-swift messages 标准格式 + 测试 all green
- [x] 持久化策略落地：cell-1-sft-code 改 cache-first，paths 加 `MODELS_DIR + resolve_model_path()`
- [x] 文档：`design.md` v1.1 (D-011~D-015), `debug_log.md` 会话 2+3 (问题 1-17), `optimization_plan.md` (bilingual 6-phase plan)
- [x] **本地 4 个模型权重全部下载完成**（`models/` 下 ~11 GB）
- [x] **本地 BM25 索引建好**（`outputs/bm25_index/bm25/` 5 个文件，~200 MB）→ 本地 retrieval 现在可 dry-run
- [x] **AutoDL dense 索引建好**（`outputs/dense_index/`, 9.2 GB）+ safetensors 转换 + messages 格式 SFT 数据全部就位
- [x] **Phase 1 baseline 评估跑通 + 诊断完成**
  - Track 2 v1: F=0.1169 Acc=0.4215 HM=**0.1830**（生产基线）
  - per-label acc: S 0.526 / R 0.500 / NEI 0.350 / DISPUTED 0.286
  - base 模型 Track 1 NEI acc=0.025 / DISPUTED acc=0.000 → 量化证实 §0.5.2 4a
- [x] **Phase 2 prompt sweep 完成**：v1 锁定。v2/v3/v4 全部回退（v3 REFUTES→0）
- [x] **关键发现：检索天花板 evidence recall ≈ 0.11**（v1-v4 全部一样，与 prompt 无关）
  - F-score 当前架构硬上限 ≈ 0.12，HM 硬上限 ≈ 0.21
- [x] **Phase 3.5 检索审计 + final_k 端到端实测**
  - recall@k 曲线：5→0.119 / 10→0.210 / 20→0.333 / 50→0.485 / 100→0.579
  - 端到端 Track 2 v1 HM：k=5 0.183 / k=10 0.196 / **k=20 0.203**
  - **锁定 `final_k=20`**：`RetrievalConfig.final_k` default 改了 + `phase1_eval --final-k` default 20
  - 副作用：k=20 NEI acc 0.025（base 模型几乎不输出 NEI） — 正好是 Phase 4 weak_buckets 要修的
- [x] **Phase 4 `weak_buckets` 实现 + 测试**
  - `build_dataset(..., weak_buckets={...})` 支持 (axis, bucket) → factor 配比，多匹配取 max
  - `build_stage0` 内置 `{nei_underspec:4, disputed_conflict:2, refutes_clear:2}`
  - 输出 `sft_*_v2.jsonl`（k=20，train 含 weak_buckets 倾斜，v1 保留 ablation）
- [x] **v2 SFT 数据本地构建 + 分布验证**：4166 records (×2.11 vs v1)；
      weak_buckets 完全按预期：nei_underspec ×4 / disputed_conflict ×2 / refutes_clear ×2；
      NEI label 占比 65.4%→**79.1%**，hard difficulty 14.3%→**24.1%**
- [x] **notebook_autodl.ipynb 9 cell 修复**：cache-first 模型加载 + ms-swift v3.6+ CLI
      （`--tuner_type` / `--quant_bits` / thinking trio）+ DATA_PATH→v2 + MAX_LEN 1024→1536
- [x] **清理仓库 `Qwen3___5-4B` 残留 5 处** → 全部统一为 `Qwen3.5-4B`（见 debug_log 复用经验 26）
- [x] **AutoDL GFW workaround 落实**：`source /etc/network_turbo` 加速 github pull
      （debug_log 复用经验 25）
- [x] **env-stack 三连碰全部解决**（debug_log 复用经验 27-28）：
  - FSDP2 ImportError on torch 2.5.1 → `scripts/patch_swift_fsdp2.py` (patch + sentinel + idempotent)
  - transformers `qwen3_5` model_type KeyError → pin `transformers==5.2.*`
  - peft × transformers 5.2 HybridCache 不兼容 → `pip install -U peft` 到 0.17+
- [x] **requirements 拆分**：`requirements.txt`（torch 2.5 默认）+ `requirements-torch26.txt`（未来）
- [x] **nbstripout 加入 deps**（debug_log 复用经验 29）— 解决 JupyterLab autosave vs git pull 死锁
- [x] **SFT v2 训练启动**（2026-05-12 ~04:05 UTC）：
  - 783 steps，step 1 loss 0.1146 → step 35 loss 0.010，grad_norm 健康，VRAM 11.4 GB / 31.5 GB
  - 配置：LoRA r=16，QLoRA 4-bit，BS=2×GA=8，lr 2e-4，max_len 1536，liger_kernel
  - thinking 三件套 + group_by_length + save_total_limit=3
  - 预估 ~4h 完成，checkpoint 落 `nlp_a3_cache/sft-out/v4-20260512-040505/`
- [x] **`scripts/phase1_eval.py` 加 `--sft-adapter` flag**：Track 3 (base+SFT+RAG) 端到端评估

---

## 🎯 明天的下一步（按顺序）

### ✅ Step 1 — AutoDL 环境 / cache 全部就位（已完成）

参考下方"附录：Step 1 完整命令"（保留作环境重建参考）。

### ✅ Step 2 — Phase 1 baseline 跑通（已完成，但数字需诊断）

```bash
python -m scripts.phase1_eval --tracks 1,2 --prompts v1 --dataset diag_test
```

产出：
- `outputs/eval_phase1/track1_v1_diag_test.{json,md}` — Acc=0.3223
- `outputs/eval_phase1/track2_v1_diag_test.{json,md}` — F=0.1169, Acc=0.4215, HM=0.1830
- `outputs/eval_phase1/summary_diag_test.md`

### ✅ Step 2.5 — Phase 1 诊断 (已完成)

`diagnose_phase1.py` 已确认：非 parser fallback；问题是 base 模型完全缺 NEI/DISPUTED 概念，
RAG 部分补救。详见 `outputs/eval_phase1/diagnose_diag_test.md`。

### ✅ Step 3 — Phase 2 prompt sweep (已完成)

v1 锁定。`summary_diag_test.md` 含完整 v1-v4 对比。v2/v3/v4 全部回退。

### ✅ Step 4 — Phase 3.5 检索天花板审计 (已完成)

`scripts.retrieval_ceiling --mode final_k` + 端到端 k=5/10/20 实测。
锁定 `final_k=20`。详见 `outputs/eval_phase1/retrieval_ceiling_diag_test.md`。

### ✅ Step 5a — Phase 4 weak_buckets 实现 (已完成，代码 + tests 全绿)

`build_dataset` 加参数；`build_stage0` 内置 Phase 4 配比；tests pass。

### ✅ Step 5b — 重建 SFT 数据 (本地完成 2026-05-12)

实测：4166 records，weak_buckets 完全按预期 (nei_underspec ×4, disputed ×2,
refutes_clear ×2)。AutoDL 上需要重跑 `python -m src.build_stage0 --force`
确保 train/inference 一致（`outputs/sft_data/*.jsonl` 在 .gitignore 里 → git
不传，AutoDL 必须本地重建或 `scp` 传过去）。

### ✅ Step 6 — SFT v2 第一次训练完成（v4-20260512-040505）

训练 4h 完成，merged via `swift export --merge_lora true`。Track 3 评估**失败**：
HM 0.140 < Track 2 baseline 0.201，predicted NEI 92.6% / non-NEI acc 0.06。
**Class-collapse**：训练数据 79% NEI（n_hard_neg=1 + nei_underspec ×4 双重放大）
让模型学到"看到啥都猜 NEI"。详见 debug_log 复用经验 32 + diagnose_diag_test.md track3_v1 行。

### ✅ Step 6.5 — SFT 数据重平衡（已完成）

`src/build_stage0.py` 改：
- `nei_underspec ×4 → ×2`
- `disputed_conflict ×2 → ×3`
- `n_hard_neg=1 → 0`（关键：去掉 hard-neg 同义重复）

加 `_print_label_dist()` sanity check：每个 split build 完打印 SFT vs gold
ratio + warn >2×/<0.5×（如果再撞 class-collapse 不再盲跑 4h）。

本地重建 v2 已验证：1567 records，NEI 占比 79.1% → 38.7%（gold 33.1%）。

### ❌ Step 7 — SFT v3-rebalanced 也失败（2026-05-12 PM 实测）

v5-20260512-180014/checkpoint-294 → merge_v3 → eval：
- predicted NEI 占比 92.6% → **94.2%**（更差）
- non-NEI acc 0.062 → **0.049**（更差）
- HM **0.140** = v2-cut-1（持平没救）

**class imbalance 不是根因**，根因是 `pad_with_random=True` 让训练分布
跟低 recall RAG 推理高度重合 → 模型学到"evidence 模糊 → NEI"捷径。
看 debug_log 复用经验 34 + design.md D-019。

**战略转向 retrieval-first**：先把 recall@20 从 0.333 拉到 ≥ 0.50，
SFT 才有训练信号。

### 🎯 Step 8 — Phase 3.5b 检索深度审计（**当前最优先**）

```bash
source /etc/network_turbo
cd ~/autodl-tmp/NLP-A3

# 跑剩下 3 个 mode（~45-50 min on 4080 SUPER）
python -m scripts.retrieval_ceiling --dataset diag_test \
    --mode retriever,fusion_w,synonym_expand 2>&1 | tee outputs/retrieval_audit_full.log

# 看产出
cat outputs/eval_phase1/retrieval_ceiling_diag_test.md
```

读 "🏆 Best overall" 段 + recall@k 曲线。判定：

| 情景 | 含义 | 行动 |
|---|---|---|
| 某配置 recall@20 ≥ 0.50 | retriever / fusion / synonym 找到了 | Step 9a：锁新 RetrievalConfig 重 build SFT 重训 |
| recall@20 仍 < 0.40 | 检索器架构天花板 | Step 9b：上 LLM-rewrite (HyDE/sub-claim) |
| 0.40-0.50 中间 | 部分提升 | 同时跑 Step 9a + 9b |

### 🎯 Step 9a — 锁新检索配置 + 重训 SFT v4（最快路径，~3h）

```bash
# 改 src/retrieval/pipeline.py RetrievalConfig 默认值 → 锁定 Step 8 最佳
# 改 src/build_stage0.py 考虑 pad_with_random=False（复用经验 34 lesson）
python -m src.build_stage0 --force
python -m scripts.run_sft 2>&1 | tee outputs/sft_v4_train.log
# 训完 swift export + phase1_eval --tracks 2,3
```

### 🎯 Step 9b — LLM query rewrite (HyDE / sub-claim)（半天工作量）

如果 Step 8 找不到能拉 recall@20 ≥ 0.50 的纯检索配置：

1. 写 `scripts/rewrite_queries.py`:
   - Load base Qwen3.5-4B
   - 对每条 claim 跑 HyDE prompt + sub-claim decomposition
   - 缓存到 `outputs/query_rewrite/claim_rewrites.jsonl`
2. `RetrievalPipeline` 加 multi-query 接口（每条 claim 用 (claim, hyde_ev,
   sub_claim_1, sub_claim_2) 各检索一次 RRF 融合）
3. 重跑 retrieval audit 验证 recall 提升
4. 锁配置后回到 Step 9a 重训 SFT

### 📋 Deferred — 推理 RAM 优化（Colab 兼容性）

不阻塞主线，记账。当且仅当从 AutoDL（31.5 GB RAM）切回 Colab Free
（12.7 GB RAM）才需要。主要 lever：`faiss.IO_FLAG_MMAP`（−3 GB RAM, 0
精度损失）。详细审计 + 4 个优化选项 + 测试方式见 `optimization_plan.md`
§8.1。

### ✅ Step 10 — Track 3 评估（v3 SFT 后）— 失败留档

实际跑出来 HM 0.140 < Track 2 baseline，已记 ❌ Step 7。下面命令保留作 Track 3 v4 评估模板。

```bash
# AutoDL 上
source /etc/network_turbo
cd ~/autodl-tmp/NLP-A3

# 1. 确认 checkpoint 存在
ls -la /root/autodl-tmp/nlp_a3_cache/sft-out/v4-20260512-040505/

# 2. 软链到稳定路径（方便引用）
ln -sf /root/autodl-tmp/nlp_a3_cache/sft-out/v4-20260512-040505/checkpoint-final \
       /root/autodl-tmp/nlp_a3_cache/sft-out/checkpoint-final

# 3. Track 2 vs 3 head-to-head（~10 min）
python -m scripts.phase1_eval --tracks 2,3 --prompts v1 --dataset diag_test \
    --sft-adapter /root/autodl-tmp/nlp_a3_cache/sft-out/checkpoint-final

# 4. 诊断（< 5s）
python -m scripts.diagnose_phase1 --dataset diag_test
```

**关键判定指标**（对比 Track 2 v1 baseline HM 0.203，per-label acc S 0.737 / R 0.682 / NEI 0.025 / D 0.190）：

| 指标 | Track 2 v1 (base+RAG) | Track 3 (期望) | 达成判定 |
|---|---|---|---|
| **NEI acc** | 0.025 | **≥ 0.40** | 硬约束 §0.5.3.1 兑现 |
| non-NEI acc | 0.580 | 不跌 5pp 以上 | SFT 没 over-correct |
| 总 HM | 0.203 | **≥ 0.28** | Phase 4 红利至少 +0.08 |
| Δ HM (Track 3 − Track 2) | — | ≥ +0.05 | SFT 端到端有效 |

读 `outputs/eval_phase1/diagnose_diag_test.md` cross-run summary 表 +
Track 3 confusion matrix。

### Step 8 — DPO 训练（Track 4 准备）

`src/dpo_pairs.py` 已实现 `synthesise_disputed_contrast`，但还没写 driver
脚本。等 Track 3 确认 SFT 有效后启动 —— 大概路径：

```bash
# 1. 从 dev_holdout 上 SFT 模型的错预测挖 chosen/rejected 对
python -m src.dpo_pairs  # （此 entry 待写）

# 2. swift rlhf --rlhf_type dpo （cell-2-6-code 已就绪）
```

### Step 9 — 4-track 完整对比 + 锁定 production

```bash
python -m scripts.phase1_eval --tracks 1,2,3,4 --prompts v1 --dataset diag_test \
    --sft-adapter ... --dpo-adapter ...    # --dpo-adapter 待加
```

```bash
# ms-swift CLI（参考 notebook cell-2-sft-train，已在 debug_log Issue 9-12 调通）
# Canonical 模型路径 = models/Qwen3.5-4B/（scripts.download_models 的产物，
# 单个点不是三下划线 — 见 debug_log 复用经验 26）。
swift sft \
    --model models/Qwen3.5-4B \
    --dataset outputs/sft_data/sft_train_v2.jsonl \
    --tuner_type lora --quant_bits 4 \
    --enable_thinking false --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --bf16 true --use_liger_kernel true \
    --group_by_length true --save_total_limit 3 \
    --output_dir outputs/sft-out
```

训练完拿到 LoRA adapter 后跑端到端评估：

```bash
# 需要给 phase1_eval 加 --sft-adapter flag（下一轮再写）
python -m scripts.phase1_eval --tracks 2 --prompts v1 --dataset diag_test \
    --sft-adapter outputs/sft-out/checkpoint-final
python -m scripts.diagnose_phase1 --dataset diag_test
```

读 confusion matrix 比对 pre-SFT vs post-SFT：
- **目标 1**：NEI acc 0.025 → ≥ 0.40（硬约束 1 兑现）
- **目标 2**：non-NEI acc 0.580 不掉 5pp 以上
- **目标 3**：总 HM 从 0.203 → ≥ 0.28

详见 `optimization_plan.md` §5。

---

## 📦 附录：Step 1 完整命令（环境重建时用）

`AutoDL` 实例丢失或换机时整套重跑。前提：`data/evidence.json` 已传到 `~/autodl-tmp/NLP-A3/data/`。

```bash
cd ~/autodl-tmp/NLP-A3
git pull origin main

pip install -U modelscope huggingface_hub  # 保险

# 重生成 messages 格式 SFT 数据（~5 s）
python -m src.build_stage0

# 一键下所有模型到 models/（~11 GB）
python -m scripts.download_models

# bge-* 系列只有 .bin；transformers + torch 2.5 不让 torch.load → 本地转 safetensors
python -m scripts.convert_bin_to_safetensors

# 建索引（BM25 ~3 min + dense ~15 min on 4080 SUPER）
python -m scripts.build_indexes
```

---

## 🚧 阻塞 / 需要决策的事

1. **本地 vs AutoDL 边界已定**（见 `optimization_plan.md` §1.2 + 这次 chat）：
   - 本地：BM25 build + 代码 dev/debug + dry_run
   - AutoDL：dense build + 所有 inference + 训练
   - 不在本地跑 inference 因为 Windows 上 bitsandbytes 不稳 + 6GB VRAM 装不下 fp16 4B 模型

2. **Phase 1 跑完后的"弱桶 → SFT 数据配比"映射**（Phase 4）尚未写代码：
   - 需在 `src/sft_dataset.py:build_dataset` 加 `weak_buckets` 参数
   - 见 `optimization_plan.md` §4.4 的伪代码
   - **Phase 1 跑完拿到诊断切片后再设计**，现在不写

---

## 📁 关键文件速查

| 想看什么 | 去哪 |
|---|---|
| 整体计划 / 6 阶段细节 | `optimization_plan.md` |
| 系统架构 + 决策记录 | `design.md`（v1.1, D-001~D-015）|
| 历史问题排查 | `debug_log.md`（含会话 2 的 Qwen3.5/AutoDL 全部坑）|
| AutoDL Quick Start | `requirements.txt` 顶部注释块 |
| 当前 prompt 变体定义 | `src/prompt.py` 的 `PROMPT_VARIANTS` dict |
| Phase 1 评估入口 | `python -m scripts.phase1_eval --help` |

## 🔑 关键约束 / 不能忘

1. **`official_dev` 的 154 条只能看 ≤ 3-4 次**（design.md D-006）—— Phase 1-5 全部用 `diag_test`，只有 Phase 6 才碰 official dev
2. **Qwen3.5-4B 是 mixed-thinking VL 模型**，不是 text-only base —— 推理/训练都要 `enable_thinking=False` + 思考三件套
3. **transformers 5.x 的 `apply_chat_template` 返回 `BatchEncoding` 不是 tensor** —— 已用 helper 兜底但新写代码时记得
4. **T4 不支持 bf16 / flash-attn 2.x**；AutoDL 4080 支持 → SFT/DPO CLI 用 `--bf16 true`，T4 时切 `--fp16 true`

---

**下次 session 第一句话**：AutoDL 上 `source /etc/network_turbo && git pull && python -m src.build_stage0 --force && python -m scripts.run_sft 2>&1 | tee outputs/sft_v6_train.log`。重训 SFT v6（`pad_with_random=False`，~1.5h），训完 `swift export --merge_lora` 拿 merged_v6 → `phase1_eval --tracks 2,3 --sft-merged-dir <merged_v6>` → `diagnose_phase1`。看 Track 3 是否 HM > 0.213 (Track 2 baseline)；是 → 进 DPO；否 → 接受 Track 2 作 final report。
