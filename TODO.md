# TODO — 续跑指南

> 单页"明天打开就知道做什么"的快速恢复文档。
> 完整计划见 `optimization_plan.md`，本文只列**接下来一步要做什么**。
>
> 最后更新: 2026-05-12（Phase 1+2 done, 检索天花板发现 → Phase 3.5 优化）

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
  - **SFT 之前必须先做 Phase 3.5 检索优化**（见下方 Step 4）

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

### 🎯 Step 4 — Phase 3.5 检索天花板审计 (**明天第一步**)

> **为什么先做这个**：evidence recall ≈ 0.11 → F-score 当前架构硬上限 ≈ 0.12。
> Phase 4 SFT 哪怕把 label 提到 100% 正确，HM 也只能到 ≈ 0.21。先把检索拉起来，
> SFT 红利才能兑现。详见 `optimization_plan.md` §3.5。

```bash
# AutoDL 上 — 全模式扫描，~3 min（纯检索，无 LLM）
python -m scripts.retrieval_ceiling --dataset diag_test --mode all

# 看产出
cat outputs/eval_phase1/retrieval_ceiling_diag_test.md
```

四个 mode 会跑：
- **final_k**：5→10→20→50→100 看 recall 曲线（最可能的快速胜：现在 `final_k=5` 太紧）
- **retriever**：BM25-only / dense-only / fused / +rerank 看哪个组件贡献最大
- **fusion_w**：w_bm25 ∈ {0.1, 0.3, 0.5, 0.7, 0.9}（当前 0.3 偏 dense）
- **synonym_expand**：claim vs claim + WordNet 同义词 multi-query union

读 `retrieval_ceiling_diag_test.md` 的 "Best Overall" 段，把最佳配置写回
`optimization_plan.md` §10 决策日志 + `RetrievalConfig` 调用处。

### Step 5 — 用新检索配置重建 SFT 数据 + 跑 Phase 4

```bash
# 备份当前 v1 数据
cp -r outputs/sft_data outputs/sft_data.v1_backup

# 在 src/retrieval/pipeline.py 改默认 RetrievalConfig，或者
# 在 src/sft_dataset.py 用 RetrievalConfig 重建
python -m src.build_stage0 --force  # 重生成 sft_train_v2.jsonl
```

之后再走 Phase 4 弱桶配比（见 `optimization_plan.md` §4）。

### Step 6 — Phase 5 训练 + 评估（不变）

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

**下次 session 第一句话**：在 AutoDL 上跑 `python -m scripts.retrieval_ceiling --dataset diag_test --mode all`，把 `outputs/eval_phase1/retrieval_ceiling_diag_test.md` 的 "Best Overall" + recall@k 曲线表贴回来。
