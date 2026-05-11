# TODO — 续跑指南

> 单页"明天打开就知道做什么"的快速恢复文档。
> 完整计划见 `optimization_plan.md`，本文只列**接下来一步要做什么**。
>
> 最后更新: 2026-05-11 深夜（Phase 1 跑通，数字异常待诊断）

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
- [x] **Phase 1 baseline 第一次评估跑通**（Track 1+2 × v1 on diag_test，~4.5 min on 4080 SUPER）
  - Track 1 (no-RAG, greedy):  F=0.0000  Acc=0.3223  HM=0.0000
  - Track 2 (RAG, greedy):     F=0.1169  Acc=0.4215  HM=0.1830
  - ⚠️ Track 1 Acc=0.3223 ≈ NEI 占比 0.3306 → 触发诊断（见下方 Step 2.5）

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

### ⚠️ Step 2.5 — 诊断 Track 1 Acc ≈ NEI 占比 的异常（**新增，明天第一步**）

Track 1 Acc=0.3223 = 39/121 离"全猜 NEI 多数类"的 40/121 只差 1 条。
两种可能（见 `debug_log.md` 问题 17）：

- (a) Parser fallback 全走 NEI → 改 prompt v2 / parser
- (b) 模型真在判别只是偏弱 → 进 Phase 2 / Phase 4

**先跑诊断脚本**（< 5 s，纯分析 saved JSON，无 GPU）：

```bash
python -m scripts.diagnose_phase1 --dataset diag_test
```

产出：`outputs/eval_phase1/diagnose_diag_test.md`。看：
- **非-NEI acc** 列：接近 0 → (a) 确认；显著大于 0 → (b)
- **predicted NEI 占比**：> 50% 且自动 flag ⚠️ → (a) 确认
- **confusion matrix**：对角线有信号则 (b)；列塌缩到 NEI 那一列则 (a)

**走向 (a) 时**：跑 `scripts.test_qwen35_inference` 重生成几条 diag_test 实例，打印 raw output 看具体输出。然后改 prompt v2 或 `parse_response` 容错。
**走向 (b) 时**：直接进 Step 3 跑 Phase 2 prompt sweep。

### Step 3 — AutoDL：Phase 2 prompt 扫描（v2/v3/v4, ~15 min）

> **前置条件**：Step 2.5 诊断已确定走 (b) 路径（模型真在判别），或走 (a)
> 修复完成。否则 v2/v3/v4 在 broken pipeline 上跑没意义。

```bash
python -m scripts.phase1_eval \
    --tracks 2 --prompts v2,v3,v4 --dataset diag_test
```

`summary_diag_test.md` 会被覆盖成包含 v1-v4 的对比，看哪个 prompt 在 Track 2 上 HM 最高 → 锁定。
**Track 2 v1 HM=0.1830 是新基线**，v2/v3/v4 必须显著高才有意义。

也可以顺手把 v2/v3/v4 也跑一遍 Track 1，看 Step 2.5 诊断的 (a) 假设是否能被 prompt v2 直接修掉：
```bash
python -m scripts.phase1_eval --tracks 1 --prompts v2,v3,v4 --dataset diag_test
python -m scripts.diagnose_phase1 --dataset diag_test  # 自动覆盖所有新 run
```

### Step 4 — 把决策填到 `optimization_plan.md` §10

在决策日志表追加：
```markdown
| 2026-05-12 | Phase 1 diagnosed | (a) parser fallback 或 (b) 模型偏弱 — 实际为 ?? | diagnose_diag_test.md |
| 2026-05-12 | Phase 2 done | 锁定 prompt: v?  Track 2 HM 从 0.1830 → ? | summary_diag_test.md |
```

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

**下次 session 第一句话**：在 AutoDL 上跑 `python -m scripts.diagnose_phase1 --dataset diag_test`，把 `outputs/eval_phase1/diagnose_diag_test.md` 的 cross-run summary 表 + Track 1 confusion matrix 贴回来。
