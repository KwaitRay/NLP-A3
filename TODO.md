# TODO — 续跑指南

> 单页"明天打开就知道做什么"的快速恢复文档。
> 完整计划见 `optimization_plan.md`，本文只列**接下来一步要做什么**。
>
> 最后更新: 2026-05-11 晚

---

## ✅ 已完成（截至今天）

- [x] AutoDL 实例就绪（PyTorch 2.5.1+cu124, 4080 SUPER 31.5GB VRAM）
- [x] Smoke test on AutoDL 通过（`scripts/test_qwen35_inference.py`）
- [x] Phase 1 scaffolding 全部到位:
  - `src/prompt.py` 加 v1-v4 变体
  - `scripts/build_indexes.py` 独立索引构建
  - `scripts/phase1_eval.py` Track 1/2 × prompt 扫描 harness
  - `scripts/download_models.py` 一键下所有第三方权重到 `models/`
- [x] SFT/DPO 数据迁移到 ms-swift messages 标准格式 + 测试 all green
- [x] 持久化策略落地：cell-1-sft-code 改 cache-first，paths 加 `MODELS_DIR + resolve_model_path()`
- [x] 文档：`design.md` v1.1 (D-011~D-015), `debug_log.md` 会话 2, `optimization_plan.md` (bilingual 6-phase plan)
- [x] **本地 4 个模型权重全部下载完成**（`models/` 下 ~11 GB）
- [x] Push 到 `origin/main`（最新 commit `003122a`）

---

## 🎯 明天的下一步（按顺序）

### Step 1 — 本地：建 BM25 索引（5 分钟，零风险）

```powershell
python -m scripts.build_indexes --skip-dense
```

产出 `outputs/bm25_index/` (~200 MB)。这样本地有最小可用 retrieval，之后改代码可本地 dry-run。

### Step 2 — AutoDL：拉新代码 + 同步模型 + 建 dense 索引

> **为什么 dense 不在本地建**：bge-m3 fp16 要 ~5 GB VRAM，你本地 6 GB 太紧；编码 1.2M 段在 CPU 上慢到不可接受。AutoDL 4080 SUPER 上 ~15 min 完事。

```bash
# 在 AutoDL 上
cd ~/autodl-tmp/NLP-A3
git pull origin main

# 装新依赖（modelscope 已装；huggingface_hub 应该 transformers 自带）
pip install -U modelscope huggingface_hub  # 保险起见

# 一键下所有模型到 models/（如果之前 outputs/model_cache 下过 Qwen，--skip qwen）
python -m scripts.download_models

# 建索引（BM25 + dense 全套，~20 min）
python -m scripts.build_indexes
```

### Step 3 — AutoDL：Phase 1 baseline（v1 prompt, ~10 min）

```bash
python -m scripts.phase1_eval \
    --tracks 1,2 --prompts v1 --dataset diag_test
```

产出：
- `outputs/eval_phase1/track1_v1_diag_test.{json,md}`
- `outputs/eval_phase1/track2_v1_diag_test.{json,md}`
- `outputs/eval_phase1/summary_diag_test.md`

**关键看 `track2_v1_diag_test.md`** —— per-bucket 表已按 HM 升序，**最差的桶在最上面**，那就是 Phase 4 SFT 数据扩充的目标。

### Step 4 — AutoDL：Phase 2 prompt 扫描（v2/v3/v4, ~15 min）

```bash
python -m scripts.phase1_eval \
    --tracks 2 --prompts v2,v3,v4 --dataset diag_test
```

`summary_diag_test.md` 会被覆盖成包含 v1-v4 的对比，看哪个 prompt 在 Track 2 上 HM 最高 → 锁定。

### Step 5 — 把决策填到 `optimization_plan.md` §10

在决策日志表追加一行：
```markdown
| 2026-05-12 | Phase 1 done | Track 1 HM=?, Track 2 HM=?, 最弱 3 桶: ?? | track2_v1_diag_test.md |
| 2026-05-12 | Phase 2 done | 锁定 prompt: v?  Track 2 HM 提升 +? | summary_diag_test.md |
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

**下次 session 第一句话**：跑 Step 1 → Step 3，把 `summary_diag_test.md` 和 `track2_v1_diag_test.md` per-bucket 表贴回来。
