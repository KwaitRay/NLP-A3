# AutoDL → Colab 迁移清单 / Migration Inventory

> 目的 / Purpose：把 AutoDL 上跑出来的「贵」产物（训练好的 SFT/DPO、构建好的索引、下载好的权重）打包搬到 Colab，避免重复跑。
> 配套阅读 / See also：`DEPLOY.md`（部署 runbook）、`BENCHMARK_SUBMISSION.md`（提交流程）、`optimization_plan.md` §7.2-7.3（artifact 表 + 提交边界）。
>
> 路径约定 / Path convention（来自 `notebooks/notebook_autodl.ipynb` cell [4] 与 `src/paths.py`）：
> - **AutoDL**：`PROJECT_ROOT = /root/autodl-tmp/Assignment3`、`CACHE_ROOT = /root/autodl-tmp/nlp_a3_cache`
> - **Colab** ：`PROJECT_ROOT = /content/drive/MyDrive/NLP_Assignment3/Assignment3`、`CACHE_ROOT = $PROJECT_ROOT/outputs`
> - 「训练 checkpoint」在 AutoDL 落到 `CACHE_ROOT/...`；在 Colab 落到 `outputs/...`。脚本走 `os.environ['CACHE_ROOT']` 抽象，**两边路径不同但 cell 代码一致**。

---

## 0. TL;DR

按"必须 → 可选 → 别传"三档分类。最小可推理迁移包 ≈ **6 GB**（SFT-merged base + 模型权重之一）；含 dense 索引的全量包 ≈ **20 GB**。

| 档 | 内容 | 不传会怎样 |
|---|---|---|
| **Tier 1 必须** | SFT-merged base / SFT LoRA / DPO LoRA | Colab 必须重训，~2 小时 GPU |
| **Tier 2 强烈建议** | FAISS dense 索引 + 模型权重 | Colab 重建 30-60 min + 重下 11 GB |
| **Tier 3 可选** | BM25 索引、bge-reranker | BM25 重建 2-4 min；reranker 默认关闭 |
| **Tier 4 别传** | evidence.json / splits / sft_data / 代码 | 直接 git pull + 公网下载即可 |

---

## 1. Tier 1 — 必须迁移 / Must transfer

**没有这些就得重训，每个都要 GPU。**
**Without these, retrain on Colab — each needs GPU.**

| Artifact | AutoDL 路径 / Source | Colab 落点 / Target | 估算大小 / Size | 备注 |
|---|---|---|---|---|
| **SFT-merged base** | `$CACHE_ROOT/sft-merged/` | `outputs/sft-merged/` | ~8 GB | **首选**。是 `swift export --merge_lora true` 把 LoRA 合进 Qwen3.5-4B 的产物，加载时当普通 base model 用，避免 ms-swift 的 LoRA attach 失败问题（debug_log 复用经验 31）。`run_inference.py --sft-merged-dir` 直接吃这个目录。 |
| SFT LoRA checkpoint | `$CACHE_ROOT/sft-out/checkpoint-final/` | `outputs/sft-out/checkpoint-final/` | ~100-500 MB | 备份。如果 sft-merged 被覆盖了可以重新 export；单独加载会出现 LoRA params=0 的问题（同上）。 |
| DPO LoRA checkpoint | `$CACHE_ROOT/dpo-out/checkpoint-final/` | `outputs/dpo-out/checkpoint-final/` | ~100-500 MB | 只有跑了 DPO 才有；可选加载增益。 |
| 训练日志 / training logs | `$CACHE_ROOT/sft-out/logging.jsonl`、`runs/` | `outputs/sft-out/logging.jsonl` 等 | < 10 MB | 报告里写 loss 曲线要用，丢了没法补。 |

> 小心 / Watch out：`checkpoint-final` 是 ms-swift 训完最后保存的 symlink，要 `tar --dereference` 或 `rsync -L` 跟链接，不然解出来是空目录。
> `checkpoint-final` is a symlink — use `tar --dereference` or `rsync -L`, otherwise the unpacked dir is empty.

---

## 2. Tier 2 — 强烈建议 / Strongly recommended

**Colab 也能重建，但贵。这些是迁移收益最大的项。**
**Re-buildable on Colab but expensive — biggest bang-for-byte to migrate.**

| Artifact | AutoDL 路径 | Colab 落点 | 估算大小 | 不传的代价 |
|---|---|---|---|---|
| **FAISS dense 索引** | `$PROJECT_ROOT/outputs/dense_index/` | `outputs/dense_index/` | ~5 GB | Colab 重建 30-60 min（bge-m3 全语料 embed），还得有 GPU |
| **bge-m3 权重** | `$PROJECT_ROOT/models/bge-m3/` | `models/bge-m3/` | 4.3 GB | Colab 重下 ~5 min，但占 Drive 配额 + 占用网络 |
| **Qwen3.5-4B 权重** | `$PROJECT_ROOT/models/Qwen3.5-4B/` | `models/Qwen3.5-4B/` | 8.8 GB | Colab 重下 ~10 min；ModelScope 有时被墙 |
| **bge-small-en-v1.5** | `$PROJECT_ROOT/models/bge-small-en-v1.5/` | `models/bge-small-en-v1.5/` | 383 MB | 仅作 dense 备援；可选 |

> 替代方案 / Alternative：模型权重也可以在 Colab 上直接 `python -m scripts.download_models` 重下（公网带宽足够 + 有 HF 镜像）。如果 Drive 配额紧张，**只迁 dense_index + sft-merged 是最划算的组合**。
> Models can be re-downloaded on Colab via `scripts.download_models`. If Drive quota is tight, **migrating just `dense_index/` + `sft-merged/` is the sweet spot.**

---

## 3. Tier 3 — 可选 / Optional

| Artifact | AutoDL 路径 | Colab 落点 | 大小 | 何时迁 |
|---|---|---|---|---|
| BM25 索引 | `$PROJECT_ROOT/outputs/bm25_index/` | `outputs/bm25_index/` | ~200 MB | 如果想省 2-4 min 的重建时间。BM25 是纯 CPU 操作，重建很便宜，**通常不值得迁** |
| bge-reranker-base 权重 | `$PROJECT_ROOT/models/bge-reranker-base/` | `models/bge-reranker-base/` | 3.2 GB | 默认关闭（Phase 3.5b 审计：在气候领域伤 recall@5 ×1.68，debug_log 复用经验 35）；只在做 reranker 消融时迁 |
| 评估报告 / eval reports | `$PROJECT_ROOT/outputs/eval_phase1/*.{json,md}` | `outputs/eval_phase1/` | < 5 MB | 想保留 ablation table 就一起带走 |
| ablation 报告 | `$PROJECT_ROOT/outputs/ablation/`、`outputs/dry_run_report.md` | 同名 | < 1 MB | 写报告要引用历史结果时带上 |
| 评估预测 / predictions | `$PROJECT_ROOT/outputs/predictions/*.json` | `outputs/predictions/` | < 5 MB | 历史 run 的预测；只有要复现旧 leaderboard 提交时才需要 |
| **submission ledger** | `$PROJECT_ROOT/outputs/submissions/ledger.jsonl` | `outputs/submissions/ledger.jsonl` | < 100 KB | **如果在 AutoDL 上已经提交过 Codabench**，必须迁，否则 Colab 上的配额计数器从 0 重新算，可能误算超额 |
| evidence-id 缓存 | `$PROJECT_ROOT/outputs/submissions/.evidence_ids.txt` | 同名 | ~17 MB | `build_submission` 用来快校验 evidence 真实性；不传第一次跑时会自动重建（耗 ~30 s 加载 evidence.json） |

---

## 4. Tier 4 — 别迁 / Don't migrate

这些**通过 git / 公网下载更便宜**：

- `data/train-claims.json`、`data/dev-claims.json`、`data/test-claims-unlabelled.json`、`data/dev-claims-baseline.json` — 已在仓库
- `data/evidence.json`（174 MB）— 重新从 Google Drive / Canvas 下载，比从 AutoDL 中转更快
- `outputs/splits/`、`outputs/sft_data/`、`outputs/eda/` — Stage 0 产物，确定性，**任何一台机器跑 `python -m scripts.dry_run` 都得到完全一致的输出**（per `src/splits.py` hash 切分）
- `src/`、`scripts/`、`tests/`、`notebooks/`、`requirements.txt` 等代码 — `git pull` 一秒搞定
- `outputs/model_cache/`（如果有）— 是 ModelScope 的下载缓存，重下即可

---

## 5. 打包流程 / Packaging on AutoDL

### 5.1 推荐方案：tar + 校验

```bash
# 在 AutoDL 上执行 / Run on AutoDL
cd /root/autodl-tmp

# Tier 1（必须）— 训练产物，用 --dereference 解 checkpoint-final 软链接
tar --dereference -czvf sft_artifacts.tar.gz \
    nlp_a3_cache/sft-merged \
    nlp_a3_cache/sft-out/checkpoint-final \
    nlp_a3_cache/sft-out/logging.jsonl \
    nlp_a3_cache/dpo-out/checkpoint-final 2>/dev/null || true

# Tier 2（强烈建议）— 索引 + 权重，已经是 safetensors，不再压缩
tar -cvf dense_index.tar  Assignment3/outputs/dense_index/
tar -cvf models_qwen.tar  Assignment3/models/Qwen3.5-4B/
tar -cvf models_bge.tar   Assignment3/models/bge-m3/

# Tier 3（按需）
tar -czvf bm25_index.tar.gz Assignment3/outputs/bm25_index/
tar -czvf eval_history.tar.gz \
    Assignment3/outputs/eval_phase1/ \
    Assignment3/outputs/ablation/ \
    Assignment3/outputs/dry_run_report.md \
    Assignment3/outputs/PROGRESS.md

# 校验 / verify — 每个包记一行 sha256
sha256sum *.tar* > MIGRATION_MANIFEST.txt
ls -lh *.tar* MIGRATION_MANIFEST.txt
```

### 5.2 传输方式 / Transfer

按 AutoDL 的能力选一个：

| 方案 | 适用 | 命令骨架 |
|---|---|---|
| **AutoDL 公网带宽**（推荐） | AutoDL → 本地 → Drive 上传 | AutoDL 实例端 `python -m http.server` + 本地 `wget` |
| **scp 直传** | 本地能 ssh AutoDL | `scp -r root@<host>:/root/autodl-tmp/*.tar* ./migration/` |
| **AutoDL 自家网盘** | 实例间共享 | 直接拖到 `autodl-pub` |
| **rclone + Drive** | 大文件直传 Drive，跳过本地中转 | `rclone copy *.tar* gdrive:NLP_Assignment3/migration/` |

> ⚠️ 不要走 git/lfs：git LFS 5 GB 配额不够，且 push 到 GitHub 会触发 secret scan / size limit。
> Don't use git LFS — quota is 5 GB, and large pushes hit GitHub's secret-scan + size-limit.

---

## 6. Colab 加载流程 / Loading on Colab

### 6.1 解包到正确路径

```bash
# 在 Colab 上执行 / Run on Colab
PROJECT=/content/drive/MyDrive/NLP_Assignment3/Assignment3

# 把 AutoDL 的 nlp_a3_cache/X 映射到 Colab 的 outputs/X
tar -xzvf sft_artifacts.tar.gz -C /tmp
mkdir -p $PROJECT/outputs/sft-merged $PROJECT/outputs/sft-out $PROJECT/outputs/dpo-out
mv /tmp/nlp_a3_cache/sft-merged/*       $PROJECT/outputs/sft-merged/
mv /tmp/nlp_a3_cache/sft-out/*          $PROJECT/outputs/sft-out/
mv /tmp/nlp_a3_cache/dpo-out/*          $PROJECT/outputs/dpo-out/

# 索引 + 模型直接对应 outputs/ 和 models/
tar -xvf dense_index.tar -C $PROJECT/    # 已经带 outputs/dense_index 前缀
tar -xvf models_qwen.tar -C $PROJECT/    # 已经带 models/Qwen3.5-4B 前缀
tar -xvf models_bge.tar  -C $PROJECT/
```

### 6.2 用 cache audit cell 校验

打开 `notebooks/notebook_autodl.ipynb`，跑 setup cells 让 `PROJECT_ROOT` / `CACHE_ROOT` 进 env，然后跑 **cell [9]（Cache audit）**。预期输出 11/12 OK（缺的只有 `evidence-id cache`，第一次跑 `build_submission` 时会自动建）。

Open `notebook_autodl.ipynb` → run setup cells → run **cell [9] (Cache audit)**. Expect 11/12 OK; the only missing one (`evidence-id cache`) auto-builds on first `build_submission`.

### 6.3 manifest 校验 / sha256 verify

```bash
sha256sum -c MIGRATION_MANIFEST.txt   # 全部 OK 才算迁移完成
```

---

## 7. Smoke test：迁完先跑这一句

```bash
# 不烧 LLM、不消耗 Codabench 配额，只验证索引 + 模型路径都对
python -m scripts.run_inference \
    --target diag_test --tag migration_smoke \
    --decoding retrieval-only --limit 10
```

**期望结果**：3-5 秒内写出 `outputs/predictions/migration_smoke__diag_test.json`，10 条预测每条带 `claim_text` + `≤5 evidence-* IDs`。如果报 `BM25 index missing` 或 `dense index missing`，说明 §6.1 的解包路径错了。

**Expected**: writes `outputs/predictions/migration_smoke__diag_test.json` in 3-5 s with 10 valid predictions. Index errors here mean the §6.1 unpack paths are wrong.

---

## 8. 大小预估 / Size budget

| 迁移档位 | 总大小 | 适用场景 |
|---|---|---|
| **最小推理包** = SFT-merged + Qwen3.5-4B | ~17 GB | Colab Drive 紧张，索引在 Colab 重建 |
| **推荐包** = SFT-merged + Qwen3.5-4B + bge-m3 + dense_index | ~22 GB | Drive 25 GB 内能塞下，开机即可推理 |
| **全量包** = 上面 + DPO + BM25 + reranker + 评估报告 | ~26 GB | Drive 100 GB 计划；做 reranker 消融时用 |

> Colab Free 的 Drive 配额 **15 GB**，跑全量包不够；Drive Pro 是 100 GB / 2 TB。**最小包 17 GB 已经超 Free 上限**，所以 Colab Free 用户必须考虑：（a）Drive Pro，或（b）权重在 Colab 重下 + 只迁 dense_index + sft-merged + ledger。
> Colab Free Drive quota is 15 GB — even the minimal package (17 GB) doesn't fit. Free users must either upgrade or re-download model weights on Colab and only migrate dense_index + sft-merged + ledger.

---

## 9. 检查清单 / Final checklist

迁移完成后，下列项必须全绿：

- [ ] AutoDL 端 `MIGRATION_MANIFEST.txt` 与 Colab 端解包后 `sha256sum -c` 全部一致
- [ ] Colab 端 cell [9] cache audit ≥ 10/12 OK（FAISS + SFT-merged 必须 OK）
- [ ] §7 的 retrieval-only smoke 跑通，输出 10 条合格预测
- [ ] `outputs/submissions/ledger.jsonl` 已迁（如有过历史提交），不然 phase 配额会算错
- [ ] `git pull` 拉到最新代码（`scripts/run_inference.py`、`scripts/build_submission.py`、`BENCHMARK_SUBMISSION.md` 都是 2026-05-15 加的）
