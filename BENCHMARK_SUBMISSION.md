# Codabench Benchmark 提交要求分析与流程实现 / Submission Requirements Analysis & Process Implementation

> 来源 / Sources
> - `materials/scorebench/bench平台要求.docx` — Codabench leaderboard 规则正文 / leaderboard rules
> - `materials/peer_review_instruction.pdf` & `peer-review-scoring.pdf` — Assignment 3 Part 1/2 同行评审规则（与 leaderboard **互不相关**，本文不展开）/ peer-review rules, **orthogonal** to the leaderboard
> - `eval.py`、`src/data_io.py`、`src/inference.py`、`data/test-claims-unlabelled.json` — 当前项目代码与数据 / current code & data
>
> 配套阅读 / See also: `DEPLOY.md` Phase 9（提交打包）、`optimization_plan.md` §7.3（提交边界）

---

## 0. TL;DR

- **平台**：Codabench leaderboard，2026 学期可选项（不计入最终成绩，仅作公共对比）。
  **Platform**: Codabench leaderboard for COMP90042 2026 — *optional*, does not affect final mark, used purely as a public benchmark.
- **关键交付物**：一个 `.zip`，里面**只放**一个 UTF-8 JSON 文件 `test-output.json`，覆盖 `data/test-claims-unlabelled.json` 的全部 **153** 条 `claim_id`。
  **Deliverable**: a `.zip` containing **exactly one** UTF-8 JSON file `test-output.json` covering all **153** `claim_id`s in `data/test-claims-unlabelled.json`.
- **Schema**：每个 entry 必须含 `claim_text`、`claim_label`（4 类之一）、`evidences`（非空字符串列表）。
  **Schema**: every entry must contain `claim_text`, `claim_label` (one of 4), `evidences` (non-empty list of string IDs).
- **指标**：F（证据检索 F1）、A（标签 Accuracy）、H_FA（前两者的调和平均，最终排名指标）。
  **Metrics**: F (evidence retrieval F1), A (label accuracy), H_FA (harmonic mean — *the* ranking metric).
- **配额**：Phase 1（2026-05-01 ~ 05-18）5 次/天、共 100 次；Phase 2（05-19 ~ 05-22）总共 3 次。
  **Quota**: Phase 1 (1–18 May 2026) 5/day, 100 total; Phase 2 (19–22 May 2026) 3 total.
- **现状缺口**：当前 `predict_all()` 输出**不带 `claim_text`**，且没有打包脚本、没有提交配额追踪。本文给出补齐方案。
  **Gap**: current `predict_all()` output **omits `claim_text`**, and there is no packaging script or quota ledger. Plan below.

---

## 1. 平台规则解读 / Platform Rules — Verbatim → 实现影响 / Implications

### 1.1 提交格式 / Submission format

| Codabench 要求 / Required | 项目对应 / Maps to | 是否就绪 / Ready? |
|---|---|---|
| 单个 `.zip`，内部仅一个文件 | 需要 `scripts/build_submission.py` 打包 | ❌ 缺 |
| 文件名严格为 `test-output.json` | 当前写到 `outputs/dry_run/preds.json` 等任意名字 | ❌ 改 |
| UTF-8 JSON dict，key = `claim_id` | `write_predictions()` 已用 `ensure_ascii=False, indent=2` | ✅ |
| 每个 entry 含 `claim_text` / `claim_label` / `evidences` | **`claim_text` 当前未写入** —— `eval.py` 不校验，但 Codabench 例子里有 | ⚠️ 补 |
| `claim_label ∈ {SUPPORTS, REFUTES, NOT_ENOUGH_INFO, DISPUTED}` | `src/paths.py: LABELS` 已锁定，`write_predictions()` 已校验 | ✅ |
| `evidences` 是非空 list | `write_predictions()` 已校验 | ✅ |

**关键差异 / Key delta**：
> Codabench 示例 entry 包含 `claim_text`；本地 `eval.py` 不读这一字段；保险起见，**提交版必须把 `claim_text` 从 `test-claims-unlabelled.json` 合并回来**，这样万一平台脚本去校验也不会漏字段。
> The Codabench example includes `claim_text`; local `eval.py` ignores it; to be safe, the submission build step **must merge `claim_text` back from `test-claims-unlabelled.json`** in case the platform validator checks it.

### 1.2 评估指标 / Metrics — 与本地 `eval.py` 对齐 / Mirrors local `eval.py`

排名指标即本地的 `mean_f`、`mean_acc`、`hmean` 三件套（`eval.py:74-76`）。`eval.py:43` 中 `top_six_ev = set(...)` 暗示**多于 6 条证据不会被惩罚 precision**（实际上有惩罚——precision 分母用的是预测长度，不是 6），所以**保留 ≤ 5 条证据**仍是稳妥做法（与项目原有 `top_k=5` 一致）。
The leaderboard's three numbers map 1-for-1 to the local `eval.py` outputs (`eval.py:74-76`). The `top_six_ev = set(...)` line at `eval.py:43` does **not** cap at 6 — precision is penalised by the full predicted length — so the existing `top_k=5` retention budget remains the right call.

### 1.3 提交配额 / Submission quota

| Phase | 窗口 | 每日 | 总计 |
|---|---|---|---|
| Phase 1 (Ongoing) | 2026-05-01 ~ 05-18 | 5 | 100 |
| Phase 2 (Final)   | 2026-05-19 ~ 05-22 | — | 3 |

**实现建议 / Suggested**：在 `outputs/submissions/ledger.jsonl` 里每打包一次记一行（时间戳、目标 phase、git SHA、preds 文件 sha256、本地 dev H_FA），临提交前读 ledger 校验"今日还剩几次 / 本 phase 还剩几次"。
**Suggestion**: append one row to `outputs/submissions/ledger.jsonl` per package (timestamp, target phase, git SHA, preds sha256, local dev H_FA); the build script reads the ledger and aborts if today's or the phase budget is exhausted.

### 1.4 合规红线 / Compliance red lines

| 规则 / Rule | 项目应对 / How we comply |
|---|---|
| 仅用 `train-claims.json` / `dev-claims.json` / `evidence.json` | `src/paths.py` 仅指向这三份；`build_stage0` 不引外源 |
| 禁止人工查看 `test-claims-unlabelled.json` | dry-run / 推理脚本只对它做整体迭代，**不在交互式 notebook cell 里 print 内容** |
| 不允许 hand-fix 预测 | 提交脚本应**直接消费推理产出**，不允许中间 `Edit` |
| 至少包含一个序列模型组件 | Qwen3.5-4B（Transformer）+ bge-m3（Transformer encoder）已满足 |
| 禁止闭源 API（OpenAI / Claude / Gemini / Copilot） | 全栈 ms-swift + Qwen + BGE 开源；CI/dev-time 用 Claude Code 仅做工程辅助、不参与推理路径 |
| 不允许 hand-crafted if-then 分类规则 | label 由 LLM 自一致性投票产出（`ModelInferer.predict`），未硬编码规则 |
| 禁止照搬开源完整实现 | 检索 / SFT / DPO 全在 `src/` 自实现；只复用模型权重与训练框架 |
| 报告系统 = 提交代码 = leaderboard 跑分 | 提交清单需对齐 `optimization_plan.md` §7.3 + 报告版本 |
| 保留可复现日志 | `outputs/PROGRESS.md` + ledger + 模型 commit SHA |

---

## 2. 现状盘点 / Current State Audit

### 2.1 数据与代码 / Data & code

```
data/test-claims-unlabelled.json    # 153 claims, schema: {claim_id: {claim_text}}
eval.py                             # local scorer, computes F / A / H_FA
src/data_io.py
  └── write_predictions(preds, path)  # 校验 label & evidences；不写 claim_text
src/inference.py
  └── predict_all(claims, inferer, out_path)  # 落盘 {label, evidences}
outputs/dry_run/preds.json          # 当前命名 ≠ test-output.json
```

### 2.2 三个具体缺口 / Three concrete gaps

1. **`claim_text` 缺失** —— `predict_all()` 把 `inferer.predict(claim_text)` 的返回（只含 label+evs）原样写盘，没有把 `claim_text` 合并回去。
   **Missing `claim_text`** — `predict_all()` only stores what the inferer returns.
2. **打包脚本缺失** —— 没有把 `preds.json` → 改名 `test-output.json` → 压成 `.zip` 的脚本。
   **No packaging script** — nothing renames + zips.
3. **配额追踪缺失** —— 没有 ledger，肉眼数 100 次很危险。
   **No quota ledger** — manually tracking 100 submissions is fragile.

### 2.3 已就绪的部分 / Already in place

- 4 个标签的硬白名单（`paths.LABELS`），`write_predictions()` 校验非空 evidences。
- `RetrievalOnlyInferer` / `ZeroShotInferer` / `ModelInferer` 三种 inferer 输出 schema 完全一致，可直接喂打包脚本。
- `dry_run.py` 中 `_check_prediction_format()` 已经用 `eval.py` schema 校验过 dry-run preds。
- `outputs/dry_run/preds.json` 是真实可工作的样本，可拿来做"假提交"演练。

---

## 3. 流程实现方案 / Process Implementation Plan

### 3.1 端到端流水线（5 步）/ End-to-end pipeline (5 steps)

```
Step 1: 推理 / Inference
  python -m scripts.run_inference --target test --out outputs/predictions/test_run_<TAG>.json
    └── 在 Colab/AutoDL 上跑 ModelInferer over test-claims-unlabelled (153 claims)
    └── 落盘 {claim_id: {claim_label, evidences}}

Step 2: 打包 / Build submission
  python -m scripts.build_submission \
      --preds outputs/predictions/test_run_<TAG>.json \
      --tag <TAG> --phase 1
    └── 合并 claim_text（来自 test-claims-unlabelled.json）
    └── 校验 153 个 claim_id 全在、label 合法、evidences 非空、evidence_id ∈ evidence.json
    └── 写 outputs/submissions/<TAG>/test-output.json
    └── zip 成 outputs/submissions/<TAG>/submission.zip
    └── append outputs/submissions/ledger.jsonl

Step 3: 本地自评 / Local sanity check
  python eval.py --predictions outputs/submissions/<TAG>/test-output.json \
                 --groundtruth data/dev-claims.json
    └── 注意：用 dev 自评只是 "schema 不报错" 校验，分数无意义（test 没标）

Step 4: 上传 / Upload
  浏览器 → Codabench → My Submissions → 选 phase → 上传 submission.zip

Step 5: 回填 / Backfill
  把 leaderboard 实际跑分填进 ledger.jsonl 那一行的 score 字段
```

### 3.2 需要新增的两个脚本 / Two new scripts to add

#### `scripts/build_submission.py`

职责 / Responsibilities：
1. 读 preds.json + `test-claims-unlabelled.json`，把 `claim_text` 合并进每个 entry
2. **完整性校验**：
   - 153 个 claim_id 全部出现；
   - 每个 `claim_label ∈ LABELS`；
   - 每个 `evidences` 是 list 且非空；
   - 每个 evidence id 形如 `evidence-\d+` 且**真实存在于 `evidence.json`**（防止幻觉 ID）；
   - 每条 evidences 长度 ≤ 5（与训练时 top_k 对齐）。
3. 写 `outputs/submissions/<TAG>/test-output.json`（UTF-8、`ensure_ascii=False`、`indent=2`）
4. `zipfile.ZipFile(..., 'w', ZIP_DEFLATED)` 打包成 `submission.zip`
5. 追加 ledger 行：`{ts, tag, phase, git_sha, preds_sha256, dev_holdout_hmean, codabench_hmean: null}`
6. 校验 ledger 配额：phase=1 时今日 < 5 且累计 < 100；phase=2 时累计 < 3；超额拒绝。

#### `scripts/run_inference.py`（如果还没有）/ if absent

职责 / Responsibilities：
1. 接收 `--target {dev,test,diag_test}` 与 `--out`
2. 加载已训好的 retrieval pipeline + SFT/DPO 模型
3. 调 `predict_all(claims, inferer, out_path)` —— **改造 inferer.predict 返回 dict 时透传 `claim_text`，或在 predict_all 里合并** —— 让 preds.json 自带 `claim_text`，避免 build_submission 再去 join

> 当前 `predict_all` 已经有 `claims[cid]["claim_text"]` 在手，**最小改动**：在它写盘前给每条 record 注入 `claim_text`。这样 build_submission 只剩"改名 + zip + 校验 + ledger"四件事。
> Minimal patch: have `predict_all()` inject `claim_text` from its input dict; build_submission then only renames, zips, validates, and updates the ledger.

### 3.3 Schema 对照表 / Schema cheat sheet

| 字段 | 类型 | 来源 | 谁负责填 |
|---|---|---|---|
| `claim_text` | str | `test-claims-unlabelled.json` | `predict_all` 注入 |
| `claim_label` | str ∈ LABELS | `inferer.predict()` | inferer |
| `evidences` | list[str], 1–5, 每个 ∈ `evidence.json` | `inferer.predict()` | inferer + retriever |

---

## 4. 校验清单 / Pre-submission Checklist

提交前必须全绿 / All must be green before upload：

- [ ] preds 文件覆盖全部 **153** 个 `claim_id`（用 `set` 差集校验）
- [ ] 每个 entry 同时有 `claim_text` / `claim_label` / `evidences`
- [ ] `claim_label` 在 4 类白名单内
- [ ] `evidences` 是非空 list，每个元素是 `evidence-\d+` 形态
- [ ] 抽查 ≥ 5 个 evidence id 真实存在于 `evidence.json`
- [ ] `len(evidences) ≤ 5` per claim（与设计一致，避免 precision 被稀释）
- [ ] `test-output.json` 单文件 zip，zip 内**没有目录层级**（解压即得文件，不是 `<TAG>/test-output.json`）
- [ ] 文件用 UTF-8（`file -bi` 或 PowerShell `Get-Content -Encoding UTF8`）
- [ ] ledger 校验当日 / 当 phase 配额未超
- [ ] git working tree clean，记录 commit SHA 进 ledger
- [ ] 本地 dev H_FA 比上次提交高（或有明确解释为何允许下降）

---

## 5. 与同行评审的关系 / Relation to Peer Review

`materials/peer_review_instruction.pdf`（Part 1）与 `peer-review-scoring.pdf`（Part 2）属于 **Assignment 3 报告环节**（占 8 分），与 Codabench 跑分**完全独立**：
- Part 1：5/21–5/25，每人评 2 份匿名报告（双盲）；
- Part 2：5/26–5/28，团队成员一起给收到的 reviews 打分。

→ 本文档**只覆盖 Codabench leaderboard**；评审环节的产物是 LMS 上的 review 文本，不需要打包到 submission.zip。

`peer_review_instruction.pdf` (Part 1) and `peer-review-scoring.pdf` (Part 2) cover **Assignment 3 report peer review** (worth 8 marks), which is **fully orthogonal** to the Codabench leaderboard — this document covers leaderboard only.

---

## 6. 落地优先级 / Landing Priority

1. **P0** — 给 `predict_all()` 打 `claim_text` 注入补丁（5 行）。Without this, every run produces a non-conforming preds file.
2. **P0** — 写 `scripts/build_submission.py`（带完整性校验 + ledger），不带 ledger 也可先跑通打包。
3. **P1** — 写 `scripts/run_inference.py`（如果当前推理逻辑只活在 notebook 里）。
4. **P2** — ledger 加配额硬熔断；Phase 2 开始前手动 audit 一次 ledger。
5. **P2** — 在 `tests/` 下加 `test_build_submission.py`：用 `outputs/dry_run/preds.json` 做端到端 smoke。

> 立即可做的最小切片 / Smallest first slice：把现成的 `outputs/dry_run/preds.json` 强行喂 build_submission（即便它只覆盖 dev+diag_test 而非 test），跑一遍把"打包→校验"链路在本地点亮，然后再去 Colab 跑真正的 test 推理。
> Smallest first slice: feed the existing `outputs/dry_run/preds.json` into `build_submission.py` (even though it covers dev+diag_test, not test) just to light up the package-and-validate loop locally, **then** run real test inference on Colab.
