# `benchmark/` — Codabench leaderboard artifacts

> 所有提交到 Codabench 的产物都在这个文件夹里。一个文件夹自包含一次提交。
> Everything that goes into a Codabench submission lives here. One folder per run, fully self-contained.

---

## 目录结构 / Layout

```
benchmark/
├── README.md              # 你正在看
├── ledger.jsonl           # 跨提交配额追踪（每行一次 build_submission 调用）
├── .evidence_ids.txt      # evidence.json 的 key 集合缓存（gitignored，自动建）
└── runs/
    └── <TAG>/             # 一次完整提交的所有产物（命名约定：v1-base-rag、v2-sft-greedy、…）
        ├── config.json    # run_inference 的参数 + git SHA 快照（小，进 git）
        ├── preds.json     # run_inference 的原始输出（claim_text 已注入；大，gitignored）
        ├── test-output.json  # 复制自 preds + 校验 + 重 merge claim_text（gitignored，可重 build）
        └── submission.zip # 上传到 Codabench 的最终文件（gitignored，可重 zip）
```

**进 git 的**：`README.md`、`ledger.jsonl`、每个 `runs/<TAG>/config.json`
**不进 git 的**：`*.zip`、`preds.json`、`test-output.json`、`.evidence_ids.txt`、`runs/<TAG>/` 下其它大文件

config.json 进 git 的好处：你/同伴看 `git log benchmark/runs/` 就能看到每次提交用了什么参数；preds 和 zip 都能从 `config.json` 里的 flags 重 run 出来。

**Tracked**: `README.md`, `ledger.jsonl`, each `runs/<TAG>/config.json`.
**Gitignored**: `*.zip`, `preds.json`, `test-output.json`, `.evidence_ids.txt`, anything else under `runs/<TAG>/`.

Tracking config.json means `git log benchmark/runs/` shows exactly which flags produced each submission; preds + zip can be re-derived from those flags.

---

## 一次完整提交流程 / End-to-end flow

```bash
TAG=v1-base-rag-greedy

# 1. 推理：测试集 153 条（base + RAG，无 SFT），写到 benchmark/runs/<TAG>/
python -m scripts.run_inference \
    --target test --tag $TAG \
    --decoding greedy --prompt-version v1 --final-k 5
# 产物：benchmark/runs/<TAG>/{preds.json, config.json}

# 2. 打包：校验 schema + zip + 写 ledger 行
python -m scripts.build_submission \
    --preds benchmark/runs/$TAG/preds.json \
    --tag $TAG --phase 1
# 产物：benchmark/runs/<TAG>/{test-output.json, submission.zip}
#       benchmark/ledger.jsonl 多一行

# 3. 上传：手动 → Codabench → My Submissions → Phase 1
ls -lh benchmark/runs/$TAG/submission.zip

# 4. 平台跑分回填：把 leaderboard 拿到的 H_FA 填进对应 ledger 行的 codabench_hmean
```

---

## TAG 命名约定 / Naming convention

短小、可读、含关键变量。建议格式：`<版本>-<模型>-<RAG>-<解码>`，比如：

| TAG | 含义 |
|---|---|
| `v1-base-rag-greedy`        | Base Qwen3.5-4B + BM25+dense RAG + greedy 解码 + prompt v1 |
| `v2-base-rag-sc5`           | 同上但 self-consistency 5 samples |
| `v3-sft-merged-greedy`      | 用 SFT-merged base + RAG + greedy（如果之后 SFT 重训跑通了再用） |
| `v4-base-rag-bm25only`      | dense 索引坏了的备用方案 |
| `smoke-*`                   | 仅用于验证打包链路，不上传（用 `--force` 跑 dev 数据时） |

**别在 TAG 里写日期** —— ledger 已经记 `ts_aest_date` + `git_sha`，再写一遍冗余。

---

## 配额状态查询 / Quota check

```bash
# 看现在用了多少配额
python -c "
import json, datetime
from pathlib import Path
ledger = Path('benchmark/ledger.jsonl')
if not ledger.exists():
    print('no submissions yet'); exit()
rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
today = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=10))).date().isoformat()
for phase in (1, 2):
    in_phase = [r for r in rows if r['phase'] == phase]
    today_p = [r for r in in_phase if r.get('ts_aest_date') == today]
    cap_total = {1: 100, 2: 3}[phase]
    cap_daily = {1: 5, 2: None}[phase]
    print(f'Phase {phase}: {len(in_phase)}/{cap_total} total' + (f', {len(today_p)}/{cap_daily} today (AEST)' if cap_daily else ''))
"
```

也可以直接 `python -m scripts.audit_cache | grep ledger` 看 ledger 是否就绪。

---

## 跨机器迁移注意事项 / Cross-machine migration

详见 `AUTODL_TO_COLAB.md`。简版：

- **必须迁** `benchmark/ledger.jsonl`（不然新机器上配额计数从 0 重新开始，可能误判超额）
- **可选迁** `benchmark/.evidence_ids.txt`（不迁第一次跑会自动重建，~30 秒）
- **不需要迁** `benchmark/runs/<TAG>/preds.json` 和 `submission.zip`（可从 `config.json` 重 run）
- **必须迁** `benchmark/runs/<TAG>/config.json`（已 tracked，git pull 自带）

---

## FAQ

**Q: 为什么不放在 `outputs/submissions/` 而是项目根的 `benchmark/`？**
A: outputs/ 是「项目运行产物」（splits / sft_data / 索引 / checkpoint）；benchmark/ 是「对外提交物」。两类生命周期和迁移策略不同：outputs/ 在每台机器各自重建，benchmark/ 跨机器同步（ledger 必须一致）。

**Q: 一个 TAG 跑了好几次怎么办？**
A: 后跑的会覆盖 `preds.json` / `submission.zip`，但 ledger 里每次 `build_submission` 都会追加一行，历史可查。如果想保留旧的，新 TAG 加后缀（如 `v1-base-rag-greedy-rerun`）。

**Q: 校验都过了但 Codabench 拒收？**
A: 看 `runs/<TAG>/test-output.json` 第一条 entry 的 keys 是否完全等于 `["claim_text", "claim_label", "evidences"]`。如果是字段名错了（比如多了下划线），翻 `BENCHMARK_SUBMISSION.md` §1.1 的 schema 表对照。如果是 evidence ID 问题，跑 `python -m scripts.build_submission --refresh-evidence-cache` 重建缓存再 build。
