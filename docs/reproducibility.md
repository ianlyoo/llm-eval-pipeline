# Reproducibility Guide — llm-eval-pipeline (Offline)

> **Measured vs Target — fail-closed honesty**: `Measured` baseline `Hit@5 0.66 / faithfulness 0.8034 / failure rate 0.34` (Deterministic Proxy, `out/metrics_report.json` + `out/baseline_metrics.json`) → `Target` after `0.85 / 0.88` (Projected / Roadmap / Simulation — not measured, not Real LLM, not Actual training result).
> Sources: `out/metrics_report.json`, `out/baseline_metrics.json`, `out/rule_report.json`, `data/synthetic_qa.jsonl`, `eval/metrics.py`, `eval/synthetic_data.py`, `eval/rule_filter.py`
> **Honesty note (2026-08-23)**: `out/improved_metrics.json` is *intentionally absent* — no measured improved artifact exists yet. `scripts/compare_rag.py` will fail closed (exit 2) if invoked without a second measured artifact. Do NOT treat Target 0.85 / 0.88 as measured.

## 1. File Structure (Before/After — Measured vs Target)

```
out/
├── baseline_metrics.json     # Measured baseline — snapshot of metrics_report.json, Deterministic Proxy Hit@5 0.66
├── metrics_report.json       # Measured baseline canonical (Hit@5 0.66 / faithfulness 0.8034 / failure 0.34)
├── improved_metrics.json     # ABSENT — no measured improved yet; Target 0.85 / 0.88 is Projected/Roadmap only
└── comparison.md             # ABSENT — scripts/compare_rag.py fails closed without both measured artifacts
```

Measured baseline is `out/metrics_report.json` (and its snapshot `out/baseline_metrics.json`):

```bash
Copy-Item out/metrics_report.json out/baseline_metrics.json   # Windows
# cp out/metrics_report.json out/baseline_metrics.json        # macOS/Linux
```

`Target` After (`Hit@5 0.85 / faithfulness 0.88`) is a **Projected Roadmap** based on `eval/wrong_note.md` actions 1-2 (answer header + synonym rewrite). It has no measured artifact yet; when fixes land, a real `out/improved_metrics.json` will be generated via `python -m eval.metrics` and then `scripts/compare_rag.py` can produce a measured comparison. Until then, `comparison.md` does not exist — this is correct fail-closed behavior.

Current Measured baseline values (`out/metrics_report.json:2-76`, `out/baseline_metrics.json` identical, Deterministic Proxy):

```json
{
  "hit_at_5": {"hit_at_k": 0.66, "hits": 33, "total": 50, "k": 5, "mode": "proxy-expected_keyword"},
  "faithfulness": {"faithfulness": 0.8034, "count": 50, "min": 0.50, "max": 1.00},
  "failure_rate": {"failure_rate": 0.34, "failed": 17, "pass_rate": 0.66, "by_rule": {"expected_keyword": 17}}
}
```

## 2. One-Command Reproduction

### 2.1 Full pipeline (synthetic → rule filter → metrics)

```bash
# 1. synthetic data (50 QA, seed 42, deterministic dedup)
python -m eval.synthetic_data --count 50 --seed 42 --chunks data/chunks.jsonl --output data/synthetic_qa.jsonl

# 2. rule filter (taxonomy + failure file)
python -m eval.rule_filter --qa data/synthetic_qa.jsonl --output out/rule_report.json
# or: python -m eval.rule_filter

# 3. metrics (Hit@5 proxy + faithfulness + failure_rate)
python -m eval.metrics --qa data/synthetic_qa.jsonl --report out/rule_report.json --output out/metrics_report.json
# shortest:
python -m eval.metrics
Get-Content out/metrics_report.json
```

The task shorthand — `python -m eval.synthetic_data --count 50 && python -m eval.rule_filter && python -m eval.metrics` — regenerates `out/metrics_report.json` from scratch (each module uses defaults `data/synthetic_qa.jsonl` / `out/rule_report.json` / `out/metrics_report.json`).

### 2.2 Metrics only (fast repro, assumes QA + rule report exist)

```bash
python -m eval.metrics && python -m eval.metrics --qa data/synthetic_qa.jsonl --report out/rule_report.json --output out/metrics_report.json
cat out/metrics_report.json | rg "hit_at_k|faithfulness|failure_rate"
```

### 2.3 Comparison (Measured baseline vs Target — fail-closed)

```bash
# BEFORE fix: no measured improved exists yet — this correctly fails closed (exit 2):
python scripts/compare_rag.py --baseline out/baseline_metrics.json --improved out/improved_metrics.json
# → [compare_rag] ERROR: improved artifact not found — correct, no synthetic fallback

# AFTER fix (when measured improved exists):
# 1) apply fixes (answer header + synonym rewrite), regenerate:
#    python -m eval.synthetic_data --count 50 && python -m eval.rule_filter && python -m eval.metrics --output out/improved_metrics.json
# 2) compare two measured artifacts:
python scripts/compare_rag.py --baseline out/baseline_metrics.json --improved out/improved_metrics.json
# → writes out/comparison.md only when both are distinct measured files
Get-Content out/comparison.md
```

`scripts/compare_rag.py` is fail-closed: it loads two distinct measured JSONs and prints a markdown table (Hit@5, faithfulness, failure_rate deltas) to stdout and `out/comparison.md`. It never invents Target 0.85 / 0.88 as measured data. `Measured` vs `Target` comparison is not a measured comparison — docs must label Target as Projected/Roadmap.

## 3. Verification

```bash
pytest -q && ruff check .
python -m eval.metrics
Get-Content out/metrics_report.json                          # hit_at_k 0.66, faithfulness 0.8034
Get-Content out/rule_report.json | Select-Object -First 30   # total 50, failed 17, taxonomy expected_keyword:17
Get-Content out/judge_reliability.log                        # Variance check PASS
python -m eval.pytorch_experiment --dry-run                  # forward loss 0.6~0.8 PASS
```

Artifacts to keep under `out/` after rerun: `rule_report.json`, `metrics_report.json`, `judge_scores.jsonl`, `judge_reliability.log`, `pytorch_experiment_log.json`, `lm_eval_raw.log` (if lm-eval re-run).

## 4. Notes — Measured vs Target honesty

- `Hit@5` mode `proxy-expected_keyword` (`eval/metrics.py:64`): `expected_keyword` pass == hit; inject `retrieved_map` for real retrieval Hit@k. This is a **Deterministic Proxy**, not Real LLM retrieval.
- `Faithfulness` blended numeric 50% + token overlap 50% (`eval/metrics.py:94-97`) — deterministic heuristic, not RAGAS LLM evaluator.
- `Measured` baseline: `Hit@5 0.66 / faithfulness 0.8034 / failure rate 0.34` — from `out/baseline_metrics.json` / `out/metrics_report.json` (50 samples, Deterministic Proxy). Verified: `Get-Content out/metrics_report.json`.
- `Target` After: `0.85 / 0.88` — **Projected / Roadmap / Simulation**, based on answer-header fix (`wrong_note.md` actions 1-2). Not hallucinated but also not measured, not Real LLM, not Actual training result. Requires re-running `eval/synthetic_data.py` v2 + `eval/metrics.py` to become measured.
- To reproduce faithfulness-but-wrong analysis cross-repo, see `../rag-ops-console/docs/evaluation_failure_analysis.md` (faithfulness 1.0 but recall 0.2 case `자동 롤백 → 경비 가이드`).

---

*Reproduce (Measured): `python -m eval.synthetic_data --count 50 && python -m eval.rule_filter && python -m eval.metrics` — outputs `out/metrics_report.json` (Measured Hit@5 0.66, faith 0.8034 baseline, Deterministic Proxy). `Target` 0.85 / 0.88 is Projected — `python scripts/compare_rag.py` fails closed until a measured improved artifact exists.*
