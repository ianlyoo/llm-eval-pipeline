# Evidence Compact — llm-eval-pipeline (Deterministic proxy + Real harness)

> Compact reproduction evidence for reviewers. Full artifacts live in `out/` (mostly gitignored, reproducible in one command). This doc plus whitelisted JSONs are committed to `master` so GitHub shows measured values. Targets without artifacts remain absent by design (fail-closed).

## Measured baseline (deterministic proxy, 50 samples)

Source: `out/metrics_report.json` (1137B) and `out/baseline_metrics.json` (snapshot, same content).

| Metric | Measured value | Detail |
|--------|---------------|--------|
| **Hit@5** | **0.66** | `hit_at_k:0.66, hits:33, total:50, k:5, mode:proxy-expected_keyword` |
| **faithfulness** | **0.8034** | `count:50, min:0.50, max:1.00` (numeric 50%+token 50% blend, 50 scores) |
| **failure_rate** | **0.34** | `failed:17, passed:33, pass_rate:0.66, by_rule:{expected_keyword:17}` |
| **PyTorch tiny-MLP** | **loss 0.7615 → 0.5788, best 0.4356, acc 0.375→0.75** | 100 steps, 2 epochs, batch 16, lr 0.001, seed 42, cpu, torch 2.6.0+cpu |
| **lm-eval gsm8k** | **0.05 (flexible-extract, strict 0.05, stderr 0.05, limit 20)** | `lm_eval --model hf --model_args pretrained=gpt2,device=cpu --tasks gsm8k --limit 20` → `out/lm_eval_raw.log` (23818B) |

Targets (not measured — no artifact, fail-closed): **Hit@5 Target 0.85 / faithfulness Target 0.88 / failure Target 0.15 / lm-eval after Target 0.10** — documented in README/docs as `Projected/Roadmap/Simulation, not measured`. `out/improved_metrics.json` intentionally absent; `scripts/compare_rag.py` exits 2.

## Repro commands (1-command each)

```bash
# Hit@5 / faithfulness / failure_rate (deterministic)
python -m eval.metrics
Get-Content out/metrics_report.json   # hit_at_k 0.66, faithfulness 0.8034

# PyTorch tiny-MLP — full training (synthetic task, not LLM fine-tune)
python -m eval.pytorch_experiment --epochs 2 --steps-per-epoch 50
Get-Content out/pytorch_experiment_log.json   # 0.7615->0.5788 best 0.4356

# PyTorch dry-run (CI lightweight)
python -m eval.pytorch_experiment --dry-run

# lm-eval gsm8k (real harness, tiny limit for CI)
lm_eval --model hf --model_args pretrained=gpt2,device=cpu --tasks gsm8k --limit 20
Get-Content out/lm_eval_raw.log   # |gsm8k|3|flexible-extract|5|exact_match|0.05|0.05|

# Target fail-closed (correctly fails — improved artifact absent)
python scripts/compare_rag.py --baseline out/baseline_metrics.json --improved out/improved_metrics.json
# -> ERROR exit 2: missing improved — Target is not measured
```

## Aggregate snippets (first 5 lines, verbatim)

`out/metrics_report.json`:

```json
{
  "hit_at_5": {
    "hit_at_k": 0.66,
    "hits": 33,
    "total": 50,
```

`out/pytorch_experiment_log.json`:

```json
{
  "mode": "train",
  "epochs": 2,
  "steps_per_epoch": 50,
  "batch_size": 16,
```

## Tracked evidence files (whitelisted in `.gitignore`)

- `out/metrics_report.json` (1.1KB, measured Hit@5 0.66)
- `out/baseline_metrics.json` (1.1KB, snapshot)
- `out/pytorch_experiment_log.json` (1.5KB, 0.7615→0.5788)
- `out/lm_eval_raw.log` excluded from commit (23KB, too large) — reproduced locally; `docs/evidence_compact.md` carries the measured value. Whitelisted logs remain: `rag_baseline.log`, `synthetic_qa_sample.md`, etc.

`out/` otherwise gitignored (`out/*.json` blocked except whitelisted); targets absent = intentionally not whitelisted.

## Verify on GitHub

```bash
curl -s https://raw.githubusercontent.com/ianlyoo/llm-eval-pipeline/master/docs/evidence_compact.md | head -n 20
curl -s https://raw.githubusercontent.com/ianlyoo/llm-eval-pipeline/master/out/metrics_report.json | head -n 5
curl -s https://raw.githubusercontent.com/ianlyoo/llm-eval-pipeline/master/out/pytorch_experiment_log.json | head -n 5
```
