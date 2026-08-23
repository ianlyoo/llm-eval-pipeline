# Audit — llm-eval-pipeline 문서 수치 vs Artifact (2026-08-23)

> Scope: `llm-eval-pipeline/README.md` + `docs/{eval,limits,reproducibility,report}.md` + `docs/report.html`에 등장하는 모든 `\d+\.\d+` 주장을 `out/` artifact와 대조.

## Method

`Select-String "\d+\.\d+"` on README/docs → 각 주장의 artifact 존재/타입(measured/deterministic proxy/target/historical) 및 1-command 재현 여부 판정.

## Table

| Claim | Source artifact | Type | Reproducible? | Action |
|---|---|---|---|---|
| **Hit@5 0.66** (33/50, proxy-expected_keyword) | `out/metrics_report.json` `hit_at_k 0.66 hits33 total50 mode proxy-expected_keyword` 1137B + `out/baseline_metrics.json` | **Measured — Deterministic proxy** | **Yes** `python -m eval.metrics` | README `Measured 0.66` 명시 — 유지 |
| **faithfulness 0.8034** (min 0.50 max 1.00, 50 scores) | `out/metrics_report.json` `faithfulness 0.8034` | **Measured — Deterministic proxy** (numeric 50% + token 50%) | **Yes** 위와 동일 | README `Measured 0.8034` — 유지, `limits.md`에 관대함 서술 |
| **failure rate 0.34** (pass 0.66, failed 17, by_rule expected_keyword) | `out/metrics_report.json` `failure_rate 0.34` + `out/rule_report.json` | **Measured — Deterministic** | **Yes** | 유지 |
| **PyTorch 0.7615 -> 0.5788, best 0.4356, acc 0.375->0.75** (100 steps, 2 epochs, batch16, lr0.001, seed42, cpu) | `out/pytorch_experiment_log.json` 1522B `logs step1 0.7615 ... step100 0.5788 best 0.4356` | **Measured — Actual tiny-MLP** | **Yes** `python -m eval.pytorch_experiment --epochs 2 --steps-per-epoch 50` (or `--dry-run` for CI) | 유지 |
| **lm-eval gsm8k 0.05** (flexible+strict 0.05, stderr 0.05, limit20, gpt2 124M, 5-shot) | `out/lm_eval_raw.log` 23818B `|gsm8k|3|flexible-extract|5|exact_match|0.05|0.05|` | **Measured — Real lm-eval harness** (limit 작음) | **Yes** `lm_eval --model hf --model_args pretrained=gpt2,device=cpu --tasks gsm8k --limit 20` | `limits.md`에 샘플 수 한계 서술 — 유지 |
| **Hit@5 Target 0.85 / faithfulness Target 0.88** | **artifact 없음** (`out/improved_metrics.json` intentionally absent, `scripts/compare_rag.py` exit 2) | **Target — Projected / Roadmap / Simulation** (wrong_note actions 1-2, not Real LLM, not Actual training result) | **No** (v2 synthetic 후 측정) | README `Target (Projected / Roadmap / Simulation, not measured)` 명시 — 유지, measured로 둔갑 금지 |
| **lm-eval after Target 0.10 / failure rate Target 0.15** | **artifact 없음** | **Target — Projected** (Hit@5 개선 근거, limit100 재측정 예정) | **No** | `report.md` `After (목표) 0.10`로 명시 — 유지 |
| **Judge variance mean <0.2 max <1.0 PASS** | `out/judge_reliability.log` + `out/judge_scores.jsonl` + `out/disagreement_cases.jsonl` | **Measured** (heuristic, 목 분산) | **Yes** | `limits.md`에 실제 LLM 분산과 상관 낮음 서술 — 유지 |
| **PyTorch dry-run 0.6~0.8 PASS** | `out/pytorch_experiment_log.json` (dry-run) / `pytorch_experiment_dryrun.log` | **Measured — dry-run proxy** (CI 경량) | **Yes** `python -m eval.pytorch_experiment --dry-run` | 유지 |

## README 라벨 검증

- [x] `README.md` — `Measured baseline 0.66/0.8034/0.34` vs `Target after (Projected / Roadmap / Simulation, not measured) 0.85/0.88` 분리, `Not Real LLM, not Actual training result` 병기, `improved_metrics.json absent -> fail-closed` 서술 — **Target을 measured로 표기 0건**.
- [x] `docs/{eval,report,reproducibility}` — 전원 `Measured 0.66 -> Target 0.85 (Projected/Roadmap, not measured)` 구분.
- [x] `scripts/compare_rag.py` — fail-closed (missing/same/no-metrics → exit 2), hard-coded fallback 없음.

## Evidence

- `out/metrics_report.json` 1137B, `out/baseline_metrics.json` 1137B, `out/pytorch_experiment_log.json` 1522B, `out/lm_eval_raw.log` 23818B, `out/judge_scores.jsonl`, `out/rule_report.json`

---
No product code modified. See also `portfolio/out/final_audit_claims.md` for cross-repo summary.
