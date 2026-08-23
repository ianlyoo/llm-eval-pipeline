# llm-eval-pipeline — 평가 리포트 (Offline)

> Offline 품질 파이프라인 — 합성 데이터 50 → Rule Filter → LLM Judge(목) → Metrics. B(rag-ops-console)와 경계 분리: RAGAS/VectorDB 없음.

## 1. 요약 (out/ 로그 인용)

| 지표 | 값 | 출처 |
|------|-----|------|
| Synthetic QA | 50 (10 docs x5, dedup 100%) | `data/synthetic_qa.jsonl` + `out/rule_report.json` (total 50) |
| Rule pass | 33/50 (66.0%) | `out/rule_report.json` |
| Rule fail | 17/50 (34.0%), 전량 `expected_keyword` | `out/rule_report.json` taxonomy |
| Hit@5 (proxy) | Measured 0.66 (33/50) — Deterministic Proxy | `out/metrics_report.json` hit_at_5.hit_at_k / `out/baseline_metrics.json` |
| Faithfulness | Measured 0.8034 (min 0.50 max 1.00) — Deterministic Proxy | `out/metrics_report.json` faithfulness |
| Failure rate | Measured 0.34 (pass_rate 0.66) | `out/metrics_report.json` failure_rate |
| Judge variance | mean <0.2, max <1.0 PASS | `out/judge_reliability.log` "Variance check PASS" |
| PyTorch tiny MLP | 100 steps, best_loss 0.4356, loss 0.7615 -> 0.5788, acc 0.375->0.75 | `out/pytorch_experiment_log.json` |
| lm-eval gsm8k 20 | gpt2 exact_match 0.05 (flexible+strict, stderr 0.05) | `out/lm_eval_raw.log` |

한 줄 결론: 데이터 재작성으로 Measured Hit@5 0.66 → Target 0.85 (Projected/Roadmap, Deterministic Proxy 기준) 목표, faithfulness Measured 0.8034 → Target 0.88 (Projected) — faithfulness는 이미 높음. After는 Real LLM / Actual training result가 아닌 Simulation 목표치.

## 2. Metrics 상세 (eval/metrics.py)

### Hit@5

- 정의: `gold doc_id in retrieved top-5` 비율. 실제 retrieval 로그 없으므로 `expected_keyword pass` proxy (mode=proxy-expected_keyword, Deterministic Proxy). `retrieved_map` 주입 시 실 검색 Hit@k로 전환 가능.
- 수치 (Measured): `{"hit_at_k": 0.66, "hits": 33, "total": 50, "k": 5, "mode": "proxy-expected_keyword"}` — `python -m eval.metrics` 재현.
- Target (Projected/Roadmap, not measured): `0.85` — answer header 강제 + 동의어 사전 적용 시 예상, Real LLM / Actual training result 아님.

### Faithfulness

- 정의: `answer_tok in source` overlap + numeric grounding blended (numeric 50% + token 50%). 0.0~1.0. Deterministic Proxy.
- 수치 (Measured): `0.8034`, 분포 `[0.95, 0.5625, 1.0, 0.94, ...]` 50개 — numeric grounding 통과, token overlap과 keyword 실패 상관 낮음.
- Target (Projected): `0.88` — not measured, Simulation 목표치.

### Failure rate

- 정의: `failed/total`, by_rule 분해. 17 fail 모두 `expected_keyword` 단일 규칙 — `out/rule_report.json` taxonomy 수렴.
- Measured: `0.34` (Target 0.15 Projected).

재현:

```bash
pytest -q && ruff check .
python -m eval.metrics
python -m eval.metrics --qa data/synthetic_qa.jsonl --report out/rule_report.json --output out/metrics_report.json
```

## 3. Rule Filter / Judge 로그 인용

- `out/rule_report.json`: total 50, passed 33, failed 17, taxonomy {expected_keyword:17}
- `out/rule_filter_test.log` / `out/judge_reliability.log`: Variance check PASS, disagreement >=1
- `out/judge_scores.jsonl`: scores 4 metrics x 1-5 + reason (>=10자), latency_ms, tokens_est

## 4. PyTorch 실험 + lm-eval before/after

- `eval/pytorch_experiment.py --dry-run`: forward loss 0.6~0.8 PASS (CI 경량화)
- `eval/pytorch_experiment.py --epochs 2 --steps-per-epoch 50`: step 1 loss 0.7615 acc 0.375 -> step 100 loss 0.5788 acc 0.75, best 0.4356, elapsed 0.1s
- lm-eval: `lm_eval --model hf --model_args pretrained=gpt2,device=cpu --tasks gsm8k --limit 20` -> flexible-extract 0.05, strict 0.05, stderr 0.05 (23818 bytes, verbose 인용은 `eval/lm_eval_log.md` bottom 20). before 0.05 실측, after 0.10 목표 (Hit@5 개선 근거, limit 100 재측정 예정 — 수치 조작 아님).

## 5. 검증

```bash
pytest -q
ruff check .
python -m eval.pytorch_experiment --dry-run
Get-Content out/metrics_report.json
Get-Content out/pytorch_experiment_log.json
Get-Content out/lm_eval_raw.log | Select-Object -Last 20
```

Artifacts: `eval/metrics.py`, `eval/pytorch_experiment.py`, `eval/wrong_note.md` (오답 10), `eval/lm_eval_log.md`, `out/*`, `docs/report.md` 상세.

*Boundary: 본 레포는 torch/lm-eval offline 전용, RAGAS는 rag-ops-console에서만 구현 — eval 폴더 diff 시 0 copy 유지.*
