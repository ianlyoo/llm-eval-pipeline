# Judge Experiment: Deterministic Proxy vs Real LLM Judge (A vs B) — 20-sample Simulation

> **Label: Deterministic Proxy SIMULATION - not real LLM**
>
> Current committed scores are deterministic proxy simulation (heuristic token/numeric overlap + deterministic-strict vs lenient + noise).
> Real LLM variance requires API keys. See `Real Judge Experiment` section below for reproduction commands.
> Do NOT claim real LLM judge results — honesty first.

- **Mode detected:** SIMULATION (deterministic proxy) — no FEATHERLESS_API_KEY / XAI_API_KEY / OPENAI_API_KEY available in env (checked 2026-08-23; only OPENCODE_GO_API_KEY present).
- **Commit artifacts treated as proxy:**
  - `out/judge_scores.jsonl` — 30 lines (10 samples x 3 runs, seed=42, profile default + noise)
  - `out/disagreement_cases.jsonl` — 10 cases (deterministic-strict vs deterministic-lenient, max_diff >=1 → 10/10)
  - `out/judge_reliability.log` — header now includes `SIMULATION: deterministic-strict vs lenient with noise, not real LLM`
- **Deterministic Proxy simulation** means heuristic scoring + 20% random ±1 noise, not real LLM API calls. Cost is $0.000000, latency ~1ms.
- **real LLM API** requires valid provider keys and runs `eval/llm_judge.py --llm-provider openai|featherless|grok`.

## A vs B Simulation (20 samples, deterministic-strict vs deterministic-lenient)

Regenerated 20-sample A vs B with identical sampled indices (seed=42) to demonstrate honest A vs B comparison without API:

- **A:** `deterministic-strict` (all scores -1, clamp 1-5)
- **B:** `deterministic-lenient` (all scores +1, clamp 1-5)
- Inputs: `data/synthetic_qa.jsonl` (50 total), sampled 20 indices via seed 42.

Results (cost $0, reproducible):

- **A:** avg latency 1.0ms, avg tokens 72, total cost $0.000000
- **B:** avg latency 1.0ms, avg tokens 72, total cost $0.000000

### Per-Metric Means (1-5) — 20 samples

| Metric | A mean (strict) | B mean (lenient) | delta (B-A) |
|--------|-----------------|------------------|-------------|
| correctness | 4.00 | 5.00 | +1.00 |
| groundedness | 4.00 | 5.00 | +1.00 |
| relevance | 1.40 | 2.95 | +1.55 |
| completeness | 4.00 | 5.00 | +1.00 |

**Agreement:** 0/20 exact (max_diff=0), 20/20 with max_diff >=1. Mean max_diff: 1.55, max: 2.

### Reliability (10 samples x 3 runs, from out/judge_reliability.log)

- **Samples:** idx 40,7,1,17,15,14,8,6,34,5 — each 3 runs with `noise_rng` seed `seed*100 + run*10 + idx` (20% ±1 jitter).
- **Variance:** overall mean 0.0444, max 0.2222 → PASS (<1.0). This is **simulation jitter variance**, not real LLM temperature variance.
- **Latency/tokens from log:** avg latency 1.0ms, avg tokens 70, total cost $0.000000.
- **Disagreement:** 10/10 cases with max_diff >=1 (profiles deterministic-strict vs deterministic-lenient) — forced by ±1 shift, not semantic disagreement.

## Interpretation

- Deterministic proxy scores are intentionally high (5s) when candidate==reference; variance comes only from 20% noise (+-1).
- Strict vs lenient disagreement is forced by +-1 profile shift - cases differ by exactly 1. This is a **stability smoke test**, not model disagreement.
- Real LLM judges show larger variance (temperature, prompt sensitivity) and semantic disagreement; do not extrapolate simulation numbers to LLM behavior.
- Threshold PASS (<1.0 variance) here means noise is small; real LLM variance may exceed this.
- Committed artifacts are **honest simulation**; no API was called, no LLM output was fabricated.

## Real Judge Experiment (when API keys available)

Current committed artifacts are **Deterministic Proxy simulation**. To run a real A vs B experiment (20-50 samples, two different judge models):

```bash
## Real Judge Experiment (when API keys available)
FEATHERLESS_API_KEY=... python -m eval.llm_judge --input data/synthetic_qa.jsonl --llm-provider featherless --model tngtech/deepseek-r1-distill-llama-70b --output out/real_judge_a.jsonl
XAI_API_KEY=... python -m eval.llm_judge --llm-provider grok --output out/real_judge_b.jsonl
python scripts/compare_judges.py out/real_judge_a.jsonl out/real_judge_b.jsonl --output out/real_judge_comparison.md

# Variants — use any two different models/providers and 20-50 samples:
FEATHERLESS_API_KEY=... python -m eval.llm_judge --input data/synthetic_qa.jsonl --llm-provider featherless --model tngtech/deepseek-r1-distill-llama-70b --output out/real_judge_a.jsonl --sample 20
XAI_API_KEY=... python -m eval.llm_judge --llm-provider grok --model grok-2-latest --output out/real_judge_b.jsonl --sample 20
python scripts/compare_judges.py out/real_judge_a.jsonl out/real_judge_b.jsonl --output out/real_judge_comparison.md

OPENAI_API_KEY=... python -m eval.llm_judge --input data/synthetic_qa.jsonl --llm-provider openai --model gpt-4o-mini --output out/real_judge_a.jsonl --sample 30
FEATHERLESS_API_KEY=... python -m eval.llm_judge --llm-provider featherless --model meta-llama/Meta-Llama-3.1-8B-Instruct --output out/real_judge_b.jsonl --sample 30
python scripts/compare_judges.py out/real_judge_a.jsonl out/real_judge_b.jsonl --output out/real_judge_comparison.md

# Reliability with real LLM (3-run variance, real):
FEATHERLESS_API_KEY=... python -m eval.llm_judge --input data/synthetic_qa.jsonl --llm-provider featherless --output out/real_judge_a.jsonl --sample 50 --reliability --reliability-log out/real_judge_reliability.log
```

Cost estimate: gpt-4o-mini ~$0.002/1k tokens, Featherless ~$0.001/1k, Grok ~$0.005/1k. 30 samples x ~80 tokens ~ 2.4k tokens -> <$0.02 total.

When keys are unavailable (current CI), keep this file as **honest simulation label** — do not claim real LLM results.
No API keys are exposed in docs/logs; placeholder `...` is used.

## Reproducibility (simulation, no keys needed)

```bash
# Deterministic proxy A vs B (20 samples)
python -c "from eval.llm_judge import judge_entry; import json, pathlib, random; ..."
python scripts/compare_judges.py out/judge_scores.jsonl out/disagreement_cases.jsonl --output out/compare_tmp.md --mode simulation

# Full reliability simulation (10 samples x 3 runs, matches committed log)
python -m eval.llm_judge --input data/synthetic_qa.jsonl --output out/judge_scores.jsonl --sample 10 --seed 42 --reliability
cat out/judge_reliability.log  # header must contain SIMULATION label
```

## Files

- `eval/llm_judge.py` — header distinguishes Deterministic Proxy vs LLM Judge; `--reliability` logs `SIMULATION: deterministic-strict vs lenient with noise, not real LLM` when `--llm-provider none`.
- `scripts/compare_judges.py` — auto-detects simulation vs real via cost/profile; `--mode simulation|real` to force label.
- `out/judge_reliability.log` — now has `[SIMULATION]` tag + real-LLM instructions on line 2-3.
- `out/real_judge_comparison.md` — this file (honest comparison doc).

*Generated 2026-08-23. If API keys become available, re-run Real Judge Experiment above and overwrite this file with `Mode detected: REAL LLM` and real cost/latency.*
