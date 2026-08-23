# Judge Experiment — Deterministic Proxy vs Real LLM Judge

> This doc mirrors `out/real_judge_comparison.md` for the `docs/` path requirement.
> See `out/real_judge_comparison.md` for the primary artifact and `out/judge_reliability.log` for variance details.

## Status: SIMULATION (no API keys, 2026-08-23)

- **Deterministic Proxy simulation** — heuristic scoring (token/numeric overlap) + strict/lenient ±1 + 20% noise. No LLM API called.
- **real LLM API** — requires `FEATHERLESS_API_KEY` / `XAI_API_KEY` / `OPENAI_API_KEY`. Current env has none (only `OPENCODE_GO_API_KEY`).

## Modes (from eval/llm_judge.py header)

1. **Deterministic Proxy (default)** — `python -m eval.llm_judge --input data/synthetic_qa.jsonl --output out/judge_scores.jsonl --sample 10` → $0, ~1ms, 100% reproducible.
2. **Real LLM Judge** — `--llm-provider openai|featherless|grok` + API key → OpenAI-compatible `LLM_JUDGE_PROMPT` → strict JSON (score 1-5 + reason), fallback to proxy on failure.
3. **Reliability simulation vs real** — `--reliability` with noise is simulation; with `--llm-provider` + key it is real 3-call variance.

## Honest 20-sample A vs B (simulation)

- **A strict mean:** 4.00 / **B lenient mean:** 5.00 (correctness/groundedness/completeness), relevance 1.40 vs 2.95, delta +1.00–1.55.
- **Agreement:** 0/20 exact, 20/20 with max_diff >=1 (forced).
- **Reliability 10x3:** mean variance 0.0444, max 0.2222 PASS (<1.0) — simulation jitter only.
- **Log header:** `Judge reliability - 10 samples x 3 runs (seed=42) [SIMULATION]` + `SIMULATION: deterministic-strict vs lenient with noise, not real LLM` + `Real LLM variance requires --llm-provider ...` (stderr also prints `SIMULATION: ... not real LLM`).

## Real Judge Experiment (when API keys available)

```bash
## Real Judge Experiment (when API keys available)
FEATHERLESS_API_KEY=... python -m eval.llm_judge --input data/synthetic_qa.jsonl --llm-provider featherless --model tngtech/deepseek-r1-distill-llama-70b --output out/real_judge_a.jsonl
XAI_API_KEY=... python -m eval.llm_judge --llm-provider grok --output out/real_judge_b.jsonl
python scripts/compare_judges.py out/real_judge_a.jsonl out/real_judge_b.jsonl --output out/real_judge_comparison.md
```

Also valid:

```bash
FEATHERLESS_API_KEY=... python -m eval.llm_judge --input data/synthetic_qa.jsonl --llm-provider featherless --model tngtech/deepseek-r1-distill-llama-70b --output out/real_judge_a.jsonl --sample 20
XAI_API_KEY=... python -m eval.llm_judge --llm-provider grok --model grok-2-latest --output out/real_judge_b.jsonl --sample 20
python scripts/compare_judges.py out/real_judge_a.jsonl out/real_judge_b.jsonl --output out/real_judge_comparison.md

FEATHERLESS_API_KEY=... python -m eval.llm_judge --input data/synthetic_qa.jsonl --llm-provider featherless --output out/real_judge_a.jsonl --sample 50 --reliability --reliability-log out/real_judge_reliability.log
```

See `scripts/compare_judges.py` — auto-labels SIMULATION vs REAL via cost/profile.

## Verification

```bash
Test-Path out/real_judge_comparison.md  # true
Test-Path docs/judge_experiment.md      # true
Select-String -Path out/real_judge_comparison.md -Pattern "Deterministic Proxy.*simulation"  # >=1
Select-String -Path out/real_judge_comparison.md -Pattern "real LLM.*API"                 # >=1
Get-Content out/judge_reliability.log | Select-Object -First 3  # SIMULATION label
```
