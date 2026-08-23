"""Compare two judge output JSONLs (A vs B) and emit markdown comparison.

Usage:
  python scripts/compare_judges.py out/real_judge_a.jsonl out/real_judge_b.jsonl --output out/real_judge_comparison.md
  python scripts/compare_judges.py out/judge_scores.jsonl out/disagreement_cases.jsonl --output out/compare.md --mode simulation

For simulation inputs (deterministic proxy), labels output as SIMULATION.
For real LLM inputs, computes per-metric mean, variance, agreement, cost.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two judge JSONLs (A vs B)")
    parser.add_argument("input_a", help="Judge A JSONL")
    parser.add_argument("input_b", help="Judge B JSONL")
    parser.add_argument("--output", default="out/real_judge_comparison.md", help="Output markdown")
    parser.add_argument("--mode", choices=["auto", "simulation", "real"], default="auto", help="Force label mode")
    args = parser.parse_args()

    path_a = pathlib.Path(args.input_a)
    path_b = pathlib.Path(args.input_b)
    for p in [path_a, path_b]:
        if not p.exists():
            print(f"Input not found: {p}", file=sys.stderr)
            sys.exit(1)

    rows_a = load_jsonl(path_a)
    rows_b = load_jsonl(path_b)

    # Detect simulation: cost 0, profile deterministic-*, latency 1ms
    def is_simulation(rows: list[dict]) -> bool:
        if not rows:
            return True
        costs = [r.get("cost_est", 0) for r in rows]
        profiles = [r.get("profile", "") for r in rows]
        # all costs zero + deterministic profile -> simulation
        return all(c == 0 for c in costs) and any("deterministic" in str(p) for p in profiles)

    sim_a = is_simulation(rows_a)
    sim_b = is_simulation(rows_b)
    simulation = sim_a or sim_b
    if args.mode == "real":
        simulation = False
    elif args.mode == "simulation":
        simulation = True

    # Aggregate per-metric stats
    metrics = ["correctness", "groundedness", "relevance", "completeness"]
    out_lines: list[str] = []
    out_lines.append("# Judge Comparison: A vs B")
    out_lines.append("")
    if simulation:
        out_lines.append("> **Label: Deterministic Proxy SIMULATION - not real LLM**")
        out_lines.append(">")
        out_lines.append("> Current committed scores are heuristic (token/numeric overlap + deterministic-strict vs lenient + noise).")
        out_lines.append("> Real LLM variance requires API keys. See `Real Judge Experiment` section below for reproduction commands.")
    else:
        out_lines.append("> **Label: Real LLM Judge Comparison** (API-backed)")
    out_lines.append("")
    out_lines.append(f"- Input A: `{path_a}` ({len(rows_a)} rows)")
    out_lines.append(f"- Input B: `{path_b}` ({len(rows_b)} rows)")
    out_lines.append(f"- Mode detected: {'SIMULATION (deterministic proxy)' if simulation else 'REAL LLM'}")
    out_lines.append("")

    # Cost/latency summary
    for label, rows in [("A", rows_a), ("B", rows_b)]:
        avg_lat = mean([r.get("latency_ms", 0) for r in rows])
        avg_tok = mean([r.get("tokens_est", 0) for r in rows])
        tot_cost = sum(r.get("cost_est", 0) for r in rows)
        out_lines.append(f"- {label}: avg latency {avg_lat:.1f}ms, avg tokens {avg_tok:.0f}, total cost ${tot_cost:.6f}")

    out_lines.append("")
    out_lines.append("## Per-Metric Means (1-5)")
    out_lines.append("")
    out_lines.append("| Metric | A mean | B mean | Δ (B-A) |")
    out_lines.append("|--------|--------|--------|---------|")
    for m in metrics:
        vals_a = [r.get("scores", {}).get(m, {}).get("score", 0) for r in rows_a if r.get("scores", {}).get(m)]
        vals_b = [r.get("scores", {}).get(m, {}).get("score", 0) for r in rows_b if r.get("scores", {}).get(m)]
        ma = mean([float(v) for v in vals_a]) if vals_a else 0
        mb = mean([float(v) for v in vals_b]) if vals_b else 0
        out_lines.append(f"| {m} | {ma:.2f} | {mb:.2f} | {mb - ma:+.2f} |")

    # Agreement
    if len(rows_a) == len(rows_b) and len(rows_a) > 0:
        disagreements = 0
        max_diffs: list[int] = []
        for ra, rb in zip(rows_a, rows_b, strict=False):
            sa = ra.get("scores", {})
            sb = rb.get("scores", {})
            diffs = []
            for m in metrics:
                s_a = sa.get(m, {}).get("score", 0) if isinstance(sa.get(m), dict) else sa.get(m, 0)
                s_b = sb.get(m, {}).get("score", 0) if isinstance(sb.get(m), dict) else sb.get(m, 0)
                diffs.append(abs(int(s_a) - int(s_b)))
            md = max(diffs) if diffs else 0
            max_diffs.append(md)
            if md >= 1:
                disagreements += 1
        out_lines.append("")
        out_lines.append(f"**Agreement**: {len(rows_a) - disagreements}/{len(rows_a)} exact (max_diff=0), {disagreements} with max_diff >=1")
        if max_diffs:
            out_lines.append(f"Mean max_diff: {mean([float(x) for x in max_diffs]):.2f}, max: {max(max_diffs) if max_diffs else 0}")

    out_lines.append("")
    out_lines.append("## Interpretation")
    out_lines.append("")
    if simulation:
        out_lines.append("- Deterministic proxy scores are intentionally high (5s) when candidate==reference; variance comes only from 20% noise (+-1).")
        out_lines.append("- Strict vs lenient disagreement is forced by +-1 profile shift - cases differ by exactly 1. This is a **stability smoke test**, not model disagreement.")
        out_lines.append("- Real LLM judges show larger variance (temperature, prompt sensitivity) and semantic disagreement; do not extrapolate simulation numbers to LLM behavior.")
        out_lines.append("- Threshold PASS (<1.0 variance) here means noise is small; real LLM variance may exceed this.")
    else:
        out_lines.append("- Real LLM means reflect actual model judgment; compare cost/latency tradeoffs.")
        out_lines.append("- Large disagreement suggests prompt/model sensitivity — inspect disagreement_cases for qualitative review.")

    out_lines.append("")
    out_lines.append("## Real Judge Experiment (when API keys available)")
    out_lines.append("")
    out_lines.append("Current committed artifacts are **Deterministic Proxy simulation**. To run a real A vs B experiment (20-50 samples, two different judge models):")
    out_lines.append("")
    out_lines.append("```bash")
    out_lines.append("# Option 1: Two Featherless/Grok models (requires keys)")
    out_lines.append("FEATHERLESS_API_KEY=... python -m eval.llm_judge --input data/synthetic_qa.jsonl --llm-provider featherless --model tngtech/deepseek-r1-distill-llama-70b --output out/real_judge_a.jsonl --sample 20")
    out_lines.append("XAI_API_KEY=... python -m eval.llm_judge --llm-provider grok --model grok-2-latest --output out/real_judge_b.jsonl --sample 20")
    out_lines.append("python scripts/compare_judges.py out/real_judge_a.jsonl out/real_judge_b.jsonl --output out/real_judge_comparison.md")
    out_lines.append("")
    out_lines.append("# Option 2: OpenAI vs Featherless")
    out_lines.append("OPENAI_API_KEY=... python -m eval.llm_judge --input data/synthetic_qa.jsonl --llm-provider openai --model gpt-4o-mini --output out/real_judge_a.jsonl --sample 30")
    out_lines.append("FEATHERLESS_API_KEY=... python -m eval.llm_judge --llm-provider featherless --model meta-llama/Meta-Llama-3.1-8B-Instruct --output out/real_judge_b.jsonl --sample 30")
    out_lines.append("python scripts/compare_judges.py out/real_judge_a.jsonl out/real_judge_b.jsonl --output out/real_judge_comparison.md")
    out_lines.append("")
    out_lines.append("# Option 3: Same provider, different temperatures (requires code tweak to expose temperature)")
    out_lines.append("# Or: deterministic strict vs lenient as baseline, then replace with real runs and compare variance:")
    out_lines.append("FEATHERLESS_API_KEY=... python -m eval.llm_judge --input data/synthetic_qa.jsonl --llm-provider featherless --output out/real_judge_a.jsonl --sample 50 --reliability --reliability-log out/real_judge_reliability.log")
    out_lines.append("```")
    out_lines.append("")
    out_lines.append("Cost estimate: gpt-4o-mini ~$0.002/1k tokens, Featherless ~$0.001/1k, Grok ~$0.005/1k. 30 samples x ~80 tokens ~ 2.4k tokens -> <$0.02 total.")
    out_lines.append("")
    out_lines.append("When keys are unavailable (current CI), keep this file as **honest simulation label** — do not claim real LLM results.")
    out_lines.append("")

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote comparison → {out_path} ({'SIMULATION' if simulation else 'REAL'})")
    print("\n".join(out_lines[:30]))


if __name__ == "__main__":
    main()
