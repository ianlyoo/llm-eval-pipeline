"""LLM-as-a-Judge - 4 metrics (correctness/groundedness/relevance/completeness) 1-5 + reason.

==============================================================================
 EVALUATION MODES  (honest labeling - do not misrepresent proxy as real LLM)
==============================================================================
 1) Deterministic Proxy (DEFAULT, no API key)
    - Heuristic: token/numeric overlap → 1-5 per metric (correctness,
      groundedness, relevance, completeness) + textual reason.
    - Zero cost/latency, fully reproducible (seed 42). Used in CI/offline.
    - Output profile: "default" | strict/lenient shifts for A-vs-B sim.

 2) Real LLM Judge (requires API key)
    - --llm-provider openai|featherless|grok  + env API key triggers
      OpenAI-compatible chat completion (LLM_JUDGE_PROMPT → strict JSON).
    - Tries LLM first; on failure/no key falls back to Deterministic Proxy
      with stderr warning "[llm_judge] ... fallback deterministic".
    - Cost/latency/tokens are real LLM values when LLM succeeds.

 3) Reliability / Agreement Simulation vs Real
    - SIMULATION (current committed artifacts): --reliability with
      --model-a deterministic-strict vs --model-b deterministic-lenient
      + 3-run noise (random ±1 with 20% prob) → variance <1 PASS is
      SIMULATION variance, NOT real multi-model variance.
      Log MUST include: "SIMULATION: deterministic-strict vs lenient
      with noise, not real LLM" (see reliability log header).
    - REAL: when --llm-provider is set with valid keys, the same
      --reliability flag runs 3 real LLM calls (temperature 0.2) and
      logs real variance/disagreement. See out/real_judge_comparison.md
      for reproduction commands.

Output schema per line:
  {question, candidate_answer, reference_answer, source_chunks,
   scores: {correctness:{score,reason}, groundedness:..., relevance:..., completeness:...},
   latency_ms, tokens_est, cost_est, run_id, profile}
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

# Re-export existing scorer entry points for callers of eval.llm_judge.
from eval.judge_inputs import resolve_candidate, resolve_sources
from eval.judge_proxy import (  # noqa: F401 — preserve scorer imports from this module
    _clamp_score,
    deterministic_judge,
    score_completeness,
    score_correctness,
    score_groundedness,
    score_relevance,
)

# ---------------------------------------------------------------------------
# Variance / disagreement helpers (exposed for tests)
# ---------------------------------------------------------------------------

def compute_variance(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = sum(values) / len(values)
    var = sum((x - m) ** 2 for x in values) / len(values)  # population
    return var


def detect_disagreement(scores_a: dict, scores_b: dict, threshold: int = 1) -> dict:
    """Compare two score dicts ({metric: {score}}) and return diff details."""
    diffs: dict[str, int] = {}
    is_disagreement = False
    for k in ["correctness", "groundedness", "relevance", "completeness"]:
        sa = scores_a.get(k, {}).get("score", 0) if isinstance(scores_a.get(k), dict) else scores_a.get(k, 0)
        sb = scores_b.get(k, {}).get("score", 0) if isinstance(scores_b.get(k), dict) else scores_b.get(k, 0)
        d = abs(int(sa) - int(sb))
        diffs[k] = d
        if d >= threshold:
            is_disagreement = True
    return {"is_disagreement": is_disagreement, "diffs": diffs, "max_diff": max(diffs.values()) if diffs else 0}


# ---------------------------------------------------------------------------
# LLM prompt + try LLM judge
# ---------------------------------------------------------------------------
LLM_JUDGE_PROMPT = """You are an LLM-as-a-Judge for RAG evaluation. Score 4 metrics 1-5 with reasons.

QUESTION: {question}
REFERENCE_ANSWER: {reference}
SOURCE_CHUNKS: {source_text}
CANDIDATE_ANSWER: {candidate}

Task: For each metric, give integer 1-5 and a concise reason string (no empty reasons).
Metrics:
- correctness: reference vs candidate numeric/key-token match
- groundedness: candidate numeric tokens present in source
- relevance: question keyword overlap with candidate
- completeness: reference key tokens coverage in candidate

Output STRICT JSON only:
{{"correctness": {{"score": 1, "reason": "..."}}, "groundedness": {{"score": 1, "reason": "..."}}, "relevance": {{"score": 1, "reason": "..."}}, "completeness": {{"score": 1, "reason": "..."}}}}
"""

LLM_PRICES = {
    "openai": 0.002,  # $ per 1k tokens estimate
    "featherless": 0.001,
    "grok": 0.005,
}


def try_llm_judge(
    entry: dict,
    provider: str,
    api_key: str | None,
    model: str | None = None,
) -> dict | None:
    """Attempt real LLM judge; return None on failure/no key to trigger fallback."""
    if provider in ("none", "", None):
        return None
    import os

    env_key = {
        "openai": "OPENAI_API_KEY",
        "featherless": "FEATHERLESS_API_KEY",
        "grok": "XAI_API_KEY",
    }.get(provider, "OPENAI_API_KEY")
    key = api_key or os.environ.get(env_key, "")
    if not key:
        print(f"[llm_judge] No API key for {provider} ({env_key}) - fallback deterministic.", file=sys.stderr)
        return None

    question = entry.get("question", "")
    reference = entry.get("reference_answer", "")
    candidate = resolve_candidate(entry)
    source_chunks = resolve_sources(entry)
    source_text = " ".join([c.get("text", "")[:500] for c in source_chunks if isinstance(c, dict)])[:2000]

    prompt = LLM_JUDGE_PROMPT.format(
        question=question, reference=reference, source_text=source_text, candidate=candidate
    )

    # Try openai client
    try:
        from openai import OpenAI  # type: ignore

        base_map = {
            "openai": None,
            "featherless": "https://api.featherless.ai/v1",
            "grok": "https://api.x.ai/v1",
        }
        base = base_map.get(provider)
        default_model = {
            "openai": "gpt-4o-mini",
            "featherless": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "grok": "grok-2-latest",
        }.get(provider, "gpt-4o-mini")
        mdl = model or default_model
        kwargs = {"api_key": key}
        if base:
            kwargs["base_url"] = base
        client = OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=mdl,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
        )
        content = resp.choices[0].message.content or ""
        # cost tokens from usage if available
        tokens_est = None
        try:
            usage = resp.usage
            if usage:
                tokens_est = (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
        except Exception:
            pass
        # Parse JSON
        # Extract JSON object
        import re as _re

        m = _re.search(r"\{.*\}", content, _re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            # Validate has 4 metrics with score+reason
            for k in ["correctness", "groundedness", "relevance", "completeness"]:
                if k not in obj or "score" not in obj[k] or "reason" not in obj[k]:
                    raise ValueError(f"LLM missing {k}")
                if not obj[k]["reason"] or not obj[k]["reason"].strip():
                    raise ValueError(f"LLM empty reason for {k}")
                obj[k]["score"] = _clamp_score(int(obj[k]["score"]))
            # stash tokens for caller
            obj["_tokens_est"] = tokens_est or len(prompt.split()) + len(content.split())
            return obj
        return None
    except Exception as e:
        print(f"[llm_judge] LLM call failed ({e}) - fallback deterministic.", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Single entry judge with timing/cost
# ---------------------------------------------------------------------------

def judge_entry(
    entry: dict,
    profile: str = "default",
    llm_provider: str = "none",
    api_key: str | None = None,
    model: str | None = None,
    noise_rng: random.Random | None = None,
    run_id: str = "run-0",
) -> dict:
    start = time.perf_counter()
    # Try LLM first
    llm_scores = None
    tokens_est = 0
    if llm_provider != "none":
        llm_scores = try_llm_judge(entry, llm_provider, api_key, model)
    if llm_scores is not None:
        # Extract tokens if LLM provided
        tokens_est = int(llm_scores.pop("_tokens_est", 0)) or _estimate_tokens(entry)
        scores = {k: llm_scores[k] for k in ["correctness", "groundedness", "relevance", "completeness"]}
        cost_est = tokens_est / 1000 * LLM_PRICES.get(llm_provider, 0.002)
    else:
        scores = deterministic_judge(entry, profile=profile, noise_rng=noise_rng)
        tokens_est = _estimate_tokens(entry)
        cost_est = 0.0

    elapsed = time.perf_counter() - start
    latency_ms = int(elapsed * 1000)
    # ensure minimal latency for logging (avoid 0)
    if latency_ms == 0:
        latency_ms = 1

    question = entry.get("question", "")
    candidate = resolve_candidate(entry)

    return {
        "question": question,
        "candidate_answer": candidate,
        "reference_answer": entry.get("reference_answer", ""),
        "source_chunks": entry.get("source_chunks", []),
        "scores": scores,
        "latency_ms": latency_ms,
        "tokens_est": tokens_est,
        "cost_est": round(cost_est, 6),
        "run_id": run_id,
        "profile": profile,
        "category": entry.get("category", ""),
        "difficulty": entry.get("difficulty", ""),
    }


def _estimate_tokens(entry: dict) -> int:
    # Keep estimates for existing string/missing candidates unchanged. Invalid
    # candidates use the same reference fallback as the scorer and prompt.
    candidate = entry.get("candidate_answer", "")
    if not isinstance(candidate, str):
        candidate = resolve_candidate(entry)
    text = (
        entry.get("question", "")
        + " " + entry.get("reference_answer", "")
        + " " + candidate
        + " " + " ".join([c.get("text", "")[:200] for c in resolve_sources(entry)])
    )
    # rough: 1 token ≈ 4 chars
    return max(10, len(text) // 4)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import contextlib

    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="LLM-as-a-Judge (4 metrics 1-5 + reason)")
    parser.add_argument("--input", required=True, help="Input JSONL (synthetic_qa.jsonl)")
    parser.add_argument("--output", default="out/judge_scores.jsonl", help="Output judge_scores JSONL")
    parser.add_argument("--sample", type=int, default=10, help="Number of samples (random seed 42)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--reliability", action="store_true", help="Enable judge_reliability mode (3 runs + 2-model disagreement)")
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="none",
        choices=["none", "openai", "featherless", "grok"],
        help="LLM provider (default none=deterministic)",
    )
    parser.add_argument("--api-key", type=str, default=None, help="API key override")
    parser.add_argument("--model", type=str, default=None, help="Model override")
    parser.add_argument("--model-a", type=str, default="deterministic-strict", help="Model A profile for disagreement")
    parser.add_argument("--model-b", type=str, default="deterministic-lenient", help="Model B profile for disagreement")
    parser.add_argument("--disagreement-output", type=str, default="out/disagreement_cases.jsonl", help="Disagreement output")
    parser.add_argument("--reliability-log", type=str, default="out/judge_reliability.log", help="Reliability log file")
    args = parser.parse_args()

    in_path = pathlib.Path(args.input)
    out_path = pathlib.Path(args.output)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    entries: list[dict] = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if len(entries) == 0:
        raise SystemExit("No entries loaded")

    rng = random.Random(args.seed)
    sample_n = min(args.sample, len(entries))
    indices = rng.sample(range(len(entries)), k=sample_n)
    sampled = [(indices[i], entries[indices[i]]) for i in range(sample_n)]

    # Sort by original index for reproducibility display? keep sampled order
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.reliability:
        # Simple single-run mode
        with out_path.open("w", encoding="utf-8") as out:
            for idx, entry in sampled:
                res = judge_entry(
                    entry,
                    profile="default",
                    llm_provider=args.llm_provider,
                    api_key=args.api_key,
                    model=args.model,
                    run_id=f"sample-{idx}",
                )
                # Validate reason present
                for k, v in res["scores"].items():
                    if not v.get("reason") or not v["reason"].strip():
                        raise SystemExit(f"FAIL: empty reason for {k} idx {idx}")
                out.write(json.dumps(res, ensure_ascii=False) + "\n")
        print(f"Judged {sample_n} samples → {out_path}")
        return

    # Reliability mode: 3 runs + 2-model comparison
    # Collect per-sample per-run scores
    all_runs: list[dict] = []
    # per-sample list of 3 run scores per metric
    per_sample_runs: dict[int, list[dict]] = {idx: [] for idx, _ in sampled}

    for run_idx in range(3):
        for idx, entry in sampled:
            noise_seed = args.seed * 100 + run_idx * 10 + idx
            noise_rng = random.Random(noise_seed)
            res = judge_entry(
                entry,
                profile="default",
                llm_provider=args.llm_provider,
                api_key=args.api_key,
                model=args.model,
                noise_rng=noise_rng,
                run_id=f"run-{run_idx}-sample-{idx}",
            )
            all_runs.append(res)
            per_sample_runs[idx].append(res["scores"])

    # Write judge_scores: all 3*10 lines
    with out_path.open("w", encoding="utf-8") as out:
        for r in all_runs:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute variance per sample per metric
    log_path = pathlib.Path(args.reliability_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    # Label simulation vs real at top - honest distinction (P1-EXPERIMENT)
    is_simulation = args.llm_provider == "none"
    if is_simulation:
        lines.append(f"Judge reliability - {sample_n} samples x 3 runs (seed={args.seed}) [SIMULATION]")
        lines.append("SIMULATION: deterministic-strict vs lenient with noise, not real LLM (no API key; deterministic proxy)")
        lines.append("Real LLM variance requires --llm-provider openai|featherless|grok + API key; see out/real_judge_comparison.md")
    else:
        lines.append(f"Judge reliability - {sample_n} samples x 3 runs (seed={args.seed}) [REAL LLM: provider={args.llm_provider} model={args.model or 'default'}]")
        lines.append("Mode: real LLM judge (API key present) - variance is real multi-call variance, not simulation")
    # Also log to stderr for visibility
    if is_simulation:
        print("[llm_judge] SIMULATION: deterministic-strict vs lenient with noise, not real LLM", file=sys.stderr)
    lines.append(f"Input: {in_path}  Profiles: default with noise (temp simulation)" if is_simulation else f"Input: {in_path}  Provider: {args.llm_provider}  Model: {args.model or 'default'}")
    lines.append("")

    all_variances: list[float] = []
    for idx, entry in sampled:
        runs = per_sample_runs[idx]
        # per metric variance
        for metric in ["correctness", "groundedness", "relevance", "completeness"]:
            vals = [r[metric]["score"] for r in runs]
            var = compute_variance([float(v) for v in vals])
            all_variances.append(var)
        # avg per sample
        vals_all = []
        for r in runs:
            for m in ["correctness", "groundedness", "relevance", "completeness"]:
                vals_all.append(float(r[m]["score"]))
        # compute per-metric lines
        q_short = entry.get("question", "")[:60].replace("\n", " ")
        lines.append(f"Sample idx={idx} q=\"{q_short}\"")
        for metric in ["correctness", "groundedness", "relevance", "completeness"]:
            vals = [r[metric]["score"] for r in runs]
            var = compute_variance([float(v) for v in vals])
            lines.append(f"  {metric}: scores {vals} var={var:.4f}")
        # also log latencies
        lat_vals = [r["latency_ms"] for r in all_runs if r["question"] == entry.get("question")]
        if lat_vals:
            lines.append(f"  latency_ms avg={sum(lat_vals)/len(lat_vals):.1f} tokens_est avg={sum(r['tokens_est'] for r in all_runs if r['question']==entry.get('question'))/len(lat_vals):.0f}")
        lines.append("")

    overall_var = sum(all_variances) / len(all_variances) if all_variances else 0.0
    max_var = max(all_variances) if all_variances else 0.0
    lines.append(f"Overall mean variance: {overall_var:.4f}  max: {max_var:.4f}")
    if overall_var < 1.0 and max_var < 1.0:
        lines.append("Variance check PASS (<1.0)")
    else:
        lines.append("Variance check FAIL (>=1.0) - investigate judge stability")
    lines.append("")

    # Disagreement between model-a and model-b (no noise)
    disagreement_cases: list[dict] = []
    for idx, entry in sampled:
        res_a = judge_entry(entry, profile=args.model_a, llm_provider="none", run_id=f"model-a-{idx}")
        res_b = judge_entry(entry, profile=args.model_b, llm_provider="none", run_id=f"model-b-{idx}")
        disc = detect_disagreement(res_a["scores"], res_b["scores"], threshold=1)
        if disc["is_disagreement"]:
            disagreement_cases.append(
                {
                    "question": entry.get("question", ""),
                    "reference_answer": entry.get("reference_answer", ""),
                    "candidate_answer": entry.get("candidate_answer", entry.get("reference_answer", "")),
                    "source_chunks": entry.get("source_chunks", []),
                    "model_a": args.model_a,
                    "model_b": args.model_b,
                    "scores_a": res_a["scores"],
                    "scores_b": res_b["scores"],
                    "diffs": disc["diffs"],
                    "max_diff": disc["max_diff"],
                    "index": idx,
                }
            )

    lines.append(f"Disagreement: {len(disagreement_cases)}/{sample_n} cases with max_diff >=1 (profiles {args.model_a} vs {args.model_b})")

    # Write disagreement file
    dis_path = pathlib.Path(args.disagreement_output)
    dis_path.parent.mkdir(parents=True, exist_ok=True)
    with dis_path.open("w", encoding="utf-8") as f:
        for c in disagreement_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Log costs
    avg_latency = sum(r["latency_ms"] for r in all_runs) / len(all_runs) if all_runs else 0
    avg_tokens = sum(r["tokens_est"] for r in all_runs) / len(all_runs) if all_runs else 0
    total_cost = sum(r["cost_est"] for r in all_runs)
    lines.append(f"Cost/latency: avg latency {avg_latency:.1f}ms, avg tokens {avg_tokens:.0f}, total cost ${total_cost:.6f}")
    lines.append(f"Outputs: {out_path} ({len(all_runs)} lines), {dis_path} ({len(disagreement_cases)} lines)")

    log_text = "\n".join(lines)
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(log_text + "\n")
    print(log_text)
    print(f"\nReliability log → {log_path}")
    print(f"Disagreement cases → {dis_path} ({len(disagreement_cases)})")


if __name__ == "__main__":
    main()
