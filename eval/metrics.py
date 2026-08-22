"""Retrieval & faithfulness metrics — Hit@5, Faithfulness, Failure rate.

Offline 계산: out/rule_report.json + data/synthetic_qa.jsonl 사용.
Faithfulness: citation overlap 기반 (답변 토큰/source 토큰, numeric token grounding).
Hit@5: gold doc_id가 retrieved top-5 안에 있는지 (synthetic 환경에서는
       rule 실패를 retrieval miss로 proxy, 실제 retrieval 로그가 있으면 교차 검증).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import re
import sys

_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def hit_at_k(retrieved_ids: list[str], gold_id: str, k: int = 5) -> bool:
    """Gold id가 retrieved top-k 안에 있는지."""
    return gold_id in retrieved_ids[:k]


def hit_at_5(retrieved_ids: list[str], gold_id: str) -> bool:
    return hit_at_k(retrieved_ids, gold_id, k=5)


def compute_hit_at_5(entries: list[dict], retrieved_map: dict[int, list[str]] | None = None, k: int = 5) -> dict:
    """Hit@k 계산.

    retrieved_map이 없으면 offline proxy:
      expected_keyword pass == 1, fail == 0 으로 Hit@k 근사.
    retrieved_map이 있으면 실제 retrieval 결과로 계산.
    """
    from eval.rule_filter import check_expected_keyword

    total = len(entries)
    if total == 0:
        return {"hit_at_k": 0.0, "hits": 0, "total": 0, "k": k, "mode": "empty"}

    if retrieved_map is not None:
        hits = 0
        for idx, entry in enumerate(entries):
            gold = entry.get("source_chunks", [{}])[0].get("doc_id", "") if entry.get("source_chunks") else ""
            retrieved = retrieved_map.get(idx, [])
            if hit_at_k(retrieved, gold, k=k):
                hits += 1
        rate = hits / total if total else 0.0
        return {"hit_at_k": round(rate, 4), "hits": hits, "total": total, "k": k, "mode": "retrieval"}

    # Proxy: expected_keyword pass → hit
    hits = 0
    for entry in entries:
        r = check_expected_keyword(entry)
        if r.passed:
            hits += 1
    rate = hits / total if total else 0.0
    return {"hit_at_k": round(rate, 4), "hits": hits, "total": total, "k": k, "mode": "proxy-expected_keyword"}


def faithfulness_score(answer: str, source_text: str) -> float:
    """Faithfulness: answer 토큰이 source 안에 몇 % 존재하는가.

    Numeric token은 별도로 100% grounding 필요 — numeric grounding 실패 시 penalty.
    반환 0.0~1.0
    """
    if not answer.strip():
        return 0.0
    if not source_text.strip():
        return 0.0

    # Numeric grounding
    numeric_tokens = re.findall(r"\d+(?:\.\d+)?", answer)
    if numeric_tokens:
        missing = [n for n in numeric_tokens if n not in source_text]
        ratio_num = (len(numeric_tokens) - len(missing)) / len(numeric_tokens) if missing else 1.0
    else:
        ratio_num = 1.0

    # Token overlap
    ans_toks = [t for t in _tokens(answer) if len(t) >= 2]
    if not ans_toks:
        return ratio_num  # only numeric matters
    source_set = set(_tokens(source_text))
    matched = sum(1 for t in ans_toks if t in source_set)
    ratio_tok = matched / len(ans_toks) if ans_toks else 1.0

    # Blended: numeric 50%, token 50% (if numeric exists), else pure token
    if numeric_tokens:
        return round(0.5 * ratio_num + 0.5 * ratio_tok, 4)
    return round(ratio_tok, 4)


def compute_faithfulness(entries: list[dict]) -> dict:
    """전체 entries의 faithfulness 평균."""
    if not entries:
        return {"faithfulness": 0.0, "count": 0}
    scores: list[float] = []
    for entry in entries:
        ans = entry.get("reference_answer", "")
        source_text = " ".join([c.get("text", "") for c in entry.get("source_chunks", []) if isinstance(c, dict)])
        scores.append(faithfulness_score(ans, source_text))
    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "faithfulness": round(avg, 4),
        "count": len(scores),
        "scores": scores,
        "min": round(min(scores), 4) if scores else 0.0,
        "max": round(max(scores), 4) if scores else 0.0,
    }


def compute_failure_rate(report: dict) -> dict:
    """rule_report.json 기반 failure rate."""
    total = report.get("total", 0)
    failed = report.get("failed", 0)
    passed = report.get("passed", 0)
    rate = failed / total if total else 0.0
    taxonomy = report.get("taxonomy", {})
    by_rule = {k: len(v) for k, v in taxonomy.items()} if isinstance(taxonomy, dict) else {}
    return {
        "failure_rate": round(rate, 4),
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "by_rule": by_rule,
    }


def evaluate_all(entries: list[dict], report: dict | None = None) -> dict:
    """종합 metrics 계산."""
    hit = compute_hit_at_5(entries, k=5)
    faith = compute_faithfulness(entries)
    if report is None:
        # entries만으로 failure proxy
        from eval.rule_filter import evaluate_entries

        report = evaluate_entries(entries)
    fail = compute_failure_rate(report)
    return {
        "hit_at_5": hit,
        "faithfulness": faith,
        "failure_rate": fail,
    }


def main() -> None:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Retrieval/quality metrics (Hit@5, Faithfulness, Failure rate)")
    parser.add_argument("--qa", type=str, default="data/synthetic_qa.jsonl", help="Synthetic QA JSONL")
    parser.add_argument("--report", type=str, default="out/rule_report.json", help="Rule report JSON")
    parser.add_argument("--output", type=str, default="out/metrics_report.json", help="Output metrics JSON")
    args = parser.parse_args()

    qa_path = pathlib.Path(args.qa)
    report_path = pathlib.Path(args.report)
    out_path = pathlib.Path(args.output)

    if not qa_path.exists():
        raise SystemExit(f"QA not found: {qa_path}")

    entries: list[dict] = []
    with qa_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    report: dict | None = None
    if report_path.exists():
        with report_path.open(encoding="utf-8") as f:
            report = json.load(f)
    else:
        print(f"[metrics] report not found: {report_path} — computing from entries", file=sys.stderr)

    result = evaluate_all(entries, report=report)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, indent=2)

    print(f"Metrics computed over {len(entries)} entries")
    print(f"  Hit@5: {result['hit_at_5']['hit_at_k']:.2%} ({result['hit_at_5']['hits']}/{result['hit_at_5']['total']}) mode={result['hit_at_5']['mode']}")
    print(f"  Faithfulness: {result['faithfulness']['faithfulness']:.4f} (min {result['faithfulness']['min']:.2f} max {result['faithfulness']['max']:.2f})")
    print(f"  Failure rate: {result['failure_rate']['failure_rate']:.2%} (passed {result['failure_rate']['passed']}/{result['failure_rate']['total']})")
    if result["failure_rate"]["by_rule"]:
        print(f"  By rule: {result['failure_rate']['by_rule']}")
    print(f"Report → {out_path}")


if __name__ == "__main__":
    main()
