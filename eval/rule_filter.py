"""Rule-based validation for synthetic QA — 6 checks.

Rules:
 1. citation_exists — source_chunks has doc_id/chunk_id
 2. source_coverage — source_chunks length >=1
 3. empty_answer — reference_answer not empty
 4. formatting_violation — too short / structure broken
 5. unsupported_answer — numeric tokens in answer must appear in source_chunks
 6. expected_keyword — question keyword appears in answer
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import re
import sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RuleResult:
    rule: str
    passed: bool
    reason: str


# Tokenizer (shared with pipeline)
_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]+", re.UNICODE)

# Korean particle suffixes to strip for keyword normalization
_KO_PARTICLES = [
    "에서",
    "에게",
    "한테",
    "부터",
    "까지",
    "으로",
    "로서",
    "로써",
    "으로서",
    "으로써",
    "와",
    "과",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "에",
    "로",
    "도",
    "만",
    "조차",
    "마저",
    "이다",
    "입니다",
    "인가",
    "인가요",
    "이며",
    "이고",
    "라는",
    "이라는",
]

# Stopwords for keyword extraction (lowercased)
_STOPWORDS = {
    "what",
    "is",
    "the",
    "and",
    "are",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "to",
    "with",
    "how",
    "when",
    "where",
    "why",
    "who",
    "does",
    "do",
    "did",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "was",
    "were",
    "will",
    "would",
    "can",
    "could",
    "should",
    "mean",
    "means",
    "or",
    "but",
    "if",
    "then",
    "than",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "by",
    "as",
    "at",
    "from",
    "up",
    "out",
    "about",
    "into",
    "over",
    "after",
    "before",
    "under",
    "again",
    "further",
    "무엇인가",
    "무엇",
    "얼마인가",
    "얼마",
    "어떻게",
    "언제",
    "어디",
    "누가",
    "왜",
    "인가",
    "인가요",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "의",
    "로",
    "과",
    "와",
    "하다",
    "한다",
    "있나",
    "있나요",
    "되는가",
    "있는가",
    "관한",
    "대한",
    "대해",
    "대하여",
    "관련",
    "질문",
    "답변",
    "정책",
    "무슨",
    "어느",
    "어떤",
}


def _strip_ko_particle(token: str) -> str:
    """Strip trailing Korean particle suffix iteratively."""
    # Sort by length desc to match longer first
    for p in sorted(_KO_PARTICLES, key=len, reverse=True):
        if token.endswith(p) and len(token) > len(p):
            return token[: -len(p)]
    return token


def _normalize_keyword(kw: str) -> str:
    kw = kw.lower()
    # Strip particles after lowercasing
    stripped = _strip_ko_particle(kw)
    return stripped


# ---------------------------------------------------------------------------
# Rule implementations
# ---------------------------------------------------------------------------

def check_citation_exists(entry: dict) -> RuleResult:
    """Rule 1: citation exists — source_chunks entries have doc_id/chunk_id."""
    sc = entry.get("source_chunks")
    if not isinstance(sc, list) or len(sc) == 0:
        return RuleResult(
            rule="citation_exists",
            passed=False,
            reason="source_chunks missing or empty — no citation",
        )
    for i, c in enumerate(sc):
        if not isinstance(c, dict):
            return RuleResult(
                rule="citation_exists",
                passed=False,
                reason=f"source_chunks[{i}] not a dict",
            )
        if not c.get("doc_id") or not c.get("chunk_id"):
            return RuleResult(
                rule="citation_exists",
                passed=False,
                reason=f"source_chunks[{i}] missing doc_id/chunk_id",
            )
        # text field also required for provenance
        if not c.get("text"):
            return RuleResult(
                rule="citation_exists",
                passed=False,
                reason=f"source_chunks[{i}] missing text",
            )
    return RuleResult(rule="citation_exists", passed=True, reason="citation present")


def check_source_coverage(entry: dict) -> RuleResult:
    """Rule 2: source_coverage >=1."""
    sc = entry.get("source_chunks")
    if not isinstance(sc, list):
        return RuleResult(
            rule="source_coverage",
            passed=False,
            reason="source_chunks not a list",
        )
    if len(sc) < 1:
        return RuleResult(
            rule="source_coverage",
            passed=False,
            reason="source_chunks empty — coverage 0",
        )
    return RuleResult(
        rule="source_coverage",
        passed=True,
        reason=f"coverage {len(sc)} chunk(s)",
    )


def check_empty_answer(entry: dict) -> RuleResult:
    """Rule 3: empty answer detection."""
    ans = entry.get("reference_answer")
    if ans is None or not isinstance(ans, str):
        return RuleResult(
            rule="empty_answer",
            passed=False,
            reason="reference_answer missing or not string",
        )
    if ans.strip() == "":
        return RuleResult(
            rule="empty_answer",
            passed=False,
            reason="reference_answer empty/whitespace",
        )
    return RuleResult(rule="empty_answer", passed=True, reason="answer non-empty")


def check_formatting_violation(entry: dict) -> RuleResult:
    """Rule 4: formatting violation — too short / structure broken."""
    ans = entry.get("reference_answer", "")
    if not isinstance(ans, str):
        return RuleResult(
            rule="formatting_violation",
            passed=False,
            reason="reference_answer not string",
        )
    stripped = ans.strip()
    if len(stripped) < 10:
        return RuleResult(
            rule="formatting_violation",
            passed=False,
            reason=f"too short: {len(stripped)} chars <10",
        )
    # Token count check: at least 2 tokens meaningful
    toks = _TOKEN_RE.findall(stripped)
    if len(toks) < 2:
        return RuleResult(
            rule="formatting_violation",
            passed=False,
            reason=f"too few tokens: {len(toks)} <2",
        )
    return RuleResult(rule="formatting_violation", passed=True, reason="formatting ok")


def check_unsupported_answer(entry: dict) -> RuleResult:
    """Rule 5: unsupported answer — numeric tokens must appear in source_chunks."""
    ans = entry.get("reference_answer", "")
    if not isinstance(ans, str):
        return RuleResult(
            rule="unsupported_answer",
            passed=False,
            reason="reference_answer not string",
        )
    if ans.strip() == "":
        # Empty already caught by empty_answer, but also unsupported
        return RuleResult(
            rule="unsupported_answer",
            passed=False,
            reason="empty answer — unsupported",
        )
    # Extract numeric tokens (digit sequences)
    numeric_tokens = re.findall(r"\d+", ans)
    if not numeric_tokens:
        return RuleResult(
            rule="unsupported_answer",
            passed=True,
            reason="no numeric tokens — trivially supported",
        )
    sc = entry.get("source_chunks", [])
    if not isinstance(sc, list) or len(sc) == 0:
        return RuleResult(
            rule="unsupported_answer",
            passed=False,
            reason="no source_chunks to verify numeric coverage",
        )
    source_text = " ".join([c.get("text", "") for c in sc if isinstance(c, dict)])
    # Check 100% numeric coverage
    missing = [n for n in numeric_tokens if n not in source_text]
    if missing:
        return RuleResult(
            rule="unsupported_answer",
            passed=False,
            reason=f"numeric tokens not in source: {missing}",
        )
    return RuleResult(
        rule="unsupported_answer",
        passed=True,
        reason=f"all {len(numeric_tokens)} numeric tokens found in source",
    )


def check_expected_keyword(entry: dict) -> RuleResult:
    """Rule 6: expected keyword — question keyword appears in answer."""
    q = entry.get("question", "")
    ans = entry.get("reference_answer", "")
    if not isinstance(q, str) or not isinstance(ans, str):
        return RuleResult(
            rule="expected_keyword",
            passed=False,
            reason="question/reference_answer not string",
        )
    if q.strip() == "" or ans.strip() == "":
        return RuleResult(
            rule="expected_keyword",
            passed=False,
            reason="question or answer empty",
        )
    q_tokens = [t.lower() for t in _TOKEN_RE.findall(q)]
    # Filter
    keywords = []
    for tok in q_tokens:
        if tok in _STOPWORDS:
            continue
        if len(tok) < 2:
            continue
        # Strip particle for Korean
        norm = _normalize_keyword(tok)
        if len(norm) < 2:
            continue
        if norm in _STOPWORDS:
            continue
        keywords.append(norm)
    # Deduplicate preserve order
    seen: set[str] = set()
    uniq_keywords: list[str] = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            uniq_keywords.append(k)
    if not uniq_keywords:
        return RuleResult(
            rule="expected_keyword",
            passed=True,
            reason="no extractable keywords — vacuously pass",
        )
    ans_low = ans.lower()
    matched = [kw for kw in uniq_keywords if kw in ans_low]
    if not matched:
        return RuleResult(
            rule="expected_keyword",
            passed=False,
            reason=f"no keyword overlap: question keywords {uniq_keywords} not in answer",
        )
    return RuleResult(
        rule="expected_keyword",
        passed=True,
        reason=f"matched keywords {matched}",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

ALL_RULES = [
    check_citation_exists,
    check_source_coverage,
    check_empty_answer,
    check_formatting_violation,
    check_unsupported_answer,
    check_expected_keyword,
]


def run_all_rules(entry: dict) -> list[RuleResult]:
    """Run all 6 rules on entry."""
    return [fn(entry) for fn in ALL_RULES]


def classify_failures(entries: list[dict]) -> dict:
    """Classify failures by rule.

    Returns taxonomy: {rule_name: [ {index, reason, question} ] }
    """
    taxonomy: dict[str, list[dict]] = {fn.__name__.replace("check_", ""): [] for fn in ALL_RULES}
    for idx, entry in enumerate(entries):
        results = run_all_rules(entry)
        for r in results:
            if not r.passed:
                taxonomy[r.rule].append(
                    {
                        "index": idx,
                        "reason": r.reason,
                        "question": entry.get("question", "")[:120],
                    }
                )
    # Remove empty rule entries for cleaner output? Keep all per spec but filter empty for report taxonomy
    # Spec says taxonomy: {rule_name: [failed indices/reasons]}
    # Keep only non-empty for brevity but include all keys if needed
    # We'll return only non-empty to avoid clutter; caller can include all
    compact = {k: v for k, v in taxonomy.items() if v}
    return compact


def evaluate_entries(entries: list[dict]) -> dict:
    """Evaluate entries and build report dict."""
    total = len(entries)
    failures: list[dict] = []
    passed_count = 0

    for idx, entry in enumerate(entries):
        results = run_all_rules(entry)
        failed = [r for r in results if not r.passed]
        if not failed:
            passed_count += 1
        else:
            failures.append(
                {
                    "index": idx,
                    "question": entry.get("question", ""),
                    "failed_rules": [r.rule for r in failed],
                    "reasons": {r.rule: r.reason for r in failed},
                }
            )

    taxonomy = classify_failures(entries)

    report = {
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "taxonomy": taxonomy,
        "failures": failures,
    }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Rule-based QA validation (6 checks)")
    parser.add_argument("--input", required=True, help="Input JSONL (synthetic_qa.jsonl)")
    parser.add_argument("--output", required=True, help="Output JSON report")
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

    report = evaluate_entries(entries)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2)

    # Summary to stdout
    print(f"Total: {report['total']} Passed: {report['passed']} Failed: {report['failed']}")
    if report["taxonomy"]:
        print("Taxonomy:")
        for rule, items in report["taxonomy"].items():
            print(f"  {rule}: {len(items)}")
    else:
        print("Taxonomy: all passed")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
