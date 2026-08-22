"""Tests for rule_filter — 6 rules + corruption cases."""

import copy
import json
import pathlib
import subprocess
import sys

import pytest

from eval.rule_filter import (
    check_citation_exists,
    check_empty_answer,
    check_expected_keyword,
    check_formatting_violation,
    check_source_coverage,
    check_unsupported_answer,
    classify_failures,
    evaluate_entries,
    run_all_rules,
)


def _base_entry(**overrides):
    base = {
        "question": "원격근무 핵심 시간은 언제인가?",
        "reference_answer": "핵심 시간은 10:00부터 16:00 KST이며 Slack에서 연락 가능하다.",
        "source_chunks": [
            {
                "doc_id": "01_company_policy",
                "chunk_id": "01_company_policy::chunk-0000",
                "text": "Core hours: 10:00–16:00 KST. Employees must be reachable via Slack/Email. "
                "Flexible start between 07:00–10:00, end accordingly.",
            }
        ],
        "category": "policy",
        "difficulty": "easy",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Unit tests per rule
# ---------------------------------------------------------------------------

def test_citation_exists_pass():
    e = _base_entry()
    r = check_citation_exists(e)
    assert r.passed is True


def test_citation_exists_fail_empty():
    e = _base_entry(source_chunks=[])
    r = check_citation_exists(e)
    assert r.passed is False
    assert "citation" in r.reason.lower()


def test_citation_exists_fail_missing_doc_id():
    e = _base_entry(source_chunks=[{"chunk_id": "x", "text": "hello", "doc_id": ""}])
    r = check_citation_exists(e)
    assert r.passed is False


def test_citation_exists_fail_missing_text():
    e = _base_entry(source_chunks=[{"doc_id": "d", "chunk_id": "c", "text": ""}])
    r = check_citation_exists(e)
    assert r.passed is False


def test_source_coverage_pass():
    e = _base_entry()
    assert check_source_coverage(e).passed is True


def test_source_coverage_fail_empty():
    e = _base_entry(source_chunks=[])
    assert check_source_coverage(e).passed is False


def test_empty_answer_pass():
    e = _base_entry()
    assert check_empty_answer(e).passed is True


def test_empty_answer_fail_blank():
    e = _base_entry(reference_answer="   ")
    assert check_empty_answer(e).passed is False


def test_empty_answer_fail_missing():
    e = _base_entry()
    del e["reference_answer"]
    assert check_empty_answer(e).passed is False


def test_formatting_violation_pass():
    e = _base_entry()
    assert check_formatting_violation(e).passed is True


def test_formatting_violation_fail_short():
    e = _base_entry(reference_answer="짧음")
    r = check_formatting_violation(e)
    assert r.passed is False
    assert "short" in r.reason.lower()


def test_formatting_violation_fail_single_token():
    e = _base_entry(reference_answer="hello")
    # len 5 <10 -> fails short
    assert check_formatting_violation(e).passed is False


def test_unsupported_answer_pass_no_numeric():
    e = _base_entry(reference_answer="원격근무는 유연하게 가능합니다.")
    # source has no numbers, answer has no numbers -> pass
    assert check_unsupported_answer(e).passed is True


def test_unsupported_answer_pass_numeric_in_source():
    e = _base_entry(reference_answer="핵심 시간은 10:00부터 16:00이다.")
    # source contains 10:00 and 16:00
    assert check_unsupported_answer(e).passed is True


def test_unsupported_answer_fail_fake_numeric():
    e = _base_entry(reference_answer="매출 999조 원 달성")
    r = check_unsupported_answer(e)
    assert r.passed is False
    assert "999" in r.reason


def test_unsupported_answer_fail_no_source():
    e = _base_entry(reference_answer="매출 999조", source_chunks=[])
    assert check_unsupported_answer(e).passed is False


def test_expected_keyword_pass():
    e = _base_entry(
        question="원격근무 핵심 시간은 언제인가?",
        reference_answer="핵심 시간은 10:00부터 16:00 KST이다.",
    )
    r = check_expected_keyword(e)
    assert r.passed is True


def test_expected_keyword_fail_unrelated():
    e = _base_entry(
        question="P1 에스컬레이션 절차는?",
        reference_answer="사과는 빨갛고 바나나는 노랗습니다. 전혀 관계없는 내용.",
    )
    r = check_expected_keyword(e)
    assert r.passed is False


def test_expected_keyword_empty_question_or_answer():
    e = _base_entry(question="", reference_answer="answer")
    assert check_expected_keyword(e).passed is False
    e2 = _base_entry(question="q?", reference_answer="")
    assert check_expected_keyword(e2).passed is False


def test_run_all_rules_returns_6():
    e = _base_entry()
    results = run_all_rules(e)
    assert len(results) == 6
    assert all(hasattr(r, "passed") for r in results)


def test_classify_failures_taxonomy():
    good = _base_entry()
    bad = _base_entry(source_chunks=[], reference_answer="")
    entries = [good, bad]
    tax = classify_failures(entries)
    # bad should contribute to citation and coverage and empty
    assert "citation_exists" in tax or "source_coverage" in tax
    assert "empty_answer" in tax


def test_evaluate_entries_counts():
    good = _base_entry()
    bad = _base_entry(reference_answer="")
    report = evaluate_entries([good, bad])
    assert report["total"] == 2
    assert report["passed"] == 1
    assert report["failed"] == 1
    assert len(report["failures"]) == 1


# ---------------------------------------------------------------------------
# Corruption 5 cases — must all be caught
# ---------------------------------------------------------------------------

def _load_synthetic() -> list[dict]:
    p = pathlib.Path("data/synthetic_qa.jsonl")
    if not p.exists():
        pytest.skip("synthetic_qa not generated")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_corruption_5_source_empty_caught():
    entries = _load_synthetic()
    e = copy.deepcopy(entries[5])
    e["source_chunks"] = []
    fails = [r.rule for r in run_all_rules(e) if not r.passed]
    assert "source_coverage" in fails
    assert "citation_exists" in fails


def test_corruption_10_empty_answer_caught():
    entries = _load_synthetic()
    e = copy.deepcopy(entries[10])
    e["reference_answer"] = ""
    fails = [r.rule for r in run_all_rules(e) if not r.passed]
    assert "empty_answer" in fails
    assert "formatting_violation" in fails


def test_corruption_15_fake_numeric_caught():
    entries = _load_synthetic()
    e = copy.deepcopy(entries[15])
    e["reference_answer"] = "매출 999조 원, 영업이익 888조 원 달성"
    fails = [r.rule for r in run_all_rules(e) if not r.passed]
    assert "unsupported_answer" in fails


def test_corruption_20_keyword_missing_caught():
    entries = _load_synthetic()
    e = copy.deepcopy(entries[20])
    e["reference_answer"] = "이 답변은 질문과 전혀 관계없는 내용입니다. 사과는 빨갛고 바나나는 노랗습니다."
    fails = [r.rule for r in run_all_rules(e) if not r.passed]
    assert "expected_keyword" in fails


def test_corruption_25_citation_missing_caught():
    entries = _load_synthetic()
    e = copy.deepcopy(entries[25])
    if e["source_chunks"]:
        e["source_chunks"][0].pop("doc_id", None)
    fails = [r.rule for r in run_all_rules(e) if not r.passed]
    assert "citation_exists" in fails


def test_all_five_corruptions_together_all_caught():
    entries = _load_synthetic()
    corrupted = copy.deepcopy(entries)
    corrupted[5]["source_chunks"] = []
    corrupted[10]["reference_answer"] = ""
    corrupted[15]["reference_answer"] = "매출 999조 원 근거 없는 수치"
    corrupted[20]["reference_answer"] = "전혀 관계없는 답변입니다. 바나나 사과"
    corrupted[25]["source_chunks"][0].pop("doc_id", None)

    for idx in [5, 10, 15, 20, 25]:
        fails = [r.rule for r in run_all_rules(corrupted[idx]) if not r.passed]
        assert len(fails) >= 1, f"index {idx} should have at least one fail, got {fails}"


def test_cli_generates_report(tmp_path):
    p = pathlib.Path("data/synthetic_qa.jsonl")
    if not p.exists():
        pytest.skip("synthetic_qa not generated")
    out = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, "-m", "eval.rule_filter", "--input", str(p), "--output", str(out)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "total" in data and "passed" in data and "failed" in data
    assert "taxonomy" in data and "failures" in data
    assert data["total"] == 50
