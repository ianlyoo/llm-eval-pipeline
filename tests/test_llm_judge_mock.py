"""Tests for llm_judge — mock judge schema, variance, disagreement."""

import json
import pathlib
import random
import subprocess
import sys

from eval.llm_judge import (
    compute_variance,
    detect_disagreement,
    deterministic_judge,
    judge_entry,
)


def _base_entry(**overrides):
    base = {
        "question": "원격근무 핵심 시간은 언제인가?",
        "reference_answer": "핵심 시간은 10:00부터 16:00 KST이며 Slack에서 연락 가능하다.",
        "candidate_answer": "핵심 시간은 10:00부터 16:00 KST이며 Slack에서 연락 가능하다.",
        "source_chunks": [
            {
                "doc_id": "01_company_policy",
                "chunk_id": "01_company_policy::chunk-0000",
                "text": "Core hours: 10:00–16:00 KST. Employees must be reachable via Slack/Email. Flexible start 07:00–10:00.",
            }
        ],
        "category": "policy",
        "difficulty": "easy",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Schema — scores with reasons required
# ---------------------------------------------------------------------------

def test_judge_schema_has_four_metrics_with_score_and_reason():
    e = _base_entry()
    scores = deterministic_judge(e)
    for metric in ["correctness", "groundedness", "relevance", "completeness"]:
        assert metric in scores, f"missing {metric}"
        assert "score" in scores[metric]
        assert "reason" in scores[metric]
        assert 1 <= scores[metric]["score"] <= 5
        assert scores[metric]["reason"] and scores[metric]["reason"].strip() != ""
        assert len(scores[metric]["reason"]) >= 5  # non-trivial reason


def test_judge_scores_are_1_to_5_range():
    e = _base_entry(
        reference_answer="핵심 시간은 10:00부터 16:00이다.",
        candidate_answer="전혀 다른 내용 999조 원 근거 없음",
    )
    scores = deterministic_judge(e)
    for k, v in scores.items():
        assert 1 <= v["score"] <= 5, f"{k} out of range"


def test_judge_reason_required_not_empty():
    e = _base_entry()
    scores = deterministic_judge(e)
    for k, v in scores.items():
        r = v["reason"]
        assert r.strip() != "", f"{k} reason empty — FAIL"
        # should mention metric context
        assert len(r) > 10


def test_judge_entry_schema_latency_tokens_cost():
    e = _base_entry()
    res = judge_entry(e, profile="default", run_id="test-0")
    assert "question" in res
    assert "candidate_answer" in res
    assert "scores" in res
    assert "latency_ms" in res and res["latency_ms"] >= 1
    assert "tokens_est" in res and res["tokens_est"] > 0
    assert "cost_est" in res
    assert "run_id" in res
    # all scores have reasons
    for _k, v in res["scores"].items():
        assert "reason" in v and v["reason"].strip() != ""


def test_judge_fallback_candidate_uses_reference_when_missing():
    e = _base_entry()
    del e["candidate_answer"]
    res = judge_entry(e)
    # candidate_answer should fallback to reference
    assert res["candidate_answer"] == e["reference_answer"]
    # perfect match should be high correctness
    assert res["scores"]["correctness"]["score"] >= 3


def test_groundedness_numeric_missing_low():
    e = _base_entry(candidate_answer="매출 999조 원 달성 전혀 근거 없음")
    scores = deterministic_judge(e)
    # 999 not in source → groundedness low
    assert scores["groundedness"]["score"] <= 2
    assert "999" in scores["groundedness"]["reason"] or "missing" in scores["groundedness"]["reason"]


def test_groundedness_all_numeric_found_high():
    e = _base_entry(
        reference_answer="핵심 시간 10:00부터 16:00",
        candidate_answer="핵심 시간 10:00부터 16:00이다",
    )
    scores = deterministic_judge(e)
    assert scores["groundedness"]["score"] >= 4


def test_strict_vs_lenient_profiles_differ():
    e = _base_entry()
    s_strict = deterministic_judge(e, profile="deterministic-strict")
    s_lenient = deterministic_judge(e, profile="deterministic-lenient")
    # At least one metric should differ
    diffs = [s_strict[k]["score"] != s_lenient[k]["score"] for k in s_strict]
    assert any(diffs), "strict vs lenient should differ by at least 1"


# ---------------------------------------------------------------------------
# Variance computation
# ---------------------------------------------------------------------------

def test_compute_variance_basic():
    assert compute_variance([5, 5, 5]) == 0.0
    assert compute_variance([1, 5]) == 4.0  # mean 3, var (4+4)/2=4
    v = compute_variance([3, 4, 5])  # mean 4, var (1+0+1)/3=0.666...
    assert abs(v - 0.666666) < 0.001


def test_compute_variance_single_zero():
    assert compute_variance([3]) == 0.0
    assert compute_variance([]) == 0.0


def test_variance_with_noise_is_small():
    e = _base_entry()
    # simulate 3 runs with noise
    runs = []
    for i in range(3):
        rng = random.Random(42 * 100 + i * 10 + 7)
        s = deterministic_judge(e, noise_rng=rng)
        runs.append(s)
    for metric in ["correctness", "groundedness", "relevance", "completeness"]:
        vals = [float(r[metric]["score"]) for r in runs]
        var = compute_variance(vals)
        assert var < 1.0, f"{metric} variance {var} should be <1.0"


def test_variance_deterministic_no_noise_zero():
    e = _base_entry()
    vals = []
    for _ in range(3):
        s = deterministic_judge(e)
        vals.append(float(s["correctness"]["score"]))
    assert compute_variance(vals) == 0.0


# ---------------------------------------------------------------------------
# Disagreement detection
# ---------------------------------------------------------------------------

def test_detect_disagreement_true_when_diff_ge1():
    a = {
        "correctness": {"score": 5, "reason": "r"},
        "groundedness": {"score": 5, "reason": "r"},
        "relevance": {"score": 5, "reason": "r"},
        "completeness": {"score": 5, "reason": "r"},
    }
    b = {
        "correctness": {"score": 3, "reason": "r"},
        "groundedness": {"score": 5, "reason": "r"},
        "relevance": {"score": 5, "reason": "r"},
        "completeness": {"score": 5, "reason": "r"},
    }
    disc = detect_disagreement(a, b, threshold=1)
    assert disc["is_disagreement"] is True
    assert disc["diffs"]["correctness"] == 2


def test_detect_disagreement_false_when_all_same():
    a = {
        "correctness": {"score": 4, "reason": "r"},
        "groundedness": {"score": 4, "reason": "r"},
        "relevance": {"score": 4, "reason": "r"},
        "completeness": {"score": 4, "reason": "r"},
    }
    disc = detect_disagreement(a, a, threshold=1)
    assert disc["is_disagreement"] is False


def test_disagreement_threshold():
    a = {"correctness": {"score": 4, "reason": "r"}, "groundedness": {"score": 4, "reason": "r"}, "relevance": {"score": 4, "reason": "r"}, "completeness": {"score": 4, "reason": "r"}}
    b = {"correctness": {"score": 4, "reason": "r"}, "groundedness": {"score": 4, "reason": "r"}, "relevance": {"score": 4, "reason": "r"}, "completeness": {"score": 5, "reason": "r"}}
    assert detect_disagreement(a, b, threshold=1)["is_disagreement"] is True
    assert detect_disagreement(a, b, threshold=2)["is_disagreement"] is False


def test_strict_lenient_produces_disagreement():
    e = _base_entry()
    sa = deterministic_judge(e, profile="deterministic-strict")
    sb = deterministic_judge(e, profile="deterministic-lenient")
    disc = detect_disagreement(sa, sb, threshold=1)
    assert disc["is_disagreement"] is True


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

def test_cli_judge_scores_output(tmp_path):
    inp = pathlib.Path("data/synthetic_qa.jsonl")
    if not inp.exists():
        import pytest
        pytest.skip("synthetic data missing")
    out = tmp_path / "judge_scores.jsonl"
    result = subprocess.run(
        [sys.executable, "-m", "eval.llm_judge", "--input", str(inp), "--sample", "3", "--output", str(out)],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out.exists()
    raw_lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(raw_lines) == 3
    for line in raw_lines:
        obj = json.loads(line)
        assert "scores" in obj
        for k in ["correctness", "groundedness", "relevance", "completeness"]:
            assert "score" in obj["scores"][k]
            assert "reason" in obj["scores"][k] and obj["scores"][k]["reason"].strip() != ""
        assert "latency_ms" in obj
        assert "tokens_est" in obj


def test_cli_reliability_produces_variance_and_disagreement(tmp_path):
    inp = pathlib.Path("data/synthetic_qa.jsonl")
    if not inp.exists():
        import pytest
        pytest.skip("synthetic data missing")
    out = tmp_path / "judge_scores.jsonl"
    dis = tmp_path / "disagreement.jsonl"
    log = tmp_path / "reliability.log"
    result = subprocess.run(
        [
            sys.executable, "-m", "eval.llm_judge",
            "--input", str(inp), "--sample", "5",
            "--output", str(out),
            "--reliability",
            "--disagreement-output", str(dis),
            "--reliability-log", str(log),
        ],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out.exists() and dis.exists() and log.exists()
    # 5 samples x3 =15 lines
    out_lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(out_lines) == 15
    dis_lines = [ln for ln in dis.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(dis_lines) >= 1, "should have at least 1 disagreement"
    assert all("forced" not in json.loads(line) for line in dis_lines)
    log_text = log.read_text(encoding="utf-8")
    assert "Variance check PASS" in log_text
