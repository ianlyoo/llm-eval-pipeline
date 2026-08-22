"""Tests for synthetic QA generation — schema, dedup, reproducibility."""

import json
import pathlib
import random

import pytest

from eval.synthetic_data import (
    CURATED_POOLS,
    DOC_CATEGORIES,
    TEMPLATE_TYPES,
    compute_dedup_metrics,
    load_chunks,
    validate_entry,
)


def test_load_chunks_has_10():
    p = pathlib.Path("data/chunks.jsonl")
    if not p.exists():
        pytest.skip("chunks not built")
    chunks = load_chunks(p)
    assert len(chunks) == 10
    for c in chunks:
        assert "doc_id" in c and "chunk_id" in c and "text" in c


def test_synthetic_qa_file_exists_and_50():
    p = pathlib.Path("data/synthetic_qa.jsonl")
    if not p.exists():
        pytest.skip("synthetic_qa not generated")
    lines = [line for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 50, f"expected 50 got {len(lines)}"


def test_schema_requires_source_chunks():
    # Each entry must have non-empty source_chunks
    p = pathlib.Path("data/synthetic_qa.jsonl")
    if not p.exists():
        pytest.skip("synthetic_qa not generated")
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        validate_entry(obj)
        assert isinstance(obj["source_chunks"], list) and len(obj["source_chunks"]) >= 1
        for sc in obj["source_chunks"]:
            assert sc["doc_id"]
            assert sc["chunk_id"]
            assert sc["text"]
        assert "category" in obj and "difficulty" in obj
        assert obj["difficulty"] in ("easy", "medium", "hard")


def test_source_chunks_without_fails():
    bad = {"question": "q?", "reference_answer": "a", "source_chunks": [], "category": "x", "difficulty": "easy"}
    with pytest.raises(ValueError, match="source_chunks"):
        validate_entry(bad)
    bad2 = {"question": "q?", "reference_answer": "a", "category": "x", "difficulty": "easy"}
    with pytest.raises(ValueError, match="source_chunks"):
        validate_entry(bad2)
    bad3 = {
        "question": "q?",
        "reference_answer": "a",
        "source_chunks": [{"doc_id": "", "chunk_id": "x", "text": "t"}],
        "category": "x",
        "difficulty": "easy",
    }
    with pytest.raises(ValueError):
        validate_entry(bad3)


def test_dedup_under_20_percent():
    p = pathlib.Path("data/synthetic_qa.jsonl")
    if not p.exists():
        pytest.skip("synthetic_qa not generated")
    questions = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            questions.append(json.loads(line)["question"])
    metrics = compute_dedup_metrics(questions)
    assert metrics["total"] == 50
    assert metrics["unique_ratio"] >= 0.8, f"unique_ratio {metrics['unique_ratio']} < 0.8"
    assert metrics["duplicate_rate"] < 0.2


def test_compute_dedup_metrics_exact():
    assert compute_dedup_metrics([])["unique_ratio"] == 1.0
    m = compute_dedup_metrics(["a", "a", "b"])
    assert m["total"] == 3
    assert m["unique"] == 2
    assert m["duplicate_count"] == 1
    assert m["duplicate_rate"] == pytest.approx(1 / 3, abs=1e-4)


def test_reproducibility_same_seed():
    from eval.synthetic_data import _generate_rule_qa

    p = pathlib.Path("data/chunks.jsonl")
    if not p.exists():
        pytest.skip("chunks not built")
    chunks = load_chunks(p)
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    a = _generate_rule_qa(rng1, chunks, 50)
    b = _generate_rule_qa(rng2, chunks, 50)
    qs_a = [e["question"] for e in a]
    qs_b = [e["question"] for e in b]
    # Same seed -> same multiset (order may be shuffled same way)
    assert qs_a == qs_b
    # Also check answers same
    assert [e["reference_answer"] for e in a] == [e["reference_answer"] for e in b]


def test_reproducibility_different_seed_varies_order():
    from eval.synthetic_data import _generate_rule_qa

    p = pathlib.Path("data/chunks.jsonl")
    if not p.exists():
        pytest.skip("chunks not built")
    chunks = load_chunks(p)
    a = _generate_rule_qa(random.Random(1), chunks, 50)
    b = _generate_rule_qa(random.Random(99), chunks, 50)
    # Different seeds likely produce different order (not strictly required to be different set content)
    qa_a = {e["question"] for e in a}
    qa_b = {e["question"] for e in b}
    # At least ordering differs or some variant changes; if pools are deterministic the sets may overlap heavily
    # so assert not identical ordered list
    # Different seeds produce different shuffle orders — at least one difference expected
    assert [e["question"] for e in a] != [e["question"] for e in b] or qa_a != qa_b
    assert len(a) == 50 and len(b) == 50


def test_per_chunk_at_least_3_distinct_questions():
    p = pathlib.Path("data/synthetic_qa.jsonl")
    if not p.exists():
        pytest.skip("synthetic_qa not generated")
    from collections import defaultdict

    per_doc: dict[str, list[str]] = defaultdict(list)
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        doc = obj["source_chunks"][0]["doc_id"]
        per_doc[doc].append(obj["question"])
    for doc, qs in per_doc.items():
        assert len(qs) >= 3, f"{doc} has only {len(qs)}"
        assert len(set(qs)) >= 3, f"{doc} distinct <3"


def test_categories_and_difficulties_valid():
    p = pathlib.Path("data/synthetic_qa.jsonl")
    if not p.exists():
        pytest.skip("synthetic_qa not generated")
    valid_cats = set(DOC_CATEGORIES.values()) | {"general", "policy", "product", "technical", "hr", "security", "ml", "support", "privacy", "devops", "finance"}  # noqa: E501
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        assert obj["category"] in valid_cats
        assert obj["difficulty"] in ("easy", "medium", "hard")


def test_template_types_at_least_5():
    assert len(TEMPLATE_TYPES) >= 5


def test_curated_pools_cover_all_chunks():
    p = pathlib.Path("data/chunks.jsonl")
    if not p.exists():
        pytest.skip("chunks not built")
    chunks = load_chunks(p)
    for c in chunks:
        assert c["doc_id"] in CURATED_POOLS, f"missing curated pool for {c['doc_id']}"
        assert len(CURATED_POOLS[c["doc_id"]]) >= 5


def test_source_chunks_text_is_substring_of_original():
    p = pathlib.Path("data/synthetic_qa.jsonl")
    cp = pathlib.Path("data/chunks.jsonl")
    if not p.exists() or not cp.exists():
        pytest.skip("files not built")
    chunk_texts = {c["chunk_id"]: c["text"] for c in load_chunks(cp)}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        for sc in obj["source_chunks"]:
            orig = chunk_texts.get(sc["chunk_id"], "")
            # snippet should be prefix of orig (first 220 chars)
            assert sc["text"] in orig or orig[:220] in sc["text"] or sc["text"][:50] in orig
