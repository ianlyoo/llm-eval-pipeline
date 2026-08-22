"""Tests for lightweight rag chunk/retrieval stub."""

import json
import pathlib

import pytest

from rag.docs_to_chunks import chunk_document, count_tokens
from rag.pipeline import LexicalIndex, answer_query, hit_at_k, validate_citations


def test_count_tokens_basic():
    assert count_tokens("") == 0
    assert count_tokens("hello world") == 2
    assert count_tokens("  hello   world  ") == 2


def test_chunk_document_empty():
    assert chunk_document("") == []


def test_chunk_document_single_chunk():
    text = "hello world " * 10
    chunks = chunk_document(text, max_tokens=512, overlap_tokens=50)
    assert len(chunks) == 1
    assert chunks[0]["token_count"] <= 512


def test_chunk_document_overlap_boundaries():
    # 600 tokens, max 512 overlap 50 -> step 462 -> 2 chunks
    tokens = [f"tok{i}" for i in range(600)]
    text = " ".join(tokens)
    chunks = chunk_document(text, max_tokens=512, overlap_tokens=50)
    assert len(chunks) == 2
    # First chunk 512 tokens
    assert chunks[0]["token_count"] == 512
    # Second chunk starts at 462, ends at 600 -> 138 tokens
    assert chunks[1]["token_count"] == 138
    # Overlap: tokens 462-511 should appear in both
    first_tokens = chunks[0]["text"].split()
    second_tokens = chunks[1]["text"].split()
    assert first_tokens[462:512] == second_tokens[:50]


def test_chunk_document_overlap_zero():
    tokens = [f"w{i}" for i in range(100)]
    text = " ".join(tokens)
    chunks = chunk_document(text, max_tokens=50, overlap_tokens=0)
    assert len(chunks) == 2
    assert chunks[0]["token_count"] == 50
    assert chunks[1]["token_count"] == 50


def test_chunk_document_invalid_params():
    with pytest.raises(ValueError):
        chunk_document("hello", max_tokens=0)
    with pytest.raises(ValueError):
        chunk_document("hello", max_tokens=10, overlap_tokens=10)
    with pytest.raises(ValueError):
        chunk_document("hello", max_tokens=10, overlap_tokens=-1)


def test_chunk_document_exact_512():
    tokens = [f"x{i}" for i in range(512)]
    text = " ".join(tokens)
    chunks = chunk_document(text, max_tokens=512, overlap_tokens=50)
    assert len(chunks) == 1
    assert chunks[0]["token_count"] == 512


def test_retrieval_returns_top_k_with_scores():
    chunks = [
        {"doc_id": "a", "chunk_id": "a::chunk-0000", "text": "remote work core hours VPN", "token_count": 5},
        {"doc_id": "b", "chunk_id": "b::chunk-0000", "text": "pricing sync offline export", "token_count": 4},
        {"doc_id": "c", "chunk_id": "c::chunk-0000", "text": "Atlas API rate limit Bearer token", "token_count": 6},  # noqa: E501
    ]
    idx = LexicalIndex(chunks)
    res = idx.retrieve("remote work VPN", top_k=2)
    assert len(res) == 2
    # Top should be doc a
    assert res[0]["doc_id"] == "a"
    assert "score" in res[0]
    assert res[0]["score"] >= 0


def test_answer_has_citations():
    chunks = [
        {"doc_id": "a", "chunk_id": "a::chunk-0000", "text": "remote work core hours VPN", "token_count": 5},
        {"doc_id": "b", "chunk_id": "b::chunk-0000", "text": "pricing sync offline export", "token_count": 4},
    ]
    idx = LexicalIndex(chunks)
    ans = answer_query("remote work", idx, top_k=2)
    assert "answer_text" in ans
    assert "citations" in ans
    assert len(ans["citations"]) >= 1
    for c in ans["citations"]:
        assert "doc_id" in c and "chunk_id" in c and "snippet" in c and "score" in c
    # validate should pass
    validate_citations(ans)


def test_answer_without_citations_fails_validation():
    bad = {"query": "test", "answer_text": "no source", "citations": []}
    with pytest.raises(ValueError, match="no citations"):
        validate_citations(bad)
    bad2 = {"query": "test", "answer_text": "x", "citations": [{"doc_id": "", "chunk_id": "a"}]}
    with pytest.raises(ValueError):
        validate_citations(bad2)
    # Missing citations key
    with pytest.raises(ValueError):
        validate_citations({"query": "q", "answer_text": "a"})


def test_hit_at_k():
    assert hit_at_k(["a", "b", "c"], ["b"], k=2) == 1
    assert hit_at_k(["a", "b", "c"], ["d"], k=2) == 0
    assert hit_at_k(["a", "b"], ["b"], k=1) == 0  # b not in top1
    assert hit_at_k(["a", "b"], ["b"], k=5) == 1
    with pytest.raises(ValueError):
        hit_at_k(["a"], ["a"], k=0)


def test_cli_jsonl_output_contract():
    # Verify data/chunks.jsonl exists and has correct schema
    p = pathlib.Path("data/chunks.jsonl")
    if not p.exists():
        pytest.skip("chunks not built yet")
    with p.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            assert "doc_id" in obj
            assert "chunk_id" in obj
            assert "text" in obj
            assert "token_count" in obj
            assert obj["token_count"] <= 512
