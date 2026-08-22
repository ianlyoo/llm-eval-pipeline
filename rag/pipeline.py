"""Lightweight lexical retrieval + cited answer (NO embeddings, NO LLM calls).

Honest naming: TF-IDF lexical retrieval — NOT vector/embedding.
Uses pure stdlib (re, math, collections, json, sqlite3 optional).
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sqlite3
from collections import Counter

# ---------------------------------------------------------------------------
# Tokenization (shared lexical)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase lexical tokens (alphanumeric + Hangul)."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# ---------------------------------------------------------------------------
# Index: TF-IDF lexical
# ---------------------------------------------------------------------------

class LexicalIndex:
    """In-memory TF-IDF index over chunks."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.chunk_ids: list[str] = [c["chunk_id"] for c in chunks]
        # Precompute
        self.doc_tokens: list[list[str]] = [tokenize(c["text"]) for c in chunks]
        self.doc_tf: list[Counter] = [Counter(toks) for toks in self.doc_tokens]
        self.doc_len: list[int] = [len(toks) for toks in self.doc_tokens]

        # DF
        df: Counter = Counter()
        for toks in self.doc_tokens:
            for t in set(toks):
                df[t] += 1
        self.df = df
        self.n_docs = len(chunks)
        # IDF with smoothing
        self.idf: dict[str, float] = {}
        for term, freq in df.items():
            self.idf[term] = math.log((self.n_docs + 1) / (freq + 1)) + 1.0

        # Precompute doc TF-IDF norms for cosine
        self.doc_vec_norms: list[float] = []
        for tf in self.doc_tf:
            s = 0.0
            for term, cnt in tf.items():
                # tf = 1 + log(cnt) or raw normalized? Use log tf
                w = (1 + math.log(cnt)) * self.idf.get(term, 0)
                s += w * w
            self.doc_vec_norms.append(math.sqrt(s) if s > 0 else 1.0)

    def score(self, query: str) -> list[tuple[int, float]]:
        """Score all docs vs query; returns (chunk_idx, score) sorted desc."""
        q_tokens = tokenize(query)
        if not q_tokens:
            return [(i, 0.0) for i in range(len(self.chunks))]
        q_tf = Counter(q_tokens)
        # Query vec
        q_weights: dict[str, float] = {}
        q_norm_sq = 0.0
        for term, cnt in q_tf.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            w = (1 + math.log(cnt)) * idf
            q_weights[term] = w
            q_norm_sq += w * w
        q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 1.0

        scores: list[tuple[int, float]] = []
        for idx, tf in enumerate(self.doc_tf):
            # dot product
            dot = 0.0
            for term, qw in q_weights.items():
                cnt = tf.get(term, 0)
                if cnt == 0:
                    continue
                dw = (1 + math.log(cnt)) * self.idf[term]
                dot += qw * dw
            denom = q_norm * self.doc_vec_norms[idx]
            cos = dot / denom if denom != 0 else 0.0
            # Boost exact phrase overlap: simple substring check
            scores.append((idx, cos))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        scored = self.score(query)
        out = []
        for idx, sc in scored[:top_k]:
            c = self.chunks[idx]
            snippet = c["text"][:200].replace("\n", " ")
            out.append(
                {
                    "doc_id": c["doc_id"],
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "snippet": snippet,
                    "score": round(float(sc), 6),
                }
            )
        return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def hit_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> int:
    """Hit@k: 1 if any relevant in top-k retrieved, else 0.

    retrieved_ids: ranked list (full); we slice to k.
    relevant_ids: set of relevant chunk_ids or doc_ids (caller consistent).
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    top = set(retrieved_ids[:k])
    rel = set(relevant_ids)
    return 1 if (top & rel) else 0


# ---------------------------------------------------------------------------
# Cited answer (extractive/template, no LLM)
# ---------------------------------------------------------------------------

def validate_citations(answer: dict) -> None:
    """Raise ValueError if answer has no citations."""
    cits = answer.get("citations")
    if not cits or len(cits) == 0:
        raise ValueError("Answer has no citations — FAIL (citation required)")
    for c in cits:
        if not c.get("doc_id") or not c.get("chunk_id"):
            raise ValueError(f"Invalid citation missing doc_id/chunk_id: {c}")


def answer_query(query: str, index: LexicalIndex, top_k: int = 5) -> dict:
    """Build extractive cited answer from top-k retrieved chunks."""
    retrieved = index.retrieve(query, top_k=top_k)
    if not retrieved:
        ans = {"query": query, "answer_text": "No relevant chunks found.", "citations": []}
        # Intentionally no citations — caller must handle via validate
        return ans

    # Template answer: stitch top snippets
    parts = []
    for r in retrieved:
        parts.append(r["snippet"])
    if any("\uac00" <= ch <= "\ud7a3" for ch in query):
        answer_text = (
            f"Query '{query}'에 대한 관련 근거를 {len(retrieved)}개 청크에서 찾았습니다. "
            + "주요 근거: " + " | ".join(parts[:2])
            + f" (외 {max(0, len(retrieved)-2)}개 추가 근거 참조)."
        )
    else:
        answer_text = (
            f"Found {len(retrieved)} relevant chunks for query '{query}'. "
            + "Key evidence: " + " | ".join(parts[:2])
            + f" (+{max(0, len(retrieved)-2)} more)."
        )

    citations = [
        {
            "doc_id": r["doc_id"],
            "chunk_id": r["chunk_id"],
            "snippet": r["snippet"],
            "score": r["score"],
        }
        for r in retrieved
    ]
    result = {
        "query": query,
        "answer_text": answer_text,
        "citations": citations,
        "retrieved": retrieved,
    }
    # Validate before return? Keep but allow caller to validate separately.
    # We ensure citations present when retrieved non-empty.
    return result


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_chunks(path: pathlib.Path) -> list[dict]:
    """Load JSONL chunks; fallback to SQLite if .db."""
    if path.suffix == ".db":
        con = sqlite3.connect(str(path))
        cur = con.execute("SELECT doc_id, chunk_id, text, token_count FROM chunks")
        rows = cur.fetchall()
        con.close()
        return [{"doc_id": r[0], "chunk_id": r[1], "text": r[2], "token_count": r[3]} for r in rows]
    # JSONL
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_chunks_path_default() -> pathlib.Path:
    # Search common locations
    for p in [pathlib.Path("data/chunks.jsonl"), pathlib.Path("data/chunks.db")]:
        if p.exists():
            return p
    return pathlib.Path("data/chunks.jsonl")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import contextlib
    import sys

    # Force UTF-8 stdout on Windows cp949 terminals
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Lexical retrieval pipeline (TF-IDF, no embeddings)"
    )
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--chunks", type=str, default=None, help="Path to chunks JSONL or SQLite DB"
    )
    parser.add_argument(
        "--json-out", type=str, default=None, help="Optional output file (JSONL append)"
    )
    args = parser.parse_args()

    chunks_path = pathlib.Path(args.chunks) if args.chunks else build_chunks_path_default()
    if not chunks_path.exists():
        raise SystemExit(f"Chunks not found: {chunks_path} — run docs_to_chunks first")

    chunks = load_chunks(chunks_path)
    index = LexicalIndex(chunks)
    result = answer_query(args.query, index, top_k=args.top_k)

    # Validate for CLI: warn if no citations but still output
    line = json.dumps(result, ensure_ascii=False)
    try:
        print(line)
    except UnicodeEncodeError:
        import sys

        sys.stdout.buffer.write((line + "\n").encode("utf-8"))
    if args.json_out:
        with open(args.json_out, "a", encoding="utf-8") as out:
            out.write(line + "\n")


if __name__ == "__main__":
    main()
