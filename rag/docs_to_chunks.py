"""Document chunking — whitespace-heuristic token approximation, 512/overlap 50.

Token counting approximation:
  Uses whitespace splitting (text.split()) as token proxy. This is NOT a
  BPE/tokenizer-accurate count — it under-counts subword splits and
  over-counts for CJK without spaces. Kept stdlib-only to avoid heavy deps
  (tiktoken/sentencepiece). Documented here so downstream metrics note the
  approximation. For English/Korean mixed docs, empirical ratio ~0.75-1.3x
  vs tiktoken cl100k depending on content.

Output contract (stable for synthetic QA):
  JSONL {doc_id, chunk_id, text, token_count}
  chunk_id = f"{doc_id}::chunk-{idx:04d}"
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re


def count_tokens(text: str) -> int:
    """Heuristic token count: whitespace tokens.

    Splits on whitespace; empty string -> 0.
    For CJK-heavy text without spaces, fallback: char-based estimate
    (treat 2 CJK chars ~ 1 token, mixed) — but keep simple: if no spaces
    and contains CJK, count len//2 + word count otherwise.
    Implemented as: if text has whitespace, split count; else heuristic.
    """
    stripped = text.strip()
    if not stripped:
        return 0
    # Detect CJK-dominant no-space case
    if " " not in stripped and "\n" not in stripped and "\t" not in stripped:
        cjk = len(re.findall(r"[\uac00-\ud7a3\u3040-\u30ff\u4e00-\u9fff]", stripped))
        if cjk > len(stripped) * 0.5:
            # ~1.5 chars per token approx
            return max(1, (len(stripped) + 1) // 2)
    return len(stripped.split())


def chunk_document(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> list[dict]:
    """Split text into overlapping chunks.

    Args:
        text: Full document text.
        max_tokens: Max heuristic tokens per chunk.
        overlap_tokens: Overlap between consecutive chunks (in tokens).

    Returns:
        List of {chunk_index, text, token_count}. Caller adds doc_id/chunk_id.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be < max_tokens")

    tokens = text.split()
    # For CJK-no-space docs, token splitting above is coarse; use char sliding as fallback
    # But keep unified path: if whitespace token count mismatches heuristic, chunk by words
    # Simple: chunk by whitespace tokens always; CJK docs will just be 1 chunk if short
    if not tokens:
        return []

    # If heuristic says overlap >= max, already validated
    step = max_tokens - overlap_tokens
    chunks: list[dict] = []
    idx = 0
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = " ".join(chunk_tokens)
        # token_count via heuristic (should equal len(chunk_tokens) except CJK case)
        tc = count_tokens(chunk_text)
        chunks.append(
            {
                "chunk_index": idx,
                "text": chunk_text,
                "token_count": tc,
            }
        )
        idx += 1
        if end >= len(tokens):
            break
        start += step

    return chunks


def chunk_file(
    doc_id: str,
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> list[dict]:
    """Wrap chunk_document with doc_id/chunk_id envelope."""
    raw = chunk_document(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    out = []
    for c in raw:
        out.append(
            {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}::chunk-{c['chunk_index']:04d}",
                "text": c["text"],
                "token_count": c["token_count"],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Document -> chunks JSONL (512/50)")
    parser.add_argument(
        "--input", required=True, help="Input dir with .md/.txt files or single file"
    )
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--overlap-tokens", type=int, default=50)
    args = parser.parse_args()

    in_path = pathlib.Path(args.input)
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files: list[pathlib.Path] = []
    if in_path.is_dir():
        files = sorted(
            [p for p in in_path.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt", ".markdown"}]
        )
        if not files:
            # fallback: any file
            files = sorted([p for p in in_path.rglob("*") if p.is_file()])
    elif in_path.is_file():
        files = [in_path]
    else:
        raise SystemExit(f"Input not found: {in_path}")

    total_chunks = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for fp in files:
            text = fp.read_text(encoding="utf-8", errors="replace")
            doc_id = fp.stem
            # Use relative name for uniqueness if duplicate stems
            # Keep stem as doc_id per simplest contract
            chunks = chunk_file(
                doc_id, text, max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens
            )
            for ch in chunks:
                out_f.write(json.dumps(ch, ensure_ascii=False) + "\n")
            total_chunks += len(chunks)
            print(f"{doc_id}: {len(chunks)} chunks from {fp}")

    print(f"Done: {len(files)} docs -> {total_chunks} chunks -> {out_path}")


if __name__ == "__main__":
    main()
