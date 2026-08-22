#!/usr/bin/env python
"""Generate out/rag_baseline.log — chunk stats, 20 queries, Hit@5, citation validation."""
import contextlib
import json
import pathlib
import subprocess
import sys

from rag.docs_to_chunks import chunk_document
from rag.pipeline import LexicalIndex, answer_query, hit_at_k, load_chunks, validate_citations

# Ensure UTF-8 stdout
with contextlib.suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "chunks.jsonl"
QA = ROOT / "data" / "qa_smoke.json"
OUT = ROOT / "out" / "rag_baseline.log"
PORTFOLIO_OUT = pathlib.Path(r"C:\Users\torch\Documents\code\portfolio\out\rag_baseline.log")

OUT.parent.mkdir(parents=True, exist_ok=True)
PORTFOLIO_OUT.parent.mkdir(parents=True, exist_ok=True)

lines = []
def log(s=""):
    print(s)
    lines.append(s)

log("="*72)
log("RAG Baseline Log — lightweight lexical retrieval (TF-IDF, no embeddings)")
log("="*72)
log("")

# 1. Chunk stats
chunks = load_chunks(CHUNKS)
log(f"Chunks file: {CHUNKS}")
log(f"Total chunks: {len(chunks)}")
doc_ids = sorted(set(c["doc_id"] for c in chunks))
log(f"Docs: {len(doc_ids)} -> {doc_ids}")
for c in chunks:
    log(f"  {c['chunk_id']} doc={c['doc_id']} tokens={c['token_count']} chars={len(c['text'])}")
log("")
# Token distribution
tcs = [c["token_count"] for c in chunks]
log(f"Token counts: min={min(tcs)} max={max(tcs)} avg={sum(tcs)/len(tcs):.1f}")
violations = [c for c in chunks if c["token_count"] > 512]
log(f"Chunks exceeding 512 tokens: {len(violations)} (must be 0)")
assert len(violations) == 0, "chunk overflow"
log("")

# 2. Overlap sanity: check chunk_document overlap 50
sample_text = " ".join([f"tok{i}" for i in range(600)])
sample_chunks = chunk_document(sample_text, max_tokens=512, overlap_tokens=50)
log(f"Overlap sanity (600 tokens, 512/50): {len(sample_chunks)} chunks, sizes {[c['token_count'] for c in sample_chunks]}")
assert len(sample_chunks) == 2
log("")

# 3. Index build
index = LexicalIndex(chunks)
log(f"LexicalIndex built: N={index.n_docs}, vocab={len(index.idf)}")
log("")

# 4. Run 20 queries
with QA.open(encoding="utf-8") as f:
    qa = json.load(f)
log(f"QA file: {QA} with {len(qa)} queries")
log("")

hits = []
all_cited = True
for item in qa:
    qid = item["qid"]
    query = item["query"]
    relevant_doc_ids = item["relevant_doc_ids"]
    # Also build relevant chunk_ids: any chunk whose doc_id in relevant_doc_ids
    relevant_chunk_ids = [c["chunk_id"] for c in chunks if c["doc_id"] in relevant_doc_ids]
    result = answer_query(query, index, top_k=5)
    retrieved_ids_doc = [r["doc_id"] for r in result["retrieved"]]
    retrieved_ids_chunk = [r["chunk_id"] for r in result["retrieved"]]
    # Hit@5 on doc_id and chunk_id (both)
    hit_doc = hit_at_k(retrieved_ids_doc, relevant_doc_ids, k=5)
    hit_chunk = hit_at_k(retrieved_ids_chunk, relevant_chunk_ids, k=5)
    # Use doc-level Hit as primary (more lenient, chunks are 1 per doc here)
    hit = hit_doc
    hits.append(hit)
    # citation validation
    try:
        validate_citations(result)
        cit_ok = "PASS"
    except ValueError as e:
        cit_ok = f"FAIL: {e}"
        all_cited = False

    log(f"[{qid}] query: {query}")
    log(f"  relevant doc(s): {relevant_doc_ids}")
    log(f"  retrieved top-5 doc_ids: {retrieved_ids_doc} scores: {[r['score'] for r in result['retrieved']]}")
    log(f"  Hit@5 (doc): {hit}  Hit@5 (chunk): {hit_chunk}  citations: {len(result['citations'])} [{cit_ok}]")
    # Show answer snippet (first 180 chars)
    ans_snip = result["answer_text"][:220].replace("\n"," ")
    log(f"  answer: {ans_snip}")
    # Citations detail
    for cit in result["citations"]:
        log(f"    cite -> {cit['doc_id']} {cit['chunk_id']} score={cit['score']} snippet=\"{cit['snippet'][:80]}...\"")
    log("")

hit_rate = sum(hits)/len(hits) if hits else 0
log("-"*72)
log(f"Hit@5 (doc-level): {sum(hits)}/{len(hits)} = {hit_rate:.3f} ({hit_rate*100:.1f}%)")
log(f"Citation validation: {'ALL PASS (20/20 have >=1 citation)' if all_cited else 'FAIL — some answers missing citations'}")
log("")

# 5. Negative test: answer without citations must be detectable
log("Negative test: answer without citations -> validate_citations must raise")
try:
    validate_citations({"query":"test","answer_text":"no cite","citations":[]})
    log("  FAIL: did not raise")
    neg_pass = False
except ValueError as e:
    log(f"  PASS: raised ValueError: {e}")
    neg_pass = True
log("")

# 6. CLI smoke
log("CLI smoke: python -m rag.pipeline --query \"테스트\" --top-k 5 -> JSONL")

proc = subprocess.run(
    [sys.executable, "-m", "rag.pipeline", "--query", "테스트", "--top-k", "5"],
    capture_output=True,
    encoding="utf-8",
    errors="replace",
)
out = proc.stdout.strip()
if out:
    j = json.loads(out)
    log(f"  CLI output keys: {list(j.keys())} citations={len(j.get('citations',[]))}")
    log(f"  CLI JSONL (truncated 400): {out[:400]}")
else:
    log(f"  CLI failed: stderr={proc.stderr[:500]} returncode={proc.returncode}")
log("")

log("="*72)
log(f"OVERALL: Hit@5={hit_rate:.3f} Citation={'PASS' if all_cited else 'FAIL'} NegativeTest={'PASS' if neg_pass else 'FAIL'}")
log("="*72)

# Write to both outs
text = "\n".join(lines)
OUT.write_text(text, encoding="utf-8")
PORTFOLIO_OUT.write_text(text, encoding="utf-8")
print(f"\nWrote {OUT} and {PORTFOLIO_OUT}")
