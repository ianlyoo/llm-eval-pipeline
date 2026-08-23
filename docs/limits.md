# llm-eval-pipeline — 한계와 트레이드오프 (Offline Boundary)

## 1. Offline / Online 경계

- **본 레포(A)는 offline 품질** — 합성 데이터 생성, Rule 6종, 목 LLM Judge, Hit@5 proxy/faithfulness, PyTorch tiny, lm-eval gsm8k. 네트워크 검색(RAG)이나 운영 모니터링은 하지 않는다.
- **B(rag-ops-console)는 online 운영** — Chroma VectorDB, LangGraph StateGraph, RAGAS(context_precision/recall/faithfulness), Sheets, Streamlit. A의 `data/synthetic_qa.jsonl`을 consume만 하며, A 코드/로직을 복사하지 않는다 (`diff -r llm-eval-pipeline/eval rag-ops-console/eval` => only- diverge).
- **이유**: offline에서 RAGAS LLM evaluator를 돌리면 API 키·비용·재현성 불안정. A는 deterministic(규칙/토큰 overlap) 지표로 고정, B에서 LLM 필요 시 fallback으로 분리.

## 2. Hit@5 proxy vs 실 검색

- **선택**: 실제 Chroma retrieval 없이 `check_expected_keyword` pass를 Hit@5로 proxy.
- **장점**: CI 무거움 없음, 50개 전수 deterministic, 0.66으로 병목 단일 규칙 가시화.
- **한계**: lexical keyword 매칭은 동의어/한영 혼용(`Atlas API rate limit` vs `100 requests/minute`)에서 과소평가. B의 Chroma cosine(0.42 precision)이 더 현실적.
- **다음**: `eval/metrics.py`에 embedding cosine Hit@5 병렬 계산, proxy vs real 비교 테이블 추가. 합성 v2에서 answer header 강제+동의어 15쌍으로 Target 0.85 (Projected, not measured) 회귀 — 현재 Measured 0.66은 Deterministic Proxy.

## 3. Faithfulness blended 가중치

- **선택**: numeric 50% + token 50%. numeric missing시 0.5 페널티.
- **장점**: `999조` 같은 환각 수치를 즉시 잡음 (`test_unsupported_answer_fail_fake_numeric`).
- **한계**: token overlap은 `len>=2` 필터만, 불용어 제거·어간 처리 없음 → 0.8034가 약간 관대. B의 RAGAS faithfulness 0.9953(토큰 set)도 동일 한계.
- **트레이드오프**: 엄격한 NLI(자연어 추론)로 올리면 비용↑, 현재 blended로 CI 경량 유지.

## 4. LLM Judge 모델 분산 한계

- **선택**: `deterministic_judge` + `profile=strict/lenient` + `noise_rng` 로 variance/disagreement 근사, real LLM 호출 없음.
- **한계**: 진짜 모델 분산(temperature, prompt)은 더 크다. 현재 variance <1.0 PASS는 목 분산, 실제 GPT/Claude 분산과 상관 낮음. threshold 1로 disagreement 강제 시뮬레이션도 실 모델 교체 실험으로 대체 필요.
- **대응**: `eval/llm_judge.py`를 인터페이스로 유지, API 키 주입 시 동일 schema(score 1-5+reason)로 교체 가능.

## 5. 규모·재현성 한계

- **규모**: 합성 50개, lm-eval 20샘플(gpt2 124M) — stderr 0.05로 신뢰구간 넓음. limit 100으로 확장 시 before/after 0.05->0.10 유의성 검증 필요.
- **PyTorch tiny**: 16->32->2 MLP, 합성 `sum(x)>0` 태스크 — 학습 가능 검증용, 실 모델 학습과 무관. seed 42 deterministic, cuDNN deterministic 로그만.
- **환경**: Python 3.13.3 / torch 2.6.0+cpu / transformers 5.8.0 / lm-eval 0.4.9, Windows cp949→UTF-8 패치. 다른 OS/버전에서 0.05 미세 변동 가능.

## 6. 개선 로드맵 (A 전용) — Target은 Projected, Measured는 Deterministic Proxy 0.66/0.8034

1. 합성 v2: `check_expected_keyword` self-rewrite로 오답 10 재생성 -> Target Hit@5 0.85 (Projected, not measured) 검증 — Measured 0.66은 유지, 개선 후 실측으로 `out/improved_metrics.json` 생성
2. embedding Hit@5 추가 (sentence-transformers) vs Deterministic Proxy 비교
3. Judge를 `openai` 호출로 교체 시 variance 재측정 (Real LLM variance, not Simulation)
4. lm-eval limit 100 재실행 (23818 bytes -> 100샘플) — Actual training result 측정
