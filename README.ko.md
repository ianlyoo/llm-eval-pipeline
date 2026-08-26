# llm-eval-pipeline

[English](README.md)

[![CI](https://github.com/ianlyoo/llm-eval-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/ianlyoo/llm-eval-pipeline/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release: v0.1.0](https://img.shields.io/badge/Release-v0.1.0-blue.svg)](https://github.com/ianlyoo/llm-eval-pipeline/releases/tag/v0.1.0)
[![Pages](https://img.shields.io/badge/Pages-live-brightgreen.svg)](https://ianlyoo.github.io/llm-eval-pipeline/)

Offline Self-Evolve pipeline using synthetic data generation — evaluate before/after with Rule+LLM Judge and lm-eval harness in measured workloads.

> 이 저장소는 오프라인 Self-Evolve 평가 허브입니다. 합성데이터 생성과 Rule/LLM Judge 검증을 거쳐 taxonomy로 실패를 축적하고 lm-eval 하네스로 전후 비교를 측정합니다. 온라인 서빙은 [rag-ops-console](https://github.com/ianlyoo/rag-ops-console)에서 운영합니다.

## 빠른 시작 — synthetic-data와 llm-evaluation

```bash
gh release download v0.1.0 --pattern "*.tgz" --repo ianlyoo/llm-eval-pipeline
npm install ./llm-eval-pipeline-*.tgz
```

```bash
git clone https://github.com/ianlyoo/llm-eval-pipeline.git
cd llm-eval-pipeline
bun install --frozen-lockfile
bun run build
```

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

```bash
python -m eval.synthetic_data --chunks data/chunks.jsonl --output data/synthetic_qa.jsonl --count 50 --seed 42
python -m eval.metrics
```

## 활용 사례 — self-evolution과 offline-evaluation

- 오답 기반 다음 배치 합성데이터 생성
- Rule과 LLM Judge를 통한 근거 충분성 검증
- Pytorch + lm-eval harness로 학습 전후 정량 비교
- Deterministic proxy 지표로 CI 회귀 검증

## 아키텍처 — llm-judge 파이프라인

`청크 → 합성데이터 → Rule 필터 → LLM Judge → taxonomy → lm-eval`

- `eval/synthetic_data.py` — 6종 템플릿, `source_chunks` 필수
- `eval/rule_filter.py` — 결정론적 검증
- `eval/llm_judge.py` — 근거 충분성 판정
- `eval/metrics.py` — taxonomy 집계
- `eval/pytorch_experiment.py` — lm-eval 전후 비교

## 벤치마크 — 측정된 워크로드에서의 lm-eval 하네스 평가

측정된 베이스라인 (Deterministic proxy, 50 샘플, seed 42): `Hit@5 0.66 / faithfulness 0.8034`. Target은 투영치이며 의도적으로 미제공입니다.

Limitations adjacent to benchmark: synthetic seed 42, one run per condition, 50 samples, token-overlap proxy, no training improvement measured, `out/improved_metrics.json` 없음 — fail-closed, policy may change, no billing.

## 라이선스

MIT — [LICENSE](LICENSE)를 참고하세요.
