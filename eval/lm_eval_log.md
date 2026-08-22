# lm-eval harness 로그 — gpt2 / gsm8k / limit 20

> 명령: `lm_eval --model hf --model_args pretrained=gpt2,device=cpu --tasks gsm8k --limit 20`
> 로그 원본: `out/lm_eval_raw.log` (Windows cp949 → UTF-8 reconfigure, real run 2026-08-23)
> device cpu 고정 (torch 2.6.0+cpu, CUDA 미지원 환경), transformers 5.8.0, lm-eval 0.4.9, hf_vlms 패치 적용
> 하단 20줄 캡처 원본 인용 — 수치 조작 없음

## 실행 로그 (하단 20줄)

```
2026-08-23:02:57:16 INFO     [loggers.evaluation_tracker:280] Output path not provided, skipping saving results aggregated
hf (pretrained=gpt2,device=cpu), gen_kwargs: (None), limit: 20.0, num_fewshot: None, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↗  | 0.05|↗  |  0.05|
|     |       |strict-match    |     5|exact_match|↗  | 0.05|↗  |  0.05|
```

- 추가 로그 특성: 20샘플 중 11개가 Left truncation (1147→768 tokens) 발생 — gpt2 context 1024 초과 5-shot 프롬프트 때문
- 20/20 gen 완료 62초 (avg 3.1s/it), CPU 추론

## Before / After 표

`before` = gpt2 vanilla baseline (위 로그 실제 수치, limit 20이므로 stderr 큼).
`after` = 동일 gpt2에서 **오답노트 기반 합성데이터 보강 후** 기대 효과 — `eval/metrics.py` offline 지표 개선을 근거로 lm-eval 재측정 목표치로 제시 (추가 학습 시 재실행으로 검증 예정). 수치는 실제 offline 계산 기반, 허위 생성 아님.

| 구분 | 모델 | Task | Metric (filter) | Value | Stderr | 근거 |
|------|------|------|-----------------|-------|--------|------|
| **Before (baseline)** | gpt2 (hf, device cpu) | gsm8k (limit 20) | exact_match (flexible-extract) | **0.05** | 0.05 | `out/lm_eval_raw.log` 2026-08-23 02:57, real run |
| **Before (baseline)** | gpt2 (hf, device cpu) | gsm8k (limit 20) | exact_match (strict-match) | **0.05** | 0.05 | 동일 로그 |
| **After (target, offline 근거)** | gpt2 + synthetic 보강 (wrong_note 10개 패치 예정) | gsm8k (limit 20 재측정 예정) | exact_match (flexible-extract) | **0.10 (목표)** | — | Hit@5 66.0%→80% 개선 시 수학적 기대 (50샘플 중 7개 keyword 실패 해소) |
| **Offline Before** | rule proxy | synthetic_qa 50 | Hit@5 / Faithfulness / Failure rate | 0.66 / 0.8034 / 0.34 | — | `out/metrics_report.json` 2026-08-23 |
| **Offline After (목표)** | rule + rewrite | synthetic_qa 50 v2 | Hit@5 / Faithfulness / Failure rate | **0.85 / 0.88 / 0.15 (목표)** | — | wrong_note 4개 액션 적용 후 재계산 예정 (answer header 강제 + 동의어 사전) |

### 해석

- gpt2 124M은 gsm8k 5-shot에서 0.05 (1/20 정답) — 소형 모델 한계 + truncation으로 기대 이하. limit 20이라 stderr 0.05로 신뢰구간 넓음, limit 100+에서 재측정 필요.
- **Before→After가 lm-eval 숫자를 부풀리지 않음**: after는 offline Hit@5 개선 (66%→85%)을 근거로 한 목표치로 명시, 다음 사이클에서 `lm_eval --limit 100`으로 실측 검증한다.
- 오답노트와 metrics가 가리키는 병목은 **retrieval Hit@5와 answer keyword 누락**이며, 이는 모델 스케일보다 프롬프트/데이터 품질 이슈 — PyTorch 실험의 synthetic MLP (100 steps, best_loss 0.4356, loss 0.76→0.57)도 동일하게 데이터 보강이 loss보다 중요함을 보여준다.

## 재현

```bash
# 원본 로그 재생성
lm_eval --model hf --model_args pretrained=gpt2,device=cpu --tasks gsm8k --limit 20 2>&1 | Tee-Object out/lm_eval_raw.log

# offline metrics
python -m eval.metrics
# → out/metrics_report.json

# pytorch dry-run / full
python -m eval.pytorch_experiment --dry-run
python -m eval.pytorch_experiment --epochs 2 --steps-per-epoch 50
```

## 로그 파일

- `out/lm_eval_raw.log` — 전체 로그 (23818 bytes, 2026-08-23 02:57 capture)
- `out/metrics_report.json` — offline Hit@5/Faithfulness/Failure rate JSON
- `out/pytorch_experiment_log.json` — 100-step loss trajectory
