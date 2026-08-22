# 오답노트 — Rule Filter 17 Fail 중 10선

> Source: `out/rule_report.json` (total 50, passed 33, failed 17) + `data/synthetic_qa.jsonl` + `out/judge_scores.jsonl` 샘플
> All failures are `expected_keyword` — numeric/groundedness는 통과했으나 질문 키워드가 답변에 문자 그대로 없음.
> Faithfulness 평균 0.8034지만 keyword-overlap 기반 relevance는 낮음 → retrieval/rewrite 필요.

| # | 질문 | 예상 답변 (reference) | 실패유형 | 원인 | 개선 |
|---|------|----------------------|---------|------|------|
| 1 | What is the Atlas API rate limit and what happens when exceeded? | 100 requests/minute per key (1000/day Free tier); 429 with Retry-After header. | expected_keyword | 질문 키워드 `atlas/api/rate/limit/happens/exceeded`가 답변에 없음 — 답변은 수치 위주, 키워드 미포함 | 답변 템플릿에 `Atlas API rate limit` 명시, 키워드 보강 프롬프트 추가 |
| 2 | 원격근무 정책은 언제 검토되나? | 매년 Q1에 연간 검토되며 문의는 hr@acme.example이다. | expected_keyword | `원격근무/검토되나` 토큰이 답변 정규화 후 불일치 — `검토` vs `검토되며` 조사 처리 한계 | _normalize_keyword → 형태소 단위 매칭으로 개선, 답변에 질문 주어 반복 |
| 3 | 입사 1년 이상 직원의 연차는 며칠인가? | 기본 15일에 근속 2년마다 1일 추가, 최대 25일이다. | expected_keyword | 질문의 `입사/1년/이상/직원/연차/며칠` 중 답변이 `기본 15일...`로 키워드 미포함 — 질문-답변 길이 불균형 | QA 생성 시 answer에 question noun 재삽입 규칙 (예: "입사 1년 이상 연차는 ...") |
| 4 | ML training optimizer와 learning rate는 무엇인가? | AdamW, lr 3e-4, weight_decay 0.01, cosine scheduler with warmup 500 steps. | expected_keyword | `ml/training/optimizer/learning/rate` 중 `ml`/`training`이 답변에 없음 — 약어/영문 케이스 불일치 | 동의어 사전(`ml`→`AdamW`, `training`→`ML`) 또는 embedding 기반 relevance로 교체 |
| 5 | What does HTTP 429 mean in Atlas API? | Rate limited — Retry-After header indicates seconds to wait. | expected_keyword | `http/429/atlas/api` 키워드 미포함 — 답변은 `Rate limited`만, 컨텍스트 생략 | 답변에 `HTTP 429 in Atlas API is rate limited` 형태로 질문 컨텍스트 유지 |
| 6 | GDPR에서 user rights 5가지는? | Access, rectification, erasure(right to be forgotten), portability, objection이며 30일 내 처리한다. | expected_keyword | `gdpr/user/rights/5가지` 중 `gdpr`/`5가지` 미포함 — 답변은 영문 rights만 | 다국어 키워드 매칭 강화, 답변 첫 문장에 `GDPR user rights 5가지` 헤더 포함 |
| 7 | 경조사 휴가는 얼마인가? | 결혼 5일, 부모상 5일이다. | expected_keyword | `경조사/휴가`가 답변 `결혼/부모상`으로 치환되어 키워드 불일치 | 동의어 확장: `경조사`→`결혼/부모상` 매핑, 또는 답변에 `경조사 휴가는 ...` 서두 추가 |
| 8 | 경비 청구 절차 3단계는? | 1) 영수증 스캔 expenses.acme.example 업로드 2) 매니저→재무 승인(영업일 2일) 3) 매월 15일/말일 정산 지급이다. | expected_keyword | `경비/청구/절차/3단계` 중 숫자 `3` 정규화 차이, `경비 청구` 복합명사 분리 | 절차형 답변 템플릿에 `경비 청구 절차 3단계는 다음과 같다:` 헤더 강제 |
| 9 | 월 경비 한도와 금지 항목은? | 팀원 월 50만원 매니저 100만원, 주류/개인용품/벌금은 청구 불가이다. | expected_keyword | `월/경비/한도/금지/항목` 중 답변이 수치 나열 위주, 키워드 직접 포함 없음 | 한도/금지 항목 답변에 카테고리 라벨 명시: `월 경비 한도: ... 금지 항목: ...` |
| 10 | Incident response의 containment 이후 단계는 무엇인가? | Eradicate(패치/악성 아티팩트 제거) → Recover(클린 백업 복구 및 무결성 검증) → Lessons Learned(5영업일 내 포스트모템) 순이다. | expected_keyword | `incident/response/containment/이후/단계` 중 `incident/response` 영문이 한글 답변에 없음 | 영문 전문용어 답변에 병기: `Incident response containment 이후 ...` |

## 요약 통계

- 실패 17개 전부가 `expected_keyword` (단일 taxonomy) — citation/coverage/empty/formatting/unsupported는 0 fail
- Hit@5 proxy = 66.0% (33/50), Faithfulness = 0.8034, Failure rate = 34.0% — `eval/metrics.py` 기준
- Judge 30샘플 평균 relevance는 1~2/5가 다수 (엄격한 토큰 매칭 때문) — embedding 기반 relevance 도입 시 개선 여지

## 개선 액션 (다음 합성 배치에 반영)

1. **Answer rewriting**: 생성 후 `check_expected_keyword`로 self-check, 실패 시 질문 명사를 답변 첫 문장에 삽입
2. **동의어 사전**: `경조사↔결혼/부모상`, `원격근무↔remote work`, `ML↔AdamW` 등 15쌍 추가
3. **Retrieval proxy 교정**: Hit@5 proxy를 rule 외에 embedding cosine >0.7로도 계산 (metrics.py 확장)
4. **Judge relevance 분리**: rule의 keyword overlap은 필터용, judge relevance는 별도 임베딩 스코어로 이원화
