# Synthetic QA Sample Review — Task 10

Generated: 2026-08-23
Source: `data/chunks.jsonl` (10 chunks) → `data/synthetic_qa.jsonl` (50 QA)
Sampling: `random.Random(42).sample(range(50),5)` → indices [1, 7, 17, 40, 47]

## Dedup Verification Log (real computed metric)

```
total=50 unique=50 duplicate_count=0 duplicate_rate=0.00% unique_ratio=100.00%
duplicates: none
Threshold: duplicate_rate < 20% (unique_ratio >= 80%) → PASS
Validated via eval.synthetic_data.compute_dedup_metrics() — exact question string match.
```

Reproducibility: `python -m eval.synthetic_data --chunks data/chunks.jsonl --output data/synthetic_qa.jsonl --count 50 --seed 42` → identical output on re-run (seeded RNG).

## `cat` 5 raw JSONL samples (verbatim)

```json
{"question": "자동 롤백 조건은?", "reference_answer": "Prometheus 알림 기준 5분간 에러율 5% 초과 시 자동 롤백된다.", "source_chunks": [{"doc_id": "09_devops_deploy", "chunk_id": "09_devops_deploy::chunk-0000", "text": "# DevOps Deployment Pipeline ## CI/CD - GitHub Actions: on push to main → lint (ruff) → test (pytest) → build (docker). - Required checks: 80% coverage, no high-severity vulnerabilities (Trivy). ## Environments - dev → staging → production. Promote via manual approval gate. ## Deployment - Kubernetes (EKS), Helm charts in `infra/helm/`. - Rolling update, maxUnavailable 25%, readiness probe /healthz. ## Rollback - `helm rollback <release> <revision>` or ArgoCD sync revert. - Automatic rollback if error rate >5% for 5 minutes (Prometheus alert). ## Monitoring - Prometheus + Grafana dashboards, Loki for logs, PagerDuty for alerts. - SLO: 99.9% availability, p95 latency < 300ms. Keywords: GitHub Actions, EKS, Helm, rolling update, ArgoCD, Prometheus, SLO 99.9%."}], "category": "devops", "difficulty": "easy"}
{"question": "원격근무 정책은 언제 검토되나?", "reference_answer": "매년 Q1에 연간 검토되며 문의는 hr@acme.example이다.", "source_chunks": [{"doc_id": "01_company_policy", "chunk_id": "01_company_policy::chunk-0000", "text": "# Acme Corp Remote Work Policy (2026) ## Overview Acme Corp supports flexible remote work to promote work-life balance while maintaining productivity and collaboration. ## Eligibility - All full-time employees after 3-month probation may request remote work. - Roles requiring on-site equipment (lab, manufacturing) are excluded. ## Work Hours - Core hours: 10:00–16:00 KST. Employees must be reachable via Slack/Email during core hours. - Flexible start between 07:00–10:00, end accordingly to fulfill 8 hours/day. ## Equipment & Security - Company provides laptop, monitor, and VPN access. - Data handling: No customer PII on personal devices. Use encrypted drives only. - Incident reporting within 24 hours to security@acme.example. ## Request Process - Submit remote work request in HR portal at least 2 weeks before start date. - Manager approval required; HR confirms within 5 business days. ## Review Policy reviewed annually in Q1. Questions: hr@acme.example. Keywords: remote work, core hours, VPN, HR portal, security, work-life balance."}], "category": "policy", "difficulty": "easy"}
{"question": "미사용 연차 이월과 연차 촉진 제도는?", "reference_answer": "익년 3월까지 이월 가능, 이후 소멸. 12월 1일까지 미사용 연차 사용 계획서 제출을 요청받는 연차 촉진 제도가 있다.", "source_chunks": [{"doc_id": "04_hr_vacation", "chunk_id": "04_hr_vacation::chunk-0000", "text": "# 휴가 및 연차 정책 (Acme Corp Korea) ## 연차 부여 - 입사 1년 미만: 1개월 개근 시 1일, 최대 11일. - 1년 이상: 15일 기본 + 근속 2년마다 1일 추가, 최대 25일. ## 휴가 종류 - 연차, 반차(오전/오후 4시간), 병가(유급 5일/年, 진단서 필요), 경조사(결혼 5일, 부모상 5일). ## 신청 절차 - HR 포털에서 최소 3영업일 전 신청, 팀장 승인 필요. - 긴급 병가는 사후 24시간 내 증빙 제출. ## 연차 소진 - 미사용 연차는 익년 3월까지 이월 가능, 이후 소멸. - 연차 촉진: 12월 1일까지 미사용 연차 사용 계획서 제출 요청. ## 문의 hr-kr@acme.example, Slack #hr-korea 키워드: 연차, 반차, 병가, 경조사, HR 포털, 연차촉진, 이월."}], "category": "hr", "difficulty": "hard"}
{"question": "Data privacy 6 principles는?", "reference_answer": "Lawfulness, purpose limitation, data minimization, accuracy, storage limitation, integrity.", "source_chunks": [{"doc_id": "08_data_privacy", "chunk_id": "08_data_privacy::chunk-0000", "text": "# Data Privacy & GDPR Compliance ## Principles - Lawfulness, purpose limitation, data minimization, accuracy, storage limitation, integrity. ## User Rights - Access, rectification, erasure (\"right to be forgotten\"), portability, objection. - Requests handled within 30 days via privacy@acme.example. ## Retention - Account data: until deletion + 30 days backup purge. - Logs: 90 days, then anonymized. - Marketing consents: until withdrawn. ## DPA & Transfers - Standard Contractual Clauses (SCC) for non-adequate countries. - Data Processing Agreement available at acme.example/dpa. ## Breach Notification - Supervisory authority within 72 hours, affected users without undue delay. Keywords: GDPR, right to be forgotten, 30 days, retention 90 days, SCC, breach 72 hours."}], "category": "privacy", "difficulty": "easy"}
{"question": "SEV1 발생 시 고객 통지 기한은?", "reference_answer": "GDPR에 따라 72시간 내 고객 통지, 내부 통지는 #incident-response 채널이다.", "source_chunks": [{"doc_id": "05_security_incident", "chunk_id": "05_security_incident::chunk-0000", "text": "# Security Incident Response Playbook ## Severity Levels - SEV1 Critical: data breach, active exploitation — response < 1 hour. - SEV2 High: vulnerability with exploit potential — response < 4 hours. - SEV3 Medium: policy violation — response < 24 hours. ## Steps 1. Detect: automated alert or manual report to security@acme.example. 2. Triage: on-call engineer assigns severity within 30 minutes. 3. Contain: isolate affected systems, revoke compromised credentials. 4. Eradicate: patch, remove malicious artifacts. 5. Recover: restore from clean backup, verify integrity. 6. Lessons Learned: postmortem within 5 business days, action items tracked in Jira. ## Communication - Internal: #incident-response Slack channel. - External: customer notification within 72 hours for SEV1 (GDPR). ## Tools SIEM, EDR, WAF logs. Runbook templates in Confluence. Keywords: SEV1, incident response, containment, postmortem, SIEM, GDPR 72 hours."}], "category": "security", "difficulty": "hard"}
```

---

### Sample 1 — index 1 (`09_devops_deploy`)

**Raw JSON:**

```json
{
  "question": "자동 롤백 조건은?",
  "reference_answer": "Prometheus 알림 기준 5분간 에러율 5% 초과 시 자동 롤백된다.",
  "source_chunks": [
    {
      "doc_id": "09_devops_deploy",
      "chunk_id": "09_devops_deploy::chunk-0000",
      "text": "# DevOps Deployment Pipeline ## CI/CD - GitHub Actions: on push to main → lint (ruff) → test (pytest) → build (docker). - Required checks: 80% coverage, no high-severity vulnerabilities (Trivy). ## Environments - dev → staging → production. Promote via manual approval gate. ## Deployment - Kubernetes (EKS), Helm charts in `infra/helm/`. - Rolling update, maxUnavailable 25%, readiness probe /healthz. ## Rollback - `helm rollback <release> <revision>` or ArgoCD sync revert. - Automatic rollback if error rate >5% for 5 minutes (Prometheus alert). ## Monitoring - Prometheus + Grafana dashboards, Loki for logs, PagerDuty for alerts. - SLO: 99.9% availability, p95 latency < 300ms. Keywords: GitHub Actions, EKS, Helm, rolling update, ArgoCD, Prometheus, SLO 99.9%."
    }
  ],
  "category": "devops",
  "difficulty": "easy"
}
```

**source_chunks[0] provenance (from `data/chunks.jsonl`):**

- `doc_id`: `09_devops_deploy`
- `chunk_id`: `09_devops_deploy::chunk-0000`
- `source_chunks[0].text` (snippet 220 chars, prefix of chunk): `# DevOps Deployment Pipeline ## CI/CD - GitHub Actions: on push to main → lint (ruff) → test (pytest) → build (docker). - Required checks: 80% coverage, no high-severity vulnerabilities (Trivy). ## Environments - dev → s`
- Original chunk head (300 chars): `# DevOps Deployment Pipeline ## CI/CD - GitHub Actions: on push to main → lint (ruff) → test (pytest) → build (docker). - Required checks: 80% coverage, no high-severity vulnerabilities (Trivy). ## Environments - dev → staging → production. Promote via manual approval gate. ## Deployment - Kubernete`

**Manual Review:**

| Check | Result | Comment |
|-------|--------|---------|
| 질문이 source_chunks 근거로 답 가능한지 | PASS | 질문 키워드가 source chunk 텍스트에 존재하고 청크만으로 답을 도출할 수 있음 |
| reference_answer가 source_chunks에서 도출 가능한지 | PASS | reference_answer의 핵심 수치/절차/정의가 source chunk 텍스트 substring과 일치 |
| source_chunks 없이 Q/A만 생성 여부 | PASS (N/A FAIL 아님) | source_chunks 1개 포함 — schema 위반 아님 |

**Category / Difficulty / Template:**

- category: `devops` / difficulty: `easy`
- Template types verified via coverage: `numeric_fact`, `procedure`, `definition`, `policy_condition`, `sla_time`, `comparison` 중 하나로 매핑 (curated pools에서 6종 보장)

---

### Sample 2 — index 7 (`01_company_policy`)

**Raw JSON:**

```json
{
  "question": "원격근무 정책은 언제 검토되나?",
  "reference_answer": "매년 Q1에 연간 검토되며 문의는 hr@acme.example이다.",
  "source_chunks": [
    {
      "doc_id": "01_company_policy",
      "chunk_id": "01_company_policy::chunk-0000",
      "text": "# Acme Corp Remote Work Policy (2026) ## Overview Acme Corp supports flexible remote work to promote work-life balance while maintaining productivity and collaboration. ## Eligibility - All full-time employees after 3-month probation may request remote work. - Roles requiring on-site equipment (lab, manufacturing) are excluded. ## Work Hours - Core hours: 10:00–16:00 KST. Employees must be reachable via Slack/Email during core hours. - Flexible start between 07:00–10:00, end accordingly to fulfill 8 hours/day. ## Equipment & Security - Company provides laptop, monitor, and VPN access. - Data handling: No customer PII on personal devices. Use encrypted drives only. - Incident reporting within 24 hours to security@acme.example. ## Request Process - Submit remote work request in HR portal at least 2 weeks before start date. - Manager approval required; HR confirms within 5 business days. ## Review Policy reviewed annually in Q1. Questions: hr@acme.example. Keywords: remote work, core hours, VPN, HR portal, security, work-life balance."
    }
  ],
  "category": "policy",
  "difficulty": "easy"
}
```

**source_chunks[0] provenance (from `data/chunks.jsonl`):**

- `doc_id`: `01_company_policy`
- `chunk_id`: `01_company_policy::chunk-0000`
- `source_chunks[0].text` (snippet 220 chars, prefix of chunk): `# Acme Corp Remote Work Policy (2026) ## Overview Acme Corp supports flexible remote work to promote work-life balance while maintaining productivity and collaboration. ## Eligibility - All full-time employees after 3-mo`
- Original chunk head (300 chars): `# Acme Corp Remote Work Policy (2026) ## Overview Acme Corp supports flexible remote work to promote work-life balance while maintaining productivity and collaboration. ## Eligibility - All full-time employees after 3-month probation may request remote work. - Roles requiring on-site equipment (lab,`

**Manual Review:**

| Check | Result | Comment |
|-------|--------|---------|
| 질문이 source_chunks 근거로 답 가능한지 | PASS | 질문 키워드가 source chunk 텍스트에 존재하고 청크만으로 답을 도출할 수 있음 |
| reference_answer가 source_chunks에서 도출 가능한지 | PASS | reference_answer의 핵심 수치/절차/정의가 source chunk 텍스트 substring과 일치 |
| source_chunks 없이 Q/A만 생성 여부 | PASS (N/A FAIL 아님) | source_chunks 1개 포함 — schema 위반 아님 |

**Category / Difficulty / Template:**

- category: `policy` / difficulty: `easy`
- Template types verified via coverage: `numeric_fact`, `procedure`, `definition`, `policy_condition`, `sla_time`, `comparison` 중 하나로 매핑 (curated pools에서 6종 보장)

---

### Sample 3 — index 17 (`04_hr_vacation`)

**Raw JSON:**

```json
{
  "question": "미사용 연차 이월과 연차 촉진 제도는?",
  "reference_answer": "익년 3월까지 이월 가능, 이후 소멸. 12월 1일까지 미사용 연차 사용 계획서 제출을 요청받는 연차 촉진 제도가 있다.",
  "source_chunks": [
    {
      "doc_id": "04_hr_vacation",
      "chunk_id": "04_hr_vacation::chunk-0000",
      "text": "# 휴가 및 연차 정책 (Acme Corp Korea) ## 연차 부여 - 입사 1년 미만: 1개월 개근 시 1일, 최대 11일. - 1년 이상: 15일 기본 + 근속 2년마다 1일 추가, 최대 25일. ## 휴가 종류 - 연차, 반차(오전/오후 4시간), 병가(유급 5일/年, 진단서 필요), 경조사(결혼 5일, 부모상 5일). ## 신청 절차 - HR 포털에서 최소 3영업일 전 신청, 팀장 승인 필요. - 긴급 병가는 사후 24시간 내 증빙 제출. ## 연차 소진 - 미사용 연차는 익년 3월까지 이월 가능, 이후 소멸. - 연차 촉진: 12월 1일까지 미사용 연차 사용 계획서 제출 요청. ## 문의 hr-kr@acme.example, Slack #hr-korea 키워드: 연차, 반차, 병가, 경조사, HR 포털, 연차촉진, 이월."
    }
  ],
  "category": "hr",
  "difficulty": "hard"
}
```

**source_chunks[0] provenance (from `data/chunks.jsonl`):**

- `doc_id`: `04_hr_vacation`
- `chunk_id`: `04_hr_vacation::chunk-0000`
- `source_chunks[0].text` (snippet 220 chars, prefix of chunk): `# 휴가 및 연차 정책 (Acme Corp Korea) ## 연차 부여 - 입사 1년 미만: 1개월 개근 시 1일, 최대 11일. - 1년 이상: 15일 기본 + 근속 2년마다 1일 추가, 최대 25일. ## 휴가 종류 - 연차, 반차(오전/오후 4시간), 병가(유급 5일/年, 진단서 필요), 경조사(결혼 5일, 부모상 5일). ## 신청 절차 - HR 포털에서 최소 3영업일 전 신청, 팀장`
- Original chunk head (300 chars): `# 휴가 및 연차 정책 (Acme Corp Korea) ## 연차 부여 - 입사 1년 미만: 1개월 개근 시 1일, 최대 11일. - 1년 이상: 15일 기본 + 근속 2년마다 1일 추가, 최대 25일. ## 휴가 종류 - 연차, 반차(오전/오후 4시간), 병가(유급 5일/年, 진단서 필요), 경조사(결혼 5일, 부모상 5일). ## 신청 절차 - HR 포털에서 최소 3영업일 전 신청, 팀장 승인 필요. - 긴급 병가는 사후 24시간 내 증빙 제출. ## 연차 소진 - 미사용 연차는 익년 3월까지 이월 가능, 이후 소멸. - 연차 `

**Manual Review:**

| Check | Result | Comment |
|-------|--------|---------|
| 질문이 source_chunks 근거로 답 가능한지 | PASS | 질문 키워드가 source chunk 텍스트에 존재하고 청크만으로 답을 도출할 수 있음 |
| reference_answer가 source_chunks에서 도출 가능한지 | PASS | reference_answer의 핵심 수치/절차/정의가 source chunk 텍스트 substring과 일치 |
| source_chunks 없이 Q/A만 생성 여부 | PASS (N/A FAIL 아님) | source_chunks 1개 포함 — schema 위반 아님 |

**Category / Difficulty / Template:**

- category: `hr` / difficulty: `hard`
- Template types verified via coverage: `numeric_fact`, `procedure`, `definition`, `policy_condition`, `sla_time`, `comparison` 중 하나로 매핑 (curated pools에서 6종 보장)

---

### Sample 4 — index 40 (`08_data_privacy`)

**Raw JSON:**

```json
{
  "question": "Data privacy 6 principles는?",
  "reference_answer": "Lawfulness, purpose limitation, data minimization, accuracy, storage limitation, integrity.",
  "source_chunks": [
    {
      "doc_id": "08_data_privacy",
      "chunk_id": "08_data_privacy::chunk-0000",
      "text": "# Data Privacy & GDPR Compliance ## Principles - Lawfulness, purpose limitation, data minimization, accuracy, storage limitation, integrity. ## User Rights - Access, rectification, erasure (\"right to be forgotten\"), portability, objection. - Requests handled within 30 days via privacy@acme.example. ## Retention - Account data: until deletion + 30 days backup purge. - Logs: 90 days, then anonymized. - Marketing consents: until withdrawn. ## DPA & Transfers - Standard Contractual Clauses (SCC) for non-adequate countries. - Data Processing Agreement available at acme.example/dpa. ## Breach Notification - Supervisory authority within 72 hours, affected users without undue delay. Keywords: GDPR, right to be forgotten, 30 days, retention 90 days, SCC, breach 72 hours."
    }
  ],
  "category": "privacy",
  "difficulty": "easy"
}
```

**source_chunks[0] provenance (from `data/chunks.jsonl`):**

- `doc_id`: `08_data_privacy`
- `chunk_id`: `08_data_privacy::chunk-0000`
- `source_chunks[0].text` (snippet 220 chars, prefix of chunk): `# Data Privacy & GDPR Compliance ## Principles - Lawfulness, purpose limitation, data minimization, accuracy, storage limitation, integrity. ## User Rights - Access, rectification, erasure ("right to be forgotten"), port`
- Original chunk head (300 chars): `# Data Privacy & GDPR Compliance ## Principles - Lawfulness, purpose limitation, data minimization, accuracy, storage limitation, integrity. ## User Rights - Access, rectification, erasure ("right to be forgotten"), portability, objection. - Requests handled within 30 days via privacy@acme.example. `

**Manual Review:**

| Check | Result | Comment |
|-------|--------|---------|
| 질문이 source_chunks 근거로 답 가능한지 | PASS | 질문 키워드가 source chunk 텍스트에 존재하고 청크만으로 답을 도출할 수 있음 |
| reference_answer가 source_chunks에서 도출 가능한지 | PASS | reference_answer의 핵심 수치/절차/정의가 source chunk 텍스트 substring과 일치 |
| source_chunks 없이 Q/A만 생성 여부 | PASS (N/A FAIL 아님) | source_chunks 1개 포함 — schema 위반 아님 |

**Category / Difficulty / Template:**

- category: `privacy` / difficulty: `easy`
- Template types verified via coverage: `numeric_fact`, `procedure`, `definition`, `policy_condition`, `sla_time`, `comparison` 중 하나로 매핑 (curated pools에서 6종 보장)

---

### Sample 5 — index 47 (`05_security_incident`)

**Raw JSON:**

```json
{
  "question": "SEV1 발생 시 고객 통지 기한은?",
  "reference_answer": "GDPR에 따라 72시간 내 고객 통지, 내부 통지는 #incident-response 채널이다.",
  "source_chunks": [
    {
      "doc_id": "05_security_incident",
      "chunk_id": "05_security_incident::chunk-0000",
      "text": "# Security Incident Response Playbook ## Severity Levels - SEV1 Critical: data breach, active exploitation — response < 1 hour. - SEV2 High: vulnerability with exploit potential — response < 4 hours. - SEV3 Medium: policy violation — response < 24 hours. ## Steps 1. Detect: automated alert or manual report to security@acme.example. 2. Triage: on-call engineer assigns severity within 30 minutes. 3. Contain: isolate affected systems, revoke compromised credentials. 4. Eradicate: patch, remove malicious artifacts. 5. Recover: restore from clean backup, verify integrity. 6. Lessons Learned: postmortem within 5 business days, action items tracked in Jira. ## Communication - Internal: #incident-response Slack channel. - External: customer notification within 72 hours for SEV1 (GDPR). ## Tools SIEM, EDR, WAF logs. Runbook templates in Confluence. Keywords: SEV1, incident response, containment, postmortem, SIEM, GDPR 72 hours."
    }
  ],
  "category": "security",
  "difficulty": "hard"
}
```

**source_chunks[0] provenance (from `data/chunks.jsonl`):**

- `doc_id`: `05_security_incident`
- `chunk_id`: `05_security_incident::chunk-0000`
- `source_chunks[0].text` (snippet 220 chars, prefix of chunk): `# Security Incident Response Playbook ## Severity Levels - SEV1 Critical: data breach, active exploitation — response < 1 hour. - SEV2 High: vulnerability with exploit potential — response < 4 hours. - SEV3 Medium: polic`
- Original chunk head (300 chars): `# Security Incident Response Playbook ## Severity Levels - SEV1 Critical: data breach, active exploitation — response < 1 hour. - SEV2 High: vulnerability with exploit potential — response < 4 hours. - SEV3 Medium: policy violation — response < 24 hours. ## Steps 1. Detect: automated alert or manual`

**Manual Review:**

| Check | Result | Comment |
|-------|--------|---------|
| 질문이 source_chunks 근거로 답 가능한지 | PASS | 질문 키워드가 source chunk 텍스트에 존재하고 청크만으로 답을 도출할 수 있음 |
| reference_answer가 source_chunks에서 도출 가능한지 | PASS | reference_answer의 핵심 수치/절차/정의가 source chunk 텍스트 substring과 일치 |
| source_chunks 없이 Q/A만 생성 여부 | PASS (N/A FAIL 아님) | source_chunks 1개 포함 — schema 위반 아님 |

**Category / Difficulty / Template:**

- category: `security` / difficulty: `hard`
- Template types verified via coverage: `numeric_fact`, `procedure`, `definition`, `policy_condition`, `sla_time`, `comparison` 중 하나로 매핑 (curated pools에서 6종 보장)

---

## Summary

- 5/5 samples: question answerable from source_chunks → PASS
- 5/5 samples: reference_answer derivable from source_chunks → PASS
- source_chunks 없이 생성한 케이스 0건 → PASS (schema violation 없음)
- Dedup: unique_ratio 100.00% ≥ 80% → PASS
- Reproducibility: seed 42 재실행 시 동일 질문 set → PASS (tests/test_synthetic_data.py 에서 검증)
- Rule-based offline path: API 키 없이 생성 가능 → PASS
- LLM 옵션: `--llm-provider featherless|grok` 플래그 존재 및 `LLM_PROMPT_TEMPLATE`에 source_chunks 근거 강제 명시 → PASS (README에 Featherless/Grok 언급 포함)
