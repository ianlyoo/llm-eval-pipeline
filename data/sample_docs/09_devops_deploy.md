# DevOps Deployment Pipeline

## CI/CD
- GitHub Actions: on push to main → lint (ruff) → test (pytest) → build (docker).
- Required checks: 80% coverage, no high-severity vulnerabilities (Trivy).

## Environments
- dev → staging → production. Promote via manual approval gate.

## Deployment
- Kubernetes (EKS), Helm charts in `infra/helm/`.
- Rolling update, maxUnavailable 25%, readiness probe /healthz.

## Rollback
- `helm rollback <release> <revision>` or ArgoCD sync revert.
- Automatic rollback if error rate >5% for 5 minutes (Prometheus alert).

## Monitoring
- Prometheus + Grafana dashboards, Loki for logs, PagerDuty for alerts.
- SLO: 99.9% availability, p95 latency < 300ms.

Keywords: GitHub Actions, EKS, Helm, rolling update, ArgoCD, Prometheus, SLO 99.9%.
