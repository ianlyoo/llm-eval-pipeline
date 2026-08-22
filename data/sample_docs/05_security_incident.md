# Security Incident Response Playbook

## Severity Levels
- SEV1 Critical: data breach, active exploitation — response < 1 hour.
- SEV2 High: vulnerability with exploit potential — response < 4 hours.
- SEV3 Medium: policy violation — response < 24 hours.

## Steps
1. Detect: automated alert or manual report to security@acme.example.
2. Triage: on-call engineer assigns severity within 30 minutes.
3. Contain: isolate affected systems, revoke compromised credentials.
4. Eradicate: patch, remove malicious artifacts.
5. Recover: restore from clean backup, verify integrity.
6. Lessons Learned: postmortem within 5 business days, action items tracked in Jira.

## Communication
- Internal: #incident-response Slack channel.
- External: customer notification within 72 hours for SEV1 (GDPR).

## Tools
SIEM, EDR, WAF logs. Runbook templates in Confluence.

Keywords: SEV1, incident response, containment, postmortem, SIEM, GDPR 72 hours.
