# Atlas API — Developer Guide

## Base URL
`https://api.atlas.example/v1`

## Authentication
- Header: `Authorization: Bearer <API_KEY>`
- API keys rotate every 90 days; old keys have 7-day grace period.

## Rate Limits
- 100 requests/minute per key, 1000/day on Free tier.
- Exceeding returns HTTP 429 with `Retry-After` header (seconds).

## Endpoints
### GET /projects
List projects. Query params: `page` (int), `limit` (max 100), `status` (active|archived).

### POST /projects
Create project. Body: `{ "name": string, "description": string }` → 201 with `{ id, name }`.

### GET /projects/{id}/tasks
List tasks. Supports filter `assignee` and `priority` (low|medium|high).

## Errors
| Code | Meaning |
|------|---------|
| 400 | Bad request (validation failed) |
| 401 | Invalid API key |
| 429 | Rate limited |
| 500 | Internal error |

## SDKs
Python, TypeScript, Go SDKs available at github.com/atlas-sdk.

Keywords: Atlas API, Bearer token, rate limit 100/min, Retry-After, projects, tasks, SDK.
