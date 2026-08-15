# Security Policy

## Supported Versions

This is a research/portfolio project without formal LTS versioning. Security
fixes are applied to `main` only.

| Branch | Supported |
| ------ | --------- |
| `main` | yes       |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.
Instead, report privately via GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)
feature on this repository, or contact the maintainer listed in
`CODEOWNERS` directly.

Please include:

- A description of the vulnerability and its potential impact
- Steps to reproduce
- Any relevant logs, payloads, or PoC code

You should expect an initial response within 5 business days.

## Known Hardening Items (tracked, not yet fixed)

Documented transparently rather than hidden — tracked in `CHANGELOG.md` /
`ROADMAP.md` as they're resolved:

- `.env.example` previously shipped a non-placeholder-looking
  `TMDB_API_KEY` value. Treat any key present in git history as
  compromised: rotate it at themoviedb.org and ensure only
  `your_tmdb_api_key_here`-style placeholders are ever committed.
- `SECRET_KEY` / `JWT_SECRET` default values in `docker-compose.yml` and
  `backend/app/core/config.py` are **development-only fallbacks**. They
  must be overridden via real environment variables (or a secrets
  manager) in any deployed environment — never rely on the checked-in
  default outside local `docker compose up`.
- Dependency vulnerabilities are tracked via `.github/dependabot.yml`
  (added in this PR) rather than manually.

## Secret Handling Guidelines

- Never commit real API keys, tokens, or credentials — use `.env` (git-ignored)
  populated from `.env.example` (placeholders only).
- If a secret is ever committed, rotating the credential is mandatory —
  removing it from a future commit does **not** remove it from git history.
