# Contributing to NCF Recommender System

Thanks for considering a contribution. This repository is split into two
independent worlds that must never import from each other — please respect
that boundary in any PR:

- **`research/`** — exploratory, reproducible ML research (notebooks,
  experiments, ablations, the thesis, the paper). Nothing outside
  `research/` may depend on it.
- **`production/`-equivalent layers** (`production/backend/`, `production/recommenders/`, `production/serving/`,
  `production/frontend/`, `production/gateway-optional/`) — the deployed system. It only ever *consumes*
  artifacts exported from research (trained weights, processed data), never
  research source code.

## Getting started

1. Fork and clone the repo.
2. Read `docs/developer-guide/getting-started.md` for environment setup.
3. Run `make install` (or the platform-specific step in that guide).
4. Run `make test` and `make lint` before opening a PR — both must pass in CI.

## Workflow

1. Open an issue first for anything non-trivial (bug fixes under ~20 lines
   are fine to PR directly).
2. Branch from `main`: `git checkout -b feat/short-description` (or `fix/…`,
   `docs/…`, `research/…`).
3. Keep commits scoped and use
   [Conventional Commits](https://www.conventionalcommits.org/) style
   messages (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
4. Add or update tests for any behavior change. Coverage should not drop.
5. Update the relevant `README.md` / `docs/` page if you change how a
   component is used or configured.
6. Open a PR against `main` using the PR template — CI (`ci.yaml`) must be
   green before review.

## Code style

- Python: formatted and linted with `ruff` (`ruff format .` / `ruff check .`),
  type-checked with `mypy` where types are present. Config lives in
  `pyproject.toml` / `ruff.toml` / `mypy.ini`.
- TypeScript (`production/frontend/`): `eslint` + `prettier` via `npm run lint`.
- Pre-commit hooks (`.pre-commit-config.yaml`) run these automatically —
  install with `pre-commit install` after cloning.

## Reporting bugs / requesting features

Use the issue templates under `.github/ISSUE_TEMPLATE/`. Please include
repro steps for bugs and the problem (not just the solution) for feature
requests.

## Security issues

Do **not** open a public issue for a security vulnerability — see
`SECURITY.md` for the private disclosure process.

## Code of Conduct

This project follows the `CODE_OF_CONDUCT.md`. Be respectful, be direct,
assume good faith.
