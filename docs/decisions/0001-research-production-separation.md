# ADR 0001: Strict Research/Production Separation

## Status
Accepted (already implemented prior to this hardening pass — documented
here retroactively per the Phase 1 audit).

## Context
ML portfolio repositories commonly blur research and production code:
training scripts import from the API server, or the API server imports a
notebook's helper module directly. This makes both harder to reason about
— production inherits research's instability, and research inherits
production's performance/API constraints.

## Decision
`research/` and the production layers (`backend/`, `ml/`, `model_server/`,
`apps/web/`, `gateway/`) are two independent import graphs:
- Nothing under `research/` is ever imported by production code.
- Nothing under production is ever imported by `research/`.
- The only connection is a **file-level artifact hand-off**: trained
  weights and processed datasets move from `research/` into `artifacts/`
  and `data/` (currently by hand; `scripts/export_model.py` is planned to
  formalize this — see `ROADMAP.md`).

Verified in the Phase 1 audit by grepping every `import` statement in the
codebase for cross-boundary references — none were found.

## Consequences
- ✅ Production can upgrade dependencies, refactor, or change frameworks
  without touching research code, and vice versa.
- ✅ Research can be messy/exploratory without that risk leaking into the
  deployed system.
- ⚠️ Requires discipline to maintain — any future PR that imports
  `research.*` from `backend/`, `ml/`, or `model_server/` (or the reverse)
  should be rejected in review; the PR template checklist includes this.
- ⚠️ Necessitates a deliberate promotion step (currently manual) to move
  a trained model from research into production — see ADR 0002 and the
  data/artifact duplication note in `ROADMAP.md`.
