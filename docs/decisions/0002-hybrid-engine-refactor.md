# ADR 0002: Production `HybridEngine` is a Hardened Refactor, Not a Copy

## Status
Accepted (documented retroactively per the Phase 1 audit).

## Context
`research/hybrid/hybrid_engine.py` and `research/hybrid/cold_start.py`
implement the collaborative + content + popularity blending logic used
during experimentation. `ml/models/ncf/hybrid_engine.py` and
`ml/models/ncf/cold_start.py` implement the same *concept* for production.

A naive read might assume this is unwanted duplication that should be
"deduplicated" into one shared module.

## Decision
Keep them as two separate implementations, on purpose:
- The production version strips research-only docstrings/exploration code.
- The production `.load()` path adds existence checks and a pickle
  compatibility shim for a historical module path
  (`hybrid.cold_start` → `ml.models.ncf.cold_start`), so that artifacts
  pickled during earlier research runs still deserialize correctly in
  production without forcing a re-train.
- Production deliberately does **not** import `research.hybrid` (see ADR
  0001) — sharing a module would violate the research/production boundary.

## Consequences
- ✅ Production can harden its copy (input validation, error handling)
  without those changes needing to also work inside exploratory notebooks.
- ✅ Backward-compatible loading of older pickled artifacts is possible
  without a research-layer dependency.
- ⚠️ The two files **will drift** over time by design. When a genuine
  algorithmic change is made in research and should reach production, it
  must be **manually ported**, not "synced" — this is a conscious
  trade-off, not an oversight. Document non-trivial ports in
  `CHANGELOG.md`.
- ⚠️ A future contributor unfamiliar with this ADR may try to merge the
  two files. This ADR exists specifically to prevent that.
