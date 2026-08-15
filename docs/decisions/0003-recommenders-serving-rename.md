# ADR 0003: `models/` → `recommenders/`, `inference/` → `serving/` Renames

## Status
Accepted, implemented in Phase 4.

## Context
An external review of the Phase 3 architecture tree pointed out that
`production/models/` was ambiguous — it could mean "PyTorch model
definitions" or "recommendation algorithms," and a reader wouldn't know
which without opening the folder. Similarly, `production/inference/`
doesn't match the more common industry term for this kind of component.

## Decision
- `ml/models/ncf/` → `production/recommenders/{ncf,hybrid,content,popularity}/`
  — also physically split (not just renamed) into one package per
  algorithm, since the original module bundled the NCF architecture,
  the hybrid blending logic, and both cold-start engines into a single
  `ncf/` folder.
- `model_server/` → `production/serving/` — matches the common
  "model serving" terminology used elsewhere in MLOps tooling
  (BentoML, TorchServe, KServe all use "serving").
- `gateway/fastapi_gateway/` → `production/gateway-optional/` — makes
  its non-default status visible from the path itself (see ADR-level
  note: this gateway is not wired into `docker-compose.yml`).

## Consequences
- ✅ Directory names now communicate intent without opening files.
- ✅ The recommenders split makes `docs/decisions/0002` (hybrid engine
  refactor) easier to reason about — `popularity/` and `content/` are
  now genuinely separate, independently testable units instead of two
  classes living in one file.
- ⚠️ Required a real pickle-compatibility shim
  (`production/recommenders/hybrid/_pickle_compat.py`) so that
  artifacts pickled against the historical `hybrid.cold_start` module
  path still deserialize after the split — this is more than a
  find-and-replace rename, it's a genuine (small) code change, and is
  called out explicitly rather than buried in a mechanical diff.
- ⚠️ Two `sys.path` depth bugs were introduced and fixed as part of
  this move: both `production/backend/app/main.py` and
  `production/serving/app/main.py` are now one directory deeper than
  their pre-move locations, so their `parents[N]` sys.path insert had
  to move from `parents[2]` to `parents[3]`. This was flagged as the
  #1 risk in the Phase 1 audit (§3.9) specifically because it's easy
  to move a file and forget to update this — three similar test-file
  sys.path inserts (`production/tests/unit/test_api.py`, `test_ml.py`,
  `test_metrics.py`) had the identical bug and were fixed the same way.
