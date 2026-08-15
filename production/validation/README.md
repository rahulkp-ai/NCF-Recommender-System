# `production/validation/`

## Phase 5 finding: intentionally still empty

Checked what this folder would contain (cross-field/business-rule
validation beyond what request schemas cover) and found the gap doesn't
currently exist: every endpoint already validates its inputs via FastAPI
`Query(..., ge=..., le=..., min_length=...)` constraints and Pydantic
request schemas (`production/backend/app/schemas/`) — e.g. `k` bounds on
recommendation endpoints, `min_length=1` on search queries, email/type
validation on registration.

Adding a `validation/` module today would mean either wrapping that
existing, working validation in an unnecessary indirection layer, or
shipping empty placeholder code — both are worse than leaving this
folder honestly empty with this note, consistent with
`docs/decisions/0004-deferred-enterprise-patterns.md`'s principle of not
scaffolding folders without real content behind them.

## When to actually add something here
When a validation rule spans multiple fields or requires a DB lookup to
evaluate (e.g. "rating can only be submitted for a movie the user has
already interacted with via 'click'") — that kind of rule doesn't fit
naturally in a Pydantic schema or a `Query()` constraint, and is the
concrete trigger for adding real code here.
