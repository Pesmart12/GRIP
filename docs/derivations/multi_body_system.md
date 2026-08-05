# Multiple bodies (still unconstrained)

## Why this step has no new physics

Step 3 in the build order adds multiple rigid bodies, but they don't
interact yet — no contact, no coupling. Every body evolves under its own
`step_body` exactly as validated in steps 1–2; "the system"
is just bookkeeping over a collection of independent single-body
problems.

## Representation

`std::vector<RigidBodyState>` + `std::vector<RigidBodyParams>`, indexed
together — the same split as single-body state/params (step 2), applied
at the collection level: a rollout only ever evolves the states vector,
params is invariant data threaded through unchanged.

System state `X` is the concatenation `PackSystem`/`UnpackSystem`
(`core/rigid_body.hpp`) produce: body `i` occupies `[6i, 6i+6)`, each
body's own block laid out per the single-body `Pack`/`Unpack` convention
(`q` then `v`). This is a direct extension of the single-body stacking
convention from `docs/derivations/integrator_jacobians.md`, not a new
one.

## The system Jacobian is exactly block-diagonal

Since no force couples body `i` to body `j`, `∂X_{t+1}/∂X_t` and
`∂X_{t+1}/∂U` are block-diagonal by construction: block `i` is exactly
the per-body `StepJacobians` from step 2, placed at `(6i, 6i)` in
`dX_dX` and `(6i, 3i)` in `dX_dU`. Off-diagonal blocks are exactly zero —
not approximately, exactly, since nothing in the update for body `i`
reads body `j`'s state or control at all. `test_integrator_system_jacobians.cpp`
checks this directly, same pattern as the exact-zero gravity check in
step 2.

## Why this doesn't need to be rebuilt later

Two things could change this structure, and neither one touches the
block-diagonal *assembly* itself:

- **A body's own force becomes state-dependent** (step 5's penalty
  contact against a static half-plane). This only changes what's inside
  that body's diagonal block, through the `ForceJacobian` hook already
  built in step 2 — the per-body chain rule already handles `∂f/∂q ≠ 0`.
  The block-diagonal placement code doesn't change.
- **Two bodies actually contact each other.** This is the only thing
  that would introduce a genuine off-diagonal term. It's handled later
  (steps 6–7) by the implicit function theorem through the contact
  solver's fixed point — a structurally different gradient computation,
  not an extension of this analytic chain-rule assembly. So building the
  block-diagonal system Jacobian now doesn't get superseded by that path;
  the two coexist for different parts of the problem (free-body dynamics
  vs. contact response).
