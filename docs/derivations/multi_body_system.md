# Multiple bodies (still unconstrained)

Symbols follow `docs/derivations/notation.md`.

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

System state `Z` is the concatenation `PackSystem`/`UnpackSystem`
(`core/rigid_body.hpp`) produce: body `i` occupies `[6i, 6i+6)`, each
body's own block laid out per the single-body `Pack`/`Unpack` convention
(`q` then `v`). This is a direct extension of the single-body stacking
convention from `docs/derivations/integrator_jacobians.md`, not a new
one.

## The system Jacobian is exactly block-diagonal

Since no force couples body `i` to body `j`, `∂Z_{t+1}/∂Z_t` and
`∂Z_{t+1}/∂U` are block-diagonal by construction: block `i` is exactly
the per-body `StepJacobians` from step 2, placed at `(6i, 6i)` in
`dZ_dZ` and `(6i, 3i)` in `dZ_dU`. Off-diagonal blocks are exactly zero —
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
  that introduces a genuine off-diagonal term, and it does so through the
  forces rather than through the integrator: a contact between bodies `i`
  and `j` puts `+Jᵀλ` on one and `−Jᵀλ` on the other, so `∂F/∂Q` acquires
  an `(i, j)` block.

## Where the coupling will enter

`integrate_system_jacobian` takes `std::vector<ForceJacobian>` — one per
body — and stamps the resulting `StepJacobians` onto the diagonal. That
signature *encodes the assumption* that body `i`'s force reads body `i`'s
state alone. It is true for gravity, for the control wrench, and for
contact against static scenery, and it is exactly what body-body contact
falsifies.

The chain rule itself does not change. With `F` the stacked forces and
`M_sys⁻¹ = blockdiag(Mᵢ⁻¹)`, the system update is

```
V_{t+1} = V_t + dt·M_sys⁻¹·F(Q, V, U)
Q_{t+1} = Q_t + dt·V_{t+1}
```

which is the single-body derivation of `integrator_jacobians.md` with
`3B`-dimensional blocks. So what body-body contact needs is a
representation for `∂F/∂Q` and `∂F/∂V` that can express coupling — *not*
a different integrator, and not a different chain rule.

Deliberately not built yet: a dense `3B × 3B` force Jacobian would be the
obvious generalization and the wrong one. Contacts are sparse — only
touching pairs couple — so the right structure follows the contact graph,
which does not exist to look at until detection produces one. Today
`∂F/∂Q` genuinely is block-diagonal, because the bodies genuinely do not
couple, and the per-body vector says so honestly.
