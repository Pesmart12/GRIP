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

## Where the coupling entered

`integrate_system_jacobian` used to take `std::vector<ForceJacobian>` —
one per body — and stamp the resulting `StepJacobians` onto the diagonal.
That signature *encoded the assumption* that body `i`'s force reads body
`i`'s state alone: true for gravity, for the control wrench, and for
contact against static scenery, and exactly what body-body contact
falsifies. It now takes a `SystemForceJacobian` carrying `∂F/∂Q` and
`∂F/∂V` over the whole system.

**The chain rule did not change.** With `F` the stacked forces and
`M_sys⁻¹ = blockdiag(Mᵢ⁻¹)`,

```
∂V_{t+1}/∂Q = dt·M_sys⁻¹·∂F/∂Q        ∂V_{t+1}/∂V = Id + dt·M_sys⁻¹·∂F/∂V
∂Q_{t+1}/∂Q = Id + dt·∂V_{t+1}/∂Q     ∂Q_{t+1}/∂V = dt·∂V_{t+1}/∂V
```

which is the single-body derivation of `integrator_jacobians.md` with
`3B`-dimensional blocks. So all body-body contact needed was a
representation able to *express* coupling — not a different integrator,
and not a different chain rule. `dZ_dZ` comes out block-diagonal when
`∂F/∂Q` is and dense when it is not; the assembly cannot tell.

One thing that stayed diagonal: `dZ_dF`. Body `i`'s force moves body
`i`'s state and nothing else, whatever produced that force. Coupling
lives entirely in how forces are *computed*, never in how they are
integrated.

The two layouts differ, which is the one wrinkle. `Z` interleaves
`(q, v)` per body while `Q` and `V` stack each separately, so the four
blocks are scattered into `dZ_dZ` rather than copied.

**Still dense, deliberately.** `∂F/∂Q` is `3B × 3B`, and the honest
observation is that `dZ_dZ` was already a dense `6B × 6B`, so the force
Jacobian is strictly smaller than what the step Jacobian costs anyway.
The real structure is sparse and follows the contact graph — now that
detection produces one, the question can finally be asked properly. It is
one question about both matrices, for when a benchmark exists.

## What the tests assert now

`BodiesOutOfReachDoNotCouple` used to be called
`OffDiagonalBlocksAreExactlyZero`, and it changed meaning without
changing its numbers. It used to check that an assembly which never wrote
off-diagonal entries had none — nearly a tautology. The assembly is now
fully general, so the zeros come from `∂F/∂Q` having none, which is a
statement about the physics. Same pattern as `det(dz_dz) = 1` gaining
teeth at step 5a.
