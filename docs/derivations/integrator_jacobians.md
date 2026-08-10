# Analytic Jacobians of the symplectic Euler step

Symbols follow `docs/derivations/notation.md`.

## What's being differentiated

State `z = (q, v)` is stacked as a 6-vector, `q` in the first three
components (`RigidBodyState::Pack`/`Unpack` in `core/rigid_body.hpp` are
the single place this ordering is encoded — every Jacobian below inherits
it). We want `∂z_{t+1}/∂z_t` (6×6) and `∂z_{t+1}/∂u` (6×3) for one
`step_body`.

This is why the packed `StateVector` exists alongside `RigidBodyState`
rather than replacing it. Forward simulation and force laws read `q` and
`v` as named 3-vectors — that stays readable, and stays correct once
multiple bodies arrive (no manual index bookkeeping). But a Jacobian is a
matrix acting on a point in `Rⁿ`; without a flat vector type, every
consumer (finite-difference checks here, later multi-step Jacobian
chaining, eventually the IFT gradient path) would have to invent its own
flattening. `Pack`/`Unpack` are the one, tested boundary between the two
representations — not duplication, but two views for two different kinds
of code.

## Chain rule through the update

Recall the step:

```
v_{t+1} = v_t + dt · M⁻¹ · f(q_t, v_t, u)
q_{t+1} = q_t + dt · v_{t+1}
```

Differentiating straight through, for a general force law `f(q, v, u)`:

```
∂v_{t+1}/∂q_t = dt·M⁻¹·∂f/∂q            ∂v_{t+1}/∂v_t = Id + dt·M⁻¹·∂f/∂v    ∂v_{t+1}/∂u = dt·M⁻¹·∂f/∂u
∂q_{t+1}/∂q_t = Id + dt·∂v_{t+1}/∂q_t   ∂q_{t+1}/∂v_t = dt·∂v_{t+1}/∂v_t     ∂q_{t+1}/∂u = dt·∂v_{t+1}/∂u
```

`∂f/∂u = Id` always, independent of which force law is active, because
`u` enters every force law additively (`f = f_physics(q,v) + u`) — a
consequence of the control-input design from step 1, not something each
force law needs to supply. `∂f/∂q` and `∂f/∂v` do depend on the force law,
so those two are the only pieces routed through `ForceJacobian` in
`dynamics/forces.hpp`.

For gravity, `∂f/∂q = ∂f/∂v = 0` (it's a constant wrench), which collapses
the above to:

```
∂v_{t+1}/∂q_t = 0     ∂v_{t+1}/∂v_t = Id     ∂v_{t+1}/∂u = dt·M⁻¹
∂q_{t+1}/∂q_t = Id    ∂q_{t+1}/∂v_t = dt·Id  ∂q_{t+1}/∂u = dt²·M⁻¹
```

Notice the whole system is currently linear (gravity plus additive
control), so this Jacobian is exactly constant — it doesn't depend on the
operating point `(q_t, v_t, u)` at all. That stops being true the moment a
`q`- or `v`-dependent force (the step 5 penalty spring-damper) is added;
`ForceJacobian` exists so that addition plugs into the chain-rule assembly
in `step_body_jacobian` without changing its structure.

## Integration and assembly are separate

`integrate_body` / `integrate_body_jacobian` are the integrator proper.
They take a force (or a `ForceJacobian`) that somebody else summed, and
know nothing about contact — no polygons, no half-planes, no stiffness.
`step_body` / `step_system` sit on top and do the summing.

Two things forced that split, and neither is aesthetic.

**Body-body contact does not decompose per body.** A contact between two
bodies puts `+Jᵀλ` on one and `−Jᵀλ` on the other. There is no such thing
as "body `i`'s contact force" computed from body `i` alone, so any
function that steps one body while computing its own contact force stops
being expressible. Contact has to be assembled over the system first.
Against static scenery this doesn't bite — the plane has no state — which
is exactly why the earlier shape worked and why it stops working later.

**The Jacobian never depended on the state.** `integrate_body_jacobian`
takes `(params, force_jacobian, dt)` and no state at all. Everything below
uses only the mass matrix, the timestep, and the force law's derivatives.
That was already true when the signature claimed otherwise; the split
makes the type say it.

The assembly layer is deliberately thin, stateless, and not an API
promise. It exists so the common path is a single call that cannot
evaluate the force and its Jacobian at different states, which is the one
hazard the split introduces — a caller holding a force fixed across
several steps would get a silently wrong trajectory, and the types cannot
stop them.

## Why `ForceJacobian` doesn't carry a `∂f/∂u` field, and why the integrator never sees `u`

`u` enters every force law additively, so `∂f/∂u = Id` unconditionally.
That makes `∂z_{t+1}/∂u` and `∂z_{t+1}/∂f` **the same matrix**, not merely
equal in value — which is why `integrate_body_jacobian` can omit `u`
entirely and return a single `dz_df`. Asking every force law to hand back
`∂f/∂u = Id` would restate the same constant everywhere for no benefit;
asking the integrator to accept a `u` it structurally cannot use was the
same mistake one level up.

Consumers wanting a control Jacobian read `dz_df` directly. The
finite-difference test perturbs `u` and compares against `dz_df`, so the
identity is checked rather than assumed.

## Finite-difference validation methodology

`tests/validation/test_integrator_jacobians.cpp` checks the analytic
Jacobians above against central finite differences of the actual
`step_body` function — not a symbolic shortcut exploiting the
fact that today's system happens to be linear. Central differences,
`(f(x+ε) - f(x-ε)) / 2ε`, have `O(ε²)` truncation error but `O(ε_mach/ε)`
floating-point cancellation error as `ε → 0`; balancing the two for
double precision gives `ε ~ ε_mach^(1/3) ≈ 6×10⁻⁶`, which is why
`CentralDifferenceJacobian` (`tests/utils/finite_difference.hpp`)
defaults to `1e-6`. That helper is generic (`R^InDim → R^OutDim`, any
fixed dimensions) since CLAUDE.md establishes FD-vs-analytic checks as a
permanent, recurring part of the test suite — every future analytic
Jacobian (contact, IFT) reuses the same helper rather than re-deriving the
central-difference loop.

The test also asserts `∂v_{t+1}/∂q_t` is *exactly* zero for gravity, not
just numerically close — that block should come out as an identical zero
by construction, and an exact check catches a bug (e.g. an accidental
`∂f/∂q` contribution) that a loose FD tolerance might paper over.

## Structural check: the determinant is exactly 1

For any conservative force law (`∂f/∂v = 0`, writing `A = ∂f/∂q`), the
step Jacobian is

```
dz_dz = [ Id + dt²·M⁻¹A   dt·Id ]
        [ dt·M⁻¹A         Id    ]
```

and the block formula `det = det(S)·det(P − Q·S⁻¹·R)` with `S = Id`
gives

```
det(P − Q·S⁻¹·R) = det(Id + dt²·M⁻¹A − dt·Id·dt·M⁻¹A) = det(Id) = 1
```

The two `dt²·M⁻¹A` terms cancel exactly. So `det(dz_dz) = 1` — the
discrete statement of phase-space volume preservation, which is the
defining property of a symplectic map and the same fact underlying the
bounded energy behavior in `symplectic_euler.md`.

This is a different kind of test from the finite-difference comparison.
FD asks whether the analytic Jacobian matches the implementation; the
determinant asks whether the Jacobian has the structure the integrator's
design promises. It holds for *any* conservative force, so it keeps its
teeth as force laws are added. The system-level version follows
immediately: `dZ_dZ` is block-diagonal, so its determinant is the product
of per-body determinants, hence 1 for any body count.

Redoing the same block algebra *without* assuming `∂f/∂v = 0` shows this
is a special case of a sharper identity — the `∂f/∂q` terms cancel
whatever `∂f/∂v` is, leaving

```
det(dz_dz) = det(Id + dt·M⁻¹·∂f/∂v)
```

so `∂f/∂q` never affects the determinant at all. Step 5a's penalty spring
is conservative and keeps `det = 1`, but now with a nonzero,
configuration-dependent `∂f/∂q`, which is what makes the cancellation
non-vacuous for the first time. Step 5b's damper breaks it by an exactly
computable amount — `det(dz_dz) = det(Id − dt·b·Delassus)` over the
contacts carrying force, strictly less than 1, since dissipation
contracts phase-space volume. Both derived in `penalty_contact.md`.

Once `integrate_body_jacobian` takes a `ForceJacobian` rather than a
state, this stops being a claim checked at whatever operating points a
force law happens to visit and becomes one checked directly. The tests
feed arbitrary matrices: `∂f/∂q` symmetric, `∂f/∂q` deliberately
asymmetric (the cancellation never needed it to be a Hessian), and
`∂f/∂v` both dissipative and not. That is the algebra asserted as
algebra, including the part saying `∂f/∂q` has no say at all — which no
physical force law can exercise on its own, because every one of them
couples the two blocks in some particular way.
