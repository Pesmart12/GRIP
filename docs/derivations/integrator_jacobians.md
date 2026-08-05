# Analytic Jacobians of the symplectic Euler step

## What's being differentiated

State `x = (q, v)` is stacked as a 6-vector, `q` in the first three
components (`RigidBodyState::Pack`/`Unpack` in `core/rigid_body.hpp` are
the single place this ordering is encoded — every Jacobian below inherits
it). We want `∂x_{t+1}/∂x_t` (6×6) and `∂x_{t+1}/∂u` (6×3) for one
`symplectic_euler_step`.

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
v_{t+1} = v_t + h · M⁻¹ · f(q_t, v_t, u)
q_{t+1} = q_t + h · v_{t+1}
```

Differentiating straight through, for a general force law `f(q, v, u)`:

```
∂v_{t+1}/∂q_t = h·M⁻¹·∂f/∂q          ∂v_{t+1}/∂v_t = I + h·M⁻¹·∂f/∂v          ∂v_{t+1}/∂u = h·M⁻¹·∂f/∂u
∂q_{t+1}/∂q_t = I + h·∂v_{t+1}/∂q_t   ∂q_{t+1}/∂v_t = h·∂v_{t+1}/∂v_t          ∂q_{t+1}/∂u = h·∂v_{t+1}/∂u
```

`∂f/∂u = I` always, independent of which force law is active, because
`u` enters every force law additively (`f = f_physics(q,v) + u`) — a
consequence of the control-input design from step 1, not something each
force law needs to supply. `∂f/∂q` and `∂f/∂v` do depend on the force law,
so those two are the only pieces routed through `ForceJacobian` in
`dynamics/forces.hpp`.

For gravity, `∂f/∂q = ∂f/∂v = 0` (it's a constant wrench), which collapses
the above to:

```
∂v_{t+1}/∂q_t = 0    ∂v_{t+1}/∂v_t = I    ∂v_{t+1}/∂u = h·M⁻¹
∂q_{t+1}/∂q_t = I    ∂q_{t+1}/∂v_t = h·I  ∂q_{t+1}/∂u = h²·M⁻¹
```

Notice the whole system is currently linear (gravity plus additive
control), so this Jacobian is exactly constant — it doesn't depend on the
operating point `(q_t, v_t, u)` at all. That stops being true the moment a
`q`- or `v`-dependent force (the step 5 penalty spring-damper) is added;
`ForceJacobian` exists so that addition plugs into the chain-rule assembly
in `symplectic_euler_step_jacobian` without changing its structure.

## Why `ForceJacobian` doesn't carry a `∂f/∂u` field

`u`'s Jacobian is a structural fact of the integrator (`h·M⁻¹`, always),
not a property of the force law, so it's computed directly in
`symplectic_euler_step_jacobian` rather than asked of `gravity_force_jacobian`.
Asking every force law to hand back `∂f/∂u = I` would just be restating
the same constant everywhere for no benefit.

## Finite-difference validation methodology

`tests/validation/test_integrator_jacobians.cpp` checks the analytic
Jacobians above against central finite differences of the actual
`symplectic_euler_step` function — not a symbolic shortcut exploiting the
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
