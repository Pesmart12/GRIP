# Symplectic Euler integrator — single 2D rigid body

## State and convention

Configuration `q = (x, y, θ)`, generalized velocity `v = (vx, vy, ω)`. Both
live in `Eigen::Vector3d`. `RigidBodyState` holds `(q, v)`; `RigidBodyParams`
holds `(mass, inertia)` separately — Jacobians (step 2) are taken with
respect to state and control, never with respect to body parameters.

- **y-up.** Gravity acts as `(0, -g, 0)`. This fixes the ground-plane
  convention for contact detection later: a half-plane at some `y = const`
  with outward normal `+y`.
- **θ is unbounded.** No wrap to `[-π, π)`. Wrapping is a discontinuous map
  and would break the analytic Jacobians in step 2 for no benefit; if a
  display layer ever wants a wrapped angle, that's cosmetic and happens
  outside the state.
- **2D rotation is abelian**, so `I` (scalar moment of inertia about the
  out-of-plane axis) doesn't change as the body rotates. `M = diag(m, m, I)`
  is constant — no `M(q)`, no Coriolis/gyroscopic terms, unlike 3D rigid
  body dynamics.

## Control input

`u = (fx, fy, τ)` is a generalized wrench applied at the center of mass —
the same space as `v`, so it plugs directly into `M⁻¹u` with no moment-arm
computation inside the integrator. Actuator geometry (a force applied off
the COM) is a modeling concern for a layer above the integrator, which
converts `(offset, force) → (F, τ = r×F)` before it ever becomes `u`.

`u` is reserved for external/control input specifically. Contact impulses
(step 6 onward) are computed separately and summed into the same velocity
update — same mathematical slot, different semantic bucket, not routed
through `u`.

Because `M` is constant and the force law here is additive in `u`, the
control Jacobians are closed-form: `∂v_{t+1}/∂u = h M⁻¹`,
`∂q_{t+1}/∂u = h² M⁻¹`. Useful as a direct FD-check target in step 2.

## Force law interface

`f(q, v, u)` is kept general from the start, even though gravity uses none
of `q` or `v`:

```
f(q, v, u) = gravity_force(params, g) + u = (0, -mg, 0) + u
```

Later force laws are `q`-dependent (penalty spring, step 5) or
`v`-dependent (damping, contact friction) — keeping the signature stable
now avoids changing the integrator's call shape later. Gravity is simply
the case where `∂f/∂q = ∂f/∂v = 0`, which is a good first FD test precisely
because those blocks should come out identically zero.

## Update rule

```
v_{t+1} = v_t + h · M⁻¹ · f(q_t, v_t, u_t)
q_{t+1} = q_t + h · v_{t+1}
```

`f` is evaluated at the *old* state `(q_t, v_t)` — explicit in the force.
The position update then uses the *new* `v_{t+1}` — implicit in position.
That mixture (explicit force, implicit position) is what "semi-implicit"
means here, and it's the ordering choice — not the force law — that makes
the scheme symplectic. Swapping to `q_{t+1} = q_t + h·v_t` recovers plain
explicit Euler, which has the same local (per-step) truncation order but
categorically different global behavior (see below).

Note that under constant gravity alone, `v_{t+1}` is exactly the continuous
solution — no discretization error, since acceleration is constant. `q_{t+1}`
is not exact: the closed-form recursion from rest is
`v_n = n·h·a`, `q_n = h²·a·n(n+1)/2`, which differs from the continuous
`q(t) = q_0 + v_0 t + ½at²` by a term `~ ½ a h t`, an `O(h)` offset that
grows linearly with time but shrinks with step size — ordinary first-order
discretization error, not an oscillation or drift phenomenon, because
constant gravity has no potential well and nothing periodic to compare
against.

## Energy behavior: bounded oscillation vs. secular drift

This is the property the integrator is actually chosen for, and it only
shows up in a system with a restoring force — gravity alone doesn't
exercise it. Consider the harmonic oscillator, `f(q) = -kq` (1D, mass `m`,
`ω = √(k/m)`). Symplectic Euler's step is the linear map

```
x_{n+1} = (1 - ω²h²) x_n + h v_n
v_{n+1} = -ω²h x_n + v_n
```

with `det = 1` exactly (a shear composition, not an approximation), and
trace `2 - ω²h²`. For `|ωh| < 2` the eigenvalues are a complex-conjugate
pair with `|λ| = √det = 1` exactly. The map is an exact rotation in a
skewed coordinate system: the discrete trajectory sits on a fixed
invariant ellipse in phase space forever. Energy computed in the ordinary
`(x, v)` frame therefore oscillates within a bounded band around `E₀` for
all time — it does not grow, to machine precision, independent of how long
the rollout runs.

Plain explicit Euler on the same system is instead

```
x_{n+1} = x_n + h v_n
v_{n+1} = -ω²h x_n + v_n
```

with `det = 1 + ω²h² > 1`. Eigenvalue magnitude is `√(1+ω²h²) > 1` exactly,
so every step scales the phase-space vector up — guaranteed exponential
energy growth (spiral-out), regardless of how small `h` is; smaller `h`
only slows the growth rate, it doesn't remove it.

`tests/validation/test_energy_behavior.cpp` exercises this directly through
the production `step_body`, using `u = -kq` recomputed from the
current state each step (no new force law added to `src/`), and uses a
locally-implemented explicit Euler as a negative control that must show
growth by contrast. Practical implication carried forward: the `|ωh| < 2`
stability bound will matter again once step 5 introduces a stiff contact
spring — a large step size paired with a stiff `k` can push the discrete
map outside this bound and the bounded-oscillation guarantee disappears.
