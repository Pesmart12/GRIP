# Symplectic Euler integrator — single 2D rigid body

Symbols follow `docs/derivations/notation.md`.

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
- **The mass matrix is constant and diagonal**, `M = diag(m, m, I)` — no
  `M(q)`, no Coriolis/gyroscopic terms, unlike 3D rigid body dynamics. That
  is two separate facts, derived in the next section.

## The mass matrix

The kinetic energy of a planar rigid body splits into a translational part
and a rotational part:

```
T = ½m‖ċ‖² + ½Iω²
```

with `c` the center of mass and `I` the moment of inertia about the
out-of-plane axis through it. Because the configuration is
`q = (x, y, θ)` with `c = (x, y)`, the generalized velocity is
`v = q̇ = (ẋ, ẏ, ω)` and the expression above is *already* a quadratic
form in `v` with no cross terms:

```
T = ½ vᵀ M v,      M = diag(m, m, I)
```

Two independent facts produce that shape, and they're worth separating
because they fail for different reasons.

### Diagonality comes from measuring `q` at the center of mass

`T` split cleanly above only because the translational term used `ċ` — the
velocity of the *center of mass* specifically. That is König's theorem, and
it is exactly what fails for any other reference point.

Suppose the configuration tracked some other body-fixed point, offset from
the COM by a fixed body-frame vector `ρ`, so the tracked point is
`p = c + R(θ)ρ` and the configuration is `(p, θ)`. Then `c = p − R(θ)ρ`, and
differentiating with `(dR/dθ)ρ = (Rρ)^⊥`:

```
ċ = ṗ − ω(Rρ)^⊥
```

Substituting into `T`, and using that `R` and the perp operator both
preserve norm (`‖(Rρ)^⊥‖ = ‖ρ‖`):

```
T = ½m‖ṗ‖² − mω·ṗ·(Rρ)^⊥ + ½m‖ρ‖²ω² + ½Iω²
  = ½m‖ṗ‖² − mω·ṗ·(Rρ)^⊥ + ½(I + m‖ρ‖²)ω²
```

which as a quadratic form in `v = (ṗ, ω)` is

```
        ⎡  m·Id₂          −m(Rρ)^⊥  ⎤
M(θ) =  ⎢                            ⎥
        ⎣ −m((Rρ)^⊥)ᵀ    I + m‖ρ‖²  ⎦
```

Three things went wrong at once, all of them from `ρ ≠ 0`:

- **The off-diagonal block `−m(Rρ)^⊥` is nonzero**, coupling translation to
  rotation. Pushing the body through the tracked point now also spins it.
- **`M` acquired a `θ` dependence** through `R(θ)`, so it is no longer
  constant — `Ṁ ≠ 0` brings velocity-quadratic (Coriolis-like) terms into
  the equations of motion that `M = const` has none of.
- **The rotational entry picked up `m‖ρ‖²`**, which is the parallel axis
  theorem appearing on its own.

Setting `ρ = 0` kills all three simultaneously. So `M` being diagonal is not
a simplification we chose — it is the payment for having defined `q`'s
translational part *as* the center of mass, which `RigidBodyState` does.
It's also why `BodyShape::vertices` are stored relative to the COM rather
than to some geometric origin: moving that reference would move `ρ` off zero
and reintroduce every term above.

### Constancy comes from 2D

Even at the COM, `M` need not be constant. In 3D the world-frame inertia
tensor is `R·I_body·Rᵀ`, which genuinely varies with orientation, giving
`M(q)` and the gyroscopic terms that make 3D rigid body dynamics awkward.

In 2D there is a single rotation axis, so `I` is a *scalar*, and
`R·I·Rᵀ = I·R·Rᵀ = I` for any rotation. The moment of inertia a planar body
presents is the same at every orientation. `M` is therefore constant in
time as well as diagonal, and `∂M/∂q = 0` drops out of every Jacobian in
`integrator_jacobians.md` before it is ever written down.

### Inversion is therefore free

```
M⁻¹ = diag(1/m, 1/m, 1/I)
```

Three reciprocals, no factorization, no `q` dependence, nothing to
recompute as the body moves. `src/dynamics/mass.hpp` is the single place
this is encoded, returning the three diagonal entries as a 3-vector rather
than a 3×3 — the two call sites need different spellings of the same
object (`.cwiseProduct()` against a vector operand, `.asDiagonal()` against
a matrix one) and a vector converts to either.

This stops being a throwaway convenience at step 6. The central object of a
velocity-level contact solver is the Delassus operator `J M⁻¹ Jᵀ`, the
effective mass seen at a contact point; every formulation compared in
step 8 builds it. That `M⁻¹` is diagonal and constant is what keeps that
assembly cheap and keeps its derivative with respect to `q` identically
zero.

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
control Jacobians are closed-form: `∂v_{t+1}/∂u = dt M⁻¹`,
`∂q_{t+1}/∂u = dt² M⁻¹`. Useful as a direct FD-check target in step 2.

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
v_{t+1} = v_t + dt · M⁻¹ · f(q_t, v_t, u_t)
q_{t+1} = q_t + dt · v_{t+1}
```

`f` is evaluated at the *old* state `(q_t, v_t)` — explicit in the force.
The position update then uses the *new* `v_{t+1}` — implicit in position.
That mixture (explicit force, implicit position) is what "semi-implicit"
means here, and it's the ordering choice — not the force law — that makes
the scheme symplectic. Swapping to `q_{t+1} = q_t + dt·v_t` recovers plain
explicit Euler, which has the same local (per-step) truncation order but
categorically different global behavior (see below).

Note that under constant gravity alone, `v_{t+1}` is exactly the continuous
solution — no discretization error, since acceleration is constant. `q_{t+1}`
is not exact: the closed-form recursion from rest is
`v_n = n·dt·a`, `q_n = dt²·a·n(n+1)/2`, which differs from the continuous
`q(t) = q_0 + v_0 t + ½at²` by a term `~ ½·a·dt·t`, an `O(dt)` offset that
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
x_{n+1} = (1 - ω²dt²) x_n + dt v_n
v_{n+1} = -ω²dt x_n + v_n
```

with `det = 1` exactly (a shear composition, not an approximation), and
trace `2 - ω²dt²`. For `|ω·dt| < 2` the eigenvalues are a complex-conjugate
pair with `|λ| = √det = 1` exactly. The map is an exact rotation in a
skewed coordinate system: the discrete trajectory sits on a fixed
invariant ellipse in phase space forever. Energy computed in the ordinary
`(x, v)` frame therefore oscillates within a bounded band around `E₀` for
all time — it does not grow, to machine precision, independent of how long
the rollout runs.

Plain explicit Euler on the same system is instead

```
x_{n+1} = x_n + dt v_n
v_{n+1} = -ω²dt x_n + v_n
```

with `det = 1 + ω²dt² > 1`. Eigenvalue magnitude is `√(1+ω²dt²) > 1` exactly,
so every step scales the phase-space vector up — guaranteed exponential
energy growth (spiral-out), regardless of how small `dt` is; a smaller
`dt` only slows the growth rate, it doesn't remove it.

`tests/validation/test_energy_behavior.cpp` exercises this directly through
the production `step_body`, using `u = -kq` recomputed from the
current state each step (no new force law added to `src/`), and uses a
locally-implemented explicit Euler as a negative control that must show
growth by contrast. Practical implication carried forward: the `|ω·dt| < 2`
stability bound will matter again once step 5 introduces a stiff contact
spring — a large step size paired with a stiff `k` can push the discrete
map outside this bound and the bounded-oscillation guarantee disappears.
