# GRIP

A 2D differentiable rigid-body contact simulator in C++20.

Bodies fall, bounce, tip, slide, stack and settle — against scenery and
against each other — and every derivative of that motion is analytic,
hand-derived, and validated against finite differences. Not autodiff: the
Jacobians are written down from the maths and then checked against the
forward function they claim to differentiate.

GRIP is a **library, not an application**. It simulates and it
differentiates. Control algorithms — MPC, iLQR, RL training loops,
policies, reward functions, task definitions — live in separate
repositories that link against it.

## Status

Working today:

| | |
|---|---|
| Integrator | symplectic (semi-implicit) Euler, per-step analytic Jacobians |
| Bodies | multiple planar rigid bodies, `q = (x, y, θ)` |
| Contact | convex polygon against a static half-plane, and against other polygons |
| Contact model | penalty — clamped Kelvin–Voigt spring-damper with a Coulomb friction cone |
| Gradients | per-step, plus full-rollout via a reverse-mode adjoint sweep |
| Tests | 113, covering finite-difference validation and structural invariants |

Not there yet: **joints** (every body is free-floating with a wrench at
its centre of mass), a stable public API, and Python bindings. See the
roadmap below.

## Example

```cpp
#include "contact/half_plane.hpp"
#include "contact/penalty.hpp"
#include "core/rigid_body.hpp"
#include "gradient/rollout.hpp"

using namespace grip;

// A unit square, one metre up, above the ground plane y >= 0.
std::vector<RigidBodyState> initial(1);
initial[0].q = Eigen::Vector3d(0.0, 1.2, 0.0);

const std::vector<RigidBodyParams> params(1, RigidBodyParams{/*mass=*/1.0, /*inertia=*/1.0 / 6.0});
const std::vector<BodyShape> shapes(1, BodyShape{{{-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}}});
const HalfPlane ground;
const PenaltyParams penalty{/*stiffness=*/1.0e4, /*damping=*/50.0};

const double dt = 2.0e-4;
const std::vector<std::vector<Eigen::Vector3d>> controls(3000, {Eigen::Vector3d::Zero()});

// Forward: 3001 states — 0.6 s of falling, landing, and bouncing.
const auto trajectory = rollout_system(initial, params, shapes, ground, penalty, controls, dt);

// Backward: gradient of "final height" w.r.t. the initial state and every control.
std::vector<SystemStateVector> dl_dZ(controls.size() + 1, SystemStateVector::Zero(6));
dl_dZ.back() = SystemStateVector::Unit(6, 1);
const std::vector<SystemControlVector> dl_dU(controls.size(), SystemControlVector::Zero(3));

const RolloutGradients gradients = adjoint_system(trajectory, params, shapes, ground, penalty, dl_dZ, dl_dU, dt);
// gradients.dJ_dZ0     6-vector
// gradients.dJ_dU[t]   3-vector per step
```

GRIP never sees your cost function. You supply its partial derivatives —
the seeds — and get total derivatives back. A cost is a task definition,
and tasks belong to whoever is posing them.

## Building

Needs a C++20 compiler and CMake 3.20+. Eigen and GoogleTest are fetched
automatically; there are no other dependencies.

```
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build
```

## What's in the box

```
src/
  core/        rigid body state, packing conventions
  dynamics/    mass matrix, forces, integrator
  contact/     detection (half-plane and polygon-polygon), penalty formulation
  gradient/    rollout and adjoint sweep
tests/
  unit/        per-component
  validation/  finite-difference and conservation checks
docs/
  derivations/ the maths, in Markdown + LaTeX
```

Eight derivations in `docs/derivations/` carry the reasoning behind the
code: `notation.md` (the canonical symbol table), `symplectic_euler.md`,
`integrator_jacobians.md`, `multi_body_system.md`, `contact_detection.md`,
`penalty_contact.md`, `pair_detection.md`, and `adjoint.md`. Code comments
reference these rather than re-deriving inline.

## Design notes

- **`double` throughout, and strictly deterministic.** Same input, same
  binary, same output, bit for bit.
- **The rollout path and the gradient path share one implementation.**
  Force laws compute their own inputs rather than taking precomputed
  ones, and every analytic derivative is finite-difference checked
  against the actual forward function.
- **Integration is separate from force assembly.** The integrator takes a
  force somebody else summed and knows nothing about polygons or
  stiffness. That seam is what lets contact change without the numerical
  core changing with it.
- **The adjoint stores states only.** Jacobians are pure functions of
  state, so the backward sweep rebuilds them rather than taping them — no
  operation graph, and a gradient-free rollout stays cheap.

## Testing

Beyond ordinary unit tests, three kinds of check earn their keep:

**Finite differences.** Every analytic derivative is compared against
central differences of the real forward function, written alongside the
implementation rather than after it.

**Structural invariants.** The step Jacobian satisfies

```
det(dz_dz) = det(Id + dt · M⁻¹ · ∂f/∂v)
```

exactly — `∂f/∂q` cancels out entirely. So a conservative force gives
determinant 1 (phase-space volume preserved, the defining property of a
symplectic map), and a damper contracts it by a computable amount
involving the Delassus operator. That assertion was nearly vacuous when
first written at step 2, gained teeth once contact made `∂f/∂q` nonzero,
and broke exactly as predicted when damping arrived.

**Boundary characterization.** Near contact activation, gradients are
*expected* to misbehave, so those tests record the behaviour instead of
enforcing a tolerance. A central difference straddling the spring's kink
returns a finite wrong number and stays bounded as the step shrinks;
across the damper's jump the same sweep diverges like `1/ε`. Which
severity class a formulation lands in is treated as a measurement.

Body-body contact adds a class the model did not previously contain. The
damper's jump scales with closing speed, so it vanishes for grazing
contact and concentrates predictably where impacts are fast. When the
contact normal flips between faces, the jump scales with penetration
*depth* and vanishes as nothing — the first discontinuity here not
controlled by some velocity going to zero. Two squarely stacked boxes sit
exactly on an instance of it, which the tests pin rather than avoid.

## Roadmap

- [x] Symplectic Euler integrator, analytic Jacobians
- [x] Multiple bodies, block-diagonal system Jacobian
- [x] Contact detection against a static half-plane
- [x] Penalty contact — spring and damper
- [x] Separate integration from force assembly
- [x] Rollout gradients via an adjoint sweep
- [x] Coulomb friction, in the penalty formulation
- [x] Body-body contact, with friction
- [ ] Public API and Python bindings

**Open, and not yet on the list: joints.** Every body here is
free-floating with a wrench at its centre of mass, which covers a lot —
balancing, hopping, pushing, stacking — but not articulated systems.
CartPole, Acrobot, Hopper and Walker all need bilateral constraints, and
faking them with stiff springs would undo the energy and stability
properties the contact work is built on.

Deferred, deliberately: a velocity-level NCP contact solver, implicit-
function-theorem gradients through that solve, and running the same
scenario through both formulations. Those are future ideas, not current
work — the comparison only becomes measurable once something can be
trained on top, and penalty and NCP are different *update rules* rather
than interchangeable force laws, so there was never a plug-in interface
to build.

## Scope

Not built here, on purpose: control algorithms, policies, training loops,
reward functions, task definitions. The rule of thumb is that if a
consumer repository could reasonably implement it, it probably doesn't
belong in GRIP; if it requires knowing how the physics works, it does.
