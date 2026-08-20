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
| Python | batched over independent environments, numpy in and out |
| Speed | 0.3 µs/step free, 3.3 µs in contact — faster than real time |
| Tests | 132, covering finite-difference validation and structural invariants |

Not there yet: **joints** — every body is free-floating with a wrench at
its centre of mass. See the roadmap below.

## Installing

```
pip install .          # or -e . to develop against it
```

Needs a C++20 compiler; CMake and Ninja are fetched by the build if they
are not already installed. Nothing else — tests, demos and benchmarks are
all switched off for a wheel build, so `pip install` does not clone a test
framework or a window toolkit.

An editable install does **not** rebuild on C++ changes. That would need
the compiler on `PATH` at import time, and a missing toolchain would then
turn an ordinary `import grip` into a confusing build failure. Reinstall
after touching `src/`.

## Example

Rollout and gradients, batched over independent environments.
Arrays are `(environments, bodies, 6)` for state and
`(steps, environments, bodies, 3)` for controls — the last state axis is
`(x, y, θ, vx, vy, ω)`, which is the C++ packing convention viewed as an
array rather than a translation of it.

```python
import numpy as np
import grip

scene = grip.Scene(
    params=[grip.RigidBodyParams(mass=1.0, inertia=1.0 / 6.0)],
    shapes=[grip.BodyShape([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])],
    plane=grip.HalfPlane(normal=[0.0, 1.0], offset=0.0),
    penalty=grip.PenaltyParams(stiffness=1e4, damping=50.0),
    dt=2e-4,
)

envs, steps = 64, 300
initial = np.zeros((envs, 1, 6))
initial[:, 0, 1] = np.linspace(1.0, 2.0, envs)      # a different drop height each
controls = np.zeros((steps, envs, 1, 3))

# 300 control steps of 10 integration steps each: one boundary crossing, not 3000.
trajectory = grip.rollout_batch([scene] * envs, initial, controls, substeps=10)

# Gradient of final height w.r.t. the initial state and every control.
dl_dZ = np.zeros_like(trajectory)
dl_dZ[-1, :, 0, 1] = 1.0
dJ_dZ0, dJ_dU = grip.adjoint_batch([scene] * envs, trajectory, controls, 10, dl_dZ, np.zeros_like(controls))
```

Every environment carries its own `Scene`, so masses, shapes, ground
angle, contact parameters and gravity can all be randomized across a
batch. Only the body count has to match.

GRIP never sees your cost function. You supply its partial derivatives —
the seeds — and get total derivatives back. A cost is a task definition,
and tasks belong to whoever is posing them.

`substeps` is the other half of the design. A policy runs at a fixed
control rate while penalty contact needs a much finer integration step,
so one call advances many — a control step of 10 substeps crosses the
language boundary once instead of ten times. See
[Performance](#performance) for why that matters.

## From C++

The same functions, without the array conversion:

```cpp
#include "api/scene.hpp"
#include "api/simulate.hpp"

using namespace grip;

Scene scene;
scene.params.assign(1, RigidBodyParams{/*mass=*/1.0, /*inertia=*/1.0 / 6.0});
scene.shapes.assign(1, BodyShape{{{-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}}});
scene.penalty = PenaltyParams{/*stiffness=*/1.0e4, /*damping=*/50.0};
scene.dt = 2.0e-4;

const std::vector<Scene> scenes(64, scene);
StateBatch initial = make_state_batch(scenes.size(), 1);
const ControlBatch controls = make_control_batch(300, scenes.size(), 1);

TrajectoryBatch trajectory;
rollout_batch(scenes, initial, controls, /*substeps=*/10, trajectory);

TrajectoryBatch dl_dZ = make_trajectory_batch(controls.steps, scenes.size(), 1);
dl_dZ.values[trajectory_batch_offset(dl_dZ, controls.steps, 0, 0) + 1] = 1.0;
const RolloutGradientBatch gradients = adjoint_batch(scenes, trajectory, controls, 10, dl_dZ, make_control_batch(controls.steps, scenes.size(), 1));
```

Underneath these sit the per-scene functions the physics is actually
written in — `step_system`, `rollout_system`, `adjoint_system` — which
take parallel vectors rather than a `Scene` and know nothing about
batching. They remain the single implementation both paths share.

## Performance

Measured, not claimed. Single-threaded, one scene at a time, on an
i7-14700F at `dt = 5e-4`:

| scene | forward | backward | wall-clock per simulated second |
|---|---|---|---|
| free flight, 1 body | 0.31 µs/step | 0.80 µs | 0.003 s |
| resting pair, 2 bodies | 3.3 µs/step | 6.2 µs | 0.01 s |
| stack, 10 bodies | 72 µs/step | 128 µs | 0.14 s |

Three things fall out. Contact costs an order of magnitude over free
flight — that gap is the pair path, SAT and clipping and the
second-order jets. The backward sweep runs about 2× the forward step
across every scene size, so recomputing Jacobians rather than taping them
is not the expensive choice it looks like. And everything is faster than
real time, at a timestep chosen for contact resolution rather than
stability: **penalty's cost is the timestep, not the per-step work**,
which is precisely what 2.0 buys back.

Full method and caveats in [`benchmarks/README.md`](benchmarks/README.md).

## Building

To work on GRIP itself rather than just use it. Needs a C++20 compiler
and CMake 3.20+; Eigen and pybind11 are fetched automatically, and
GoogleTest is fetched only when the tests are switched on.

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

| option | default | |
|---|---|---|
| `GRIP_BUILD_TESTS` | ON | the C++ suite; fetches GoogleTest |
| `GRIP_BUILD_BENCHMARKS` | ON | the baseline throughput benchmark |
| `GRIP_BUILD_BINDINGS` | ON | the Python module; fetches pybind11 |
| `GRIP_BUILD_DEMOS` | ON | the raylib demo; fetches raylib |

All four go OFF for a wheel build. Add `-DPython_EXECUTABLE=<path>` if
Python is not on `PATH` — `PYBIND11_FINDPYTHON` is on, so that variable
is honoured rather than quietly ignored on a machine with more than one
interpreter.

**Build Release for anything you intend to time.** Unoptimized Eigen is
slower by more than an order of magnitude; the benchmark refuses to be
taken seriously if `NDEBUG` is absent and says so in the output.

## What's in the box

```
src/
  core/        rigid body state, packing conventions
  dynamics/    mass matrix, forces, integrator
  contact/     detection (half-plane and polygon-polygon), penalty formulation
  gradient/    rollout and adjoint sweep
  api/         scene description, batched entry points
tests/
  unit/        per-component
  validation/  finite-difference and conservation checks
bindings/      pybind11 module
benchmarks/    baseline throughput
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
- **Environments in a batch never interact.** The public API is a loop
  over independent scenes, serial today. That is why a batch is
  bit-for-bit identical to running each scene alone, why determinism
  survives whatever order they are stepped in, and why a parallel backend
  can replace the loop body without disturbing anything above it.
- **`Scene` is configuration, not a simulator.** State is passed in and
  returned out, never held. Joints arrive in 2.0 as one more field on it,
  which is a recompile rather than a change of call signature.

## Testing

132 tests, run by `ctest` — a C++ suite plus a Python one exercising the
bindings against closed forms rather than against GRIP itself. Beyond
ordinary unit tests, four kinds of check earn their keep:

**Bit-for-bit equivalence.** A batch of identical environments must
produce exactly what running each alone produces — not to a tolerance.
Environments never interact, so that one assertion pins correctness and
determinism together, and it is what will catch a parallel backend in 3.0
if it ever reorders something it should not.


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

Built:

- [x] Symplectic Euler integrator, analytic Jacobians
- [x] Multiple bodies, block-diagonal system Jacobian
- [x] Contact detection against a static half-plane
- [x] Penalty contact — spring and damper
- [x] Separate integration from force assembly
- [x] Rollout gradients via an adjoint sweep
- [x] Coulomb friction, in the penalty formulation
- [x] Body-body contact, with friction

Planned, in three releases:

- [x] **1.0** — public API and Python bindings, batched over scenes. No
  new physics; this release is about making what already exists callable.
  Done: batched API, pybind11 bindings, and a pip-installable package.
- [ ] **2.0** — a velocity-level NCP contact solve with
  implicit-function-theorem gradients, replacing penalty as the default,
  plus joints as bilateral constraints in that same solve.
- [ ] **3.0** — parallelism across scenes.

**Why NCP for 2.0.** Penalty ties together two knobs that ought to be
independent: the stiffness that makes contact behave well is the same
stiffness that makes the gradient discontinuous, and no amount of tuning
escapes it. A solve with a central-path relaxation separates them —
non-penetration is exact whatever the relaxation is, and the relaxation
then controls only gradient smoothness. It also runs at `dt = 1e-2`
rather than `5e-4`, with stable stacking.

**Why joints wait for it.** Every body here is free-floating with a
wrench at its centre of mass, which covers a lot — balancing, pushing,
stacking, sliding — but not articulated systems. CartPole, Acrobot and
Hopper need bilateral constraints, and a bilateral constraint lands in
the same linear algebra as a contact constraint. Building the solve first
means joints come out exact, rather than held together by stiff springs
that would undo the energy and stability properties the contact work
rests on.

Penalty contact does not go away when NCP arrives; it stays as the
validation baseline. What is *not* planned is an abstraction over the
two. Penalty is a force and an NCP solve is a post-force projection —
different *update rules* rather than interchangeable force laws — so
there was never a plug-in interface to build.

## Scope

Not built here, on purpose: control algorithms, policies, training loops,
reward functions, task definitions. The rule of thumb is that if a
consumer repository could reasonably implement it, it probably doesn't
belong in GRIP; if it requires knowing how the physics works, it does.
