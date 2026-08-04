# CLAUDE.md — GRIP (Gradient-Reliable Instrument Physics)

## What this project is

GRIP is a differentiable planar (2D) rigid-body contact simulator written in
C++20. It is a **research instrument**, not a product. Its purpose is to
systematically characterize the quality of gradients through contact: when IFT
gradients through a hard contact formulation are trustworthy, where they
degrade, and when first-order optimization beats zeroth-order estimators on
contact-rich tasks. Target venue for resulting work is RA-L.

The broader research program is the "repair map": a 2D ontology
(pathway × representation) charting where mimicry/VLA policies fail under
contact perturbation and which injections of explicit force structure repair
them. GRIP is the apparatus for that program. The gradient-quality study is
one cell of the matrix.

GRIP serves **two consumers of the same physics**, and both are first-class:

1. **First-order:** IFT gradients through the contact solver, for
   gradient-based trajectory optimization and analytic policy gradients.
2. **Zeroth-order:** batched forward rollouts for sampling-based control
   (MPPI/CEM), used as a *matched* baseline — identical integrator, identical
   contact solver, identical friction cone, identical seeds. The scientific
   value of the comparison depends on the two estimators sharing one engine.
   Anything that makes the rollout path diverge physically from the gradient
   path is a correctness bug, not a performance tradeoff.

The working hypothesis the instrument must be able to test: contact makes the
discrete action non-smooth across contact-mode boundaries, so IFT gradients
should be reliable *within* a mode sequence and degrade near mode switches.
Everything in "Instrumentation requirements" below exists to make that
falsifiable.

## Settled architectural decisions — do NOT relitigate

These were decided deliberately. Do not propose alternatives unless Pedro
explicitly reopens a decision.

- **Dimensionality:** planar 2D. Configuration per body is
  `(x, y, θ)`; generalized velocity is `(vx, vy, ω)`; body mass matrix is
  3×3. No quaternions, no 6-DoF spatial algebra. A 3D extension is a
  possible later phase, not current scope.
- **Integration:** symplectic (semi-implicit) Euler. Higher-order,
  implicit, and multi-stage schemes (RK4, midpoint, higher-order
  variational integrators) are out of scope. Note: symplectic Euler *is*
  itself a variational integrator — the discrete Euler–Lagrange equation of
  a particular discrete Lagrangian. This is a scope restriction to
  first-order explicit schemes, not a rejection of discrete variational
  mechanics, which is the correct theoretical frame for this integrator.
- **Contact:** velocity-level NCP (nonlinear complementarity) formulation.
  Not penalty/compliant contact, not position-based dynamics.
- **Gradients:** implicit function theorem through the solver fixed point,
  in the style of Dojo. No autodiff through solver iterations (unrolling).
- **Validation:** finite-difference gradient checks are a first-class part of
  the test suite, not an afterthought. Every analytic Jacobian ships with an
  FD test.
- **Language/stack:** C++20, Eigen for linear algebra, GoogleTest for tests,
  CMake build. Python bindings via pybind11 (later phase).
- **Precision:** `double` everywhere. No `float`, no mixed precision.
- **Determinism:** bitwise-reproducible runs given the same inputs and build.
  See "Determinism under parallelism" for how this interacts with batched
  rollouts.
- **No exceptions in the hot loop.** Error handling in sim-step code uses
  status returns / assertions. Exceptions are acceptable in setup, I/O, and
  binding code.

## Instrumentation requirements

GRIP is an instrument. An instrument that computes the right answer but
cannot report *why* is useless for the study. These outputs are part of the
public API contract, not debug extras, and they are not optional:

- **Contact impulses are first-class.** Every step returns the per-contact
  normal and friction impulses, in documented frames and units, alongside
  the resulting state. The force-modality thesis requires that a policy or
  an analysis script can consume forces directly; if impulses live only
  inside the solver and only post-contact velocity escapes, the instrument
  cannot serve the thesis it was built for.

- **Contact mode signature is a first-class output.** Every step returns a
  mode signature: the set of active contacts and, per active contact, its
  friction status (sticking / sliding, and slide direction sign). This must
  be cheap to compare between steps so a driver can detect mode switches
  along a trajectory.

- **Boundary proximity margins.** Alongside the discrete signature, expose
  the continuous margins that determine it — normal gap / normal impulse
  complementarity slack, and distance to the friction cone boundary. A
  discrete switch indicator alone only supports "near a switch, yes/no";
  the margins support regressing gradient error against *distance to
  boundary*, which is the actual claim under test.

- **Solver diagnostics.** Iteration count, final residual, and a
  convergence flag per step. Gradient quality is conditional on the forward
  solve having actually converged; an unconverged solve that silently
  returns a plausible state will contaminate every downstream conclusion.

Rationale, so this doesn't get optimized away later: the research object is
the relationship between gradient error and contact-mode structure. Without
mode signatures and margins as data, gradient error can only be plotted
against time. With them, it can be plotted against the quantity the
hypothesis is about.

## Rollout and cost layer

Required for the zeroth-order arm. This is Claude-writable scaffolding — it
is infrastructure, not core physics.

- **Batched rollout API:** simulate K independent trajectories of horizon H
  from given initial states and control sequences. Preallocated per-rollout
  workspaces; no allocation inside the rollout loop after warm-up.
- **Cost functor interface:** running cost `l(state, contact_data, action)`
  plus terminal cost. The running cost must have access to contact impulses
  and mode signature, not just generalized state — contact-aware costs are
  a research variable, not a special case.
- **RNG:** counter-based (Philox/Threefry-style), keyed by
  `(seed, rollout_index, step_index)`. Sampling must be reproducible
  independently of thread count and scheduling order. Do not use
  `std::mt19937` shared across threads, and do not seed per-thread from a
  clock.
- **Benchmarks:** `bench/` gets a rollouts-per-second target for
  representative K and H. Throughput is the binding constraint on the
  zeroth-order arm, so it is measured continuously, not once.

MPPI itself (sampling, exponential weighting, weighted average, receding
shift) lives in `src/control/` and is Claude-writable. It is control code,
not physics.

## Determinism under parallelism

Batched rollouts want threads; threaded reductions break bitwise
reproducibility. The resolution, decided:

- **Per-rollout determinism is absolute.** Rollout *i* produces bitwise
  identical results regardless of K, thread count, or scheduling.
- **Reductions are deterministically ordered.** The MPPI weighted average
  accumulates in fixed rollout-index order, not completion order. Pay the
  synchronization cost.
- No atomics accumulating doubles. No `std::reduce` with unspecified
  ordering on floating-point. No thread-local accumulators merged in
  arbitrary order.
- A determinism test must run the same batch at multiple thread counts and
  bitwise-compare the result.

If this ever becomes the throughput bottleneck, raise it explicitly rather
than relaxing it — the matched-comparison argument is the entire reason the
zeroth-order arm lives in GRIP instead of in an off-the-shelf engine.

## Division of labor — respect this strictly

**Pedro writes:** core physics and solver code — the integrator, dynamics,
contact model, NCP solver, and IFT gradient derivations. This is deliberate:
the point is for Pedro to understand the physics at implementation depth.

**Claude handles:** rigorous code review, build system (CMake), test
scaffolding and GoogleTest infrastructure, CI configuration, pybind11
bindings, benchmarking harnesses, the batched rollout driver, MPPI and other
control-side code, experiment drivers, plotting, documentation, and
refactoring suggestions.

When Pedro is working on core physics:
- Review for correctness, numerical robustness, and style. Point out bugs
  and derivation errors directly and precisely.
- Do NOT write the physics implementation for him, even if asked casually
  ("just show me the solver loop"). Instead: explain the math, sketch
  pseudocode at most, write the *test* that his implementation must pass,
  or point to the relevant equations. If he explicitly overrides this rule
  in a session, confirm once, then comply.
- Writing FD validation tests against his analytic gradients is always
  in-scope for Claude — that's scaffolding, and it keeps him honest.

The boundary: anything that computes physics or its derivatives is Pedro's.
Anything that *drives*, *measures*, *batches*, or *plots* the physics is
Claude's. MPPI is a consumer of the physics, so it is Claude's; the contact
solver it consumes is not.

## Code standards

- C++20. Prefer `constexpr`, `[[nodiscard]]`, concepts where they clarify.
- Eigen types in APIs: fixed-size where dimension is known
  (`Eigen::Vector3d` for planar body twist/config, `Eigen::Matrix3d` for
  body mass matrices), `Eigen::VectorXd` / `MatrixXd` for solver-level
  quantities. Pass by `const&`.
- No raw `new`/`delete`. No heap allocation inside the sim step or the
  rollout loop after warm-up: preallocate workspaces. Flag any allocation
  in the hot loop during review.
- Naming: `snake_case` functions/variables, `PascalCase` types,
  `kCamelCase` constants, `trailing_underscore_` for private members.
- Every public function gets a doc comment stating units, frames, and
  conventions (e.g., "world frame, N·s"). Frame bugs are the #1 expected
  bug class — be paranoid about them in review.
- Warnings are errors: `-Wall -Wextra -Wpedantic -Werror`.
- clang-format enforced; config lives at repo root.

## Numerical conventions

- Generalized coordinates: planar, `(x, y, θ)` per body, θ in radians,
  wrapped consistently and documented. Generalized velocity in world frame
  unless stated otherwise; state the convention in every public doc comment.
- Time step `h` is a parameter, never hard-coded. Tests should run at
  multiple step sizes.
- Impulses, not forces, are the natural solver output at velocity level.
  Document units explicitly (N·s) and provide the `h`-division to force
  units at the API boundary rather than leaving callers to guess.
- FD gradient checks: central differences, sweep epsilon over
  {1e-4 … 1e-8}, assert best-epsilon relative error < 1e-6 for smooth
  paths; contact-adjacent tolerances documented per-test with rationale.
- Document conditioning assumptions wherever a linear solve occurs
  (what factorization, what happens near singular configurations).

## Repository layout

```
grip/
  include/grip/     public headers
  src/              implementation
    core/           types, state, math utilities, RNG
    dynamics/       rigid body dynamics, integrator
    collision/      contact detection, geometry
    solver/         NCP contact solver
    diff/           IFT gradient machinery
    sim/            batched rollout driver, cost interface
    control/        MPPI, CEM, gradient-based trajectory optimization
  tests/            GoogleTest suites (mirror src/ structure)
  bench/            benchmarks (rollout throughput, step cost)
  bindings/         pybind11 (later phase)
  scripts/          plotting, experiment drivers (Python)
  docs/             derivations, decision records
```

## Build and test

```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
ctest --test-dir build --output-on-failure
```

- Debug builds enable Eigen bounds checking and assertions.
- A test that passes only in Release is a bug.
- Run the full test suite before declaring any task complete. Never mark
  work done with failing or skipped tests.

## Testing policy

- Analytic Jacobian ⇒ FD test. No exceptions.
- Energy behavior tests for the integrator (symplectic Euler: bounded
  energy error on conservative systems, verify expected drift character).
- Determinism test: two runs, identical inputs, bitwise-compare
  trajectories. Batched determinism test: same batch, varying thread
  counts, bitwise-compare.
- Contact unit tests build up from the simplest cases: point mass on plane,
  disk drop, sliding block with friction cone, then stacks.
- **Mode signature tests:** constructed scenarios with known mode sequences
  (block transitioning stick→slip under increasing tangential load;
  contact making and breaking) assert the reported signature matches the
  analytically expected one. If the signature is wrong, every conclusion
  drawn from it is wrong.
- **Gradient-vs-boundary tests:** at least one scenario where FD and IFT
  gradients are asserted to agree away from a mode boundary and are
  *measured* (not asserted to agree) near one. Near-boundary disagreement
  is a finding, not a failure — the test records it rather than enforcing
  a tolerance.
- Regression tests get a comment linking the bug they pin down.

## Derivations

Non-trivial math (NCP formulation, IFT gradient derivation, frame
conventions, mode signature definition) lives in `docs/derivations/` as
Markdown with LaTeX. Code comments reference the derivation doc rather than
re-deriving inline. When reviewing physics code, check it against the
derivation doc; if they disagree, flag it — do not silently pick one.

## Things Claude should proactively do

- Flag hot-loop allocations, hidden copies of Eigen temporaries, and
  aliasing issues (`.noalias()` where appropriate).
- Flag any nondeterminism risk (unordered containers, uninitialized
  memory, unstable sorts, order-dependent floating-point accumulation).
- Flag any change that makes the gradient path and the rollout path
  traverse different physics code. They must share one implementation.
- Keep CMake tidy; new source files get wired into the build and a test
  target in the same change.
- Suggest missing edge-case tests, especially near contact activation
  boundaries where gradients are expected to misbehave — that boundary
  *is the research object*.

## Things Claude should never do

- Write core solver/dynamics/gradient implementation code (see division
  of labor).
- Introduce dependencies beyond Eigen, GoogleTest, pybind11 without
  discussion.
- Weaken a tolerance to make a test pass. Investigate instead.
- Use `float`, exceptions in the hot loop, or nondeterministic constructs.
- Silently drop mode-signature, impulse, margin, or solver-diagnostic
  outputs for performance. Raise the tradeoff explicitly.
- Commit commented-out code or TODO-laden placeholders in reviewed paths.
