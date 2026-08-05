# CLAUDE.md — GRIP

## What this project is

GRIP is a 2D differentiable rigid-body contact simulator written in C++20.

The technical question behind it: *how trustworthy are gradients through a
contact solver, and how does the choice of contact formulation affect them?*
That question is why the simulator is hand-built rather than pulled off the
shelf — the knobs that matter live inside the contact model, and mainstream
engines don't expose them consistently enough to compare.

That question is a destination, not a requirement for the next few months.
**Build the simulator.** No ontologies, no hypotheses, no publication
strategy in this repo.

## How we work

Claude writes code, including core physics, dynamics, contact, and gradient
implementation. Move fast.

Two things Claude owes on every non-trivial physics change:

- **Explain what it wrote** — the derivation, the convention choices, why
  this formulation and not another. A short paragraph in the response, not a
  lecture. Anything longer than a paragraph goes in `docs/derivations/`.
- **Flag the choices that were judgment calls**, so they don't silently
  harden into assumptions nobody remembers making.

Pedro reviews, redirects, and makes the calls. If a design decision has real
tradeoffs, surface them and ask rather than picking silently.

## Build order

Work in this order. Each step tested and green before the next begins.

1. **Symplectic (semi-implicit) Euler integrator** for a single 2D rigid
   body under gravity. No contact.
2. **Analytic dynamics Jacobians** ∂x_{t+1}/∂x_t and ∂x_{t+1}/∂u,
   validated against central finite differences.
3. **Multiple bodies**, still unconstrained.
4. **Contact detection** — single body against a static half-plane. Signed
   distance, contact normal, contact point.
5. **Penalty contact** (spring-damper). Simple, differentiates trivially,
   and a real baseline worth keeping permanently.
6. **Velocity-level NCP contact solver.** The hard part. Take it after
   penalty contact works end to end.
7. **IFT gradients through the contact solve**, in the style of Dojo.
8. **Formulation swapping** — change the contact model with everything else
   held fixed.

Steps 1–3 are the first milestone. Don't scope past them until they're done.

## Architecture

Settled unless explicitly reopened:

- **C++20.** No exceptions in the hot loop.
- **Eigen** for linear algebra. No other math dependencies.
- **GoogleTest** for tests.
- **double** precision throughout. Never `float`.
- **2D planar.** Configuration is `(x, y, θ)`; body mass matrices are 3×3.
- **State has two representations, deliberately.** `RigidBodyState{q, v}`
  (named 3-vectors) is the physics-facing form — forward simulation and
  force laws read it. A packed 6D `StateVector` (`q` then `v`) is the
  linear-algebra-facing form — Jacobians, FD checks, and anything that
  treats state as a point in `Rⁿ` use it. `Pack`/`Unpack` in
  `core/rigid_body.hpp` are the only conversion boundary and the single
  source of truth for the stacking order. Don't collapse these into one
  representation; each direction (struct-only or vector-only) has a real
  cost — see `docs/derivations/integrator_jacobians.md`.
- **Strictly deterministic.** Same input, same binary, same output, bit for
  bit. No unordered container iteration, no unstable sorts, no
  order-dependent floating-point accumulation, no uninitialized memory.
- **The rollout path and the gradient path share one physics
  implementation.** Never let them diverge.

New dependencies beyond Eigen, GoogleTest, and (later) pybind11 need
discussion first.

## Repository layout

```
grip/
  src/
    core/        types, math helpers, rigid body state
    dynamics/    mass matrices, forces, integrator
    contact/     detection, formulations, solver
    gradient/    adjoint / IFT path
  tests/
    unit/        per-component tests
    validation/  finite-difference and conservation checks
  docs/
    derivations/ the math, in Markdown + LaTeX
  benchmarks/
  CMakeLists.txt
```

## Testing

- Every derivative gets a finite-difference test against its analytic
  counterpart. Written alongside the implementation, not after.
- Energy-behavior tests for the integrator: symplectic Euler should show
  bounded energy oscillation, not secular drift. That test is worth more
  than a single-step tolerance check.
- Near contact activation boundaries, gradients are *expected* to misbehave.
  Tests there record and characterize the behavior rather than enforce a
  tolerance. That boundary is the interesting part, not a bug.
- **Never weaken a tolerance to make a test pass.** Investigate instead.

## Derivations

Non-trivial math — integrator update, mass matrix construction, contact
formulations, the IFT gradient derivation, frame conventions — lives in
`docs/derivations/` as Markdown with LaTeX. Claude writes these as it
implements. Code comments reference the derivation rather than re-deriving
inline.

If code and derivation disagree, flag it. Don't silently pick one.

## Things to do proactively

- Flag hot-loop allocations, hidden Eigen temporaries, aliasing issues
  (`.noalias()` where appropriate).
- Flag determinism risks.
- Suggest missing edge-case tests, especially near contact activation.
- Keep CMake tidy; new source files get wired into the build and a test
  target in the same change.
- Say when something is out of scope for the current build step.

## Things not to do

- Add abstraction layers, plugin systems, or generality no current test
  needs. Elegance that doesn't unlock a measurement is wasted effort.
- Optimize before a benchmark shows it matters.
- Reintroduce research framing into the codebase or its docs.
- Commit commented-out code or TODO placeholders in reviewed paths.
- Build three steps ahead of where the build order is.

## Standing reminder

If a session drifts toward architecture astronomy, research positioning, or
features that aren't the next numbered step, say so and point back to the
build order. Right now that step is small.
