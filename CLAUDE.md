# CLAUDE.md — GRIP

## What this project is

GRIP is a 2D differentiable rigid-body contact simulator written in C++20.

It is a **library, not an application**. Control algorithms — MPC, iLQR,
RL training loops, policies, reward functions, task definitions — live in
*separate repositories* that link against GRIP and call it. Nothing in
that list gets built here. What gets built here is the physics, its
derivatives, and the API those are reached through.

That split decides a lot of borderline questions. If a consumer repo
could reasonably implement it, it probably doesn't belong here. If it
requires knowing how the physics works, it does.

There is a longer-term technical question behind the project: *how
trustworthy are gradients through a contact solver, and how does the
choice of contact formulation affect them?* That is a destination,
possibly a thesis, and explicitly **not** what the next stretch of work
is about. It is recorded here so the architecture doesn't accidentally
foreclose it — not as a requirement. No ontologies, no hypotheses, no
publication strategy in this repo.

## How we work

Claude writes code, including core physics, dynamics, contact, and
gradient implementation. Move fast.

Two things Claude owes on every non-trivial physics change:

- **Explain what it wrote** — the derivation, the convention choices, why
  this formulation and not another. A short paragraph in the response, not
  a lecture. Anything longer than a paragraph goes in `docs/derivations/`.
- **Flag the choices that were judgment calls**, so they don't silently
  harden into assumptions nobody remembers making.

Pedro reviews, redirects, and makes the calls. If a design decision has
real tradeoffs, surface them and ask rather than picking silently.

This file is **guidelines, not scripture**. It records decisions that were
made deliberately, so they aren't re-litigated by accident — but it is
wrong sometimes, and the build order in particular has already been
rewritten once. If following it would produce something worse, say so.

## What is already built

Steps 1–9, complete and green. 113 tests.

1. **Symplectic (semi-implicit) Euler integrator** for a single 2D rigid
   body under gravity.
2. **Analytic dynamics Jacobians** `∂z_{t+1}/∂z_t` and `∂z_{t+1}/∂f`,
   validated against central finite differences.
3. **Multiple bodies**, parallel vectors, block-diagonal system Jacobian.
4. **Contact detection** — convex polygon against a static half-plane.
   Signed distance, contact normal, contact point, and the analytic
   contact Jacobian `J = ∂d/∂q`.
5. **Penalty contact** — clamped Kelvin–Voigt spring-damper, with its
   force Jacobian and the determinant identity
   `det(dz_dz) = det(Id + dt·M⁻¹·∂f/∂v)`.
6. **Integration separated from force assembly.** `integrate_*` takes a
   force somebody else summed and knows nothing about contact;
   `step_*` sits on top and does the summing.
7. **Rollout gradients** — a reverse-mode adjoint sweep over a
   trajectory, storing states only, since the Jacobians are pure
   functions of state.
8. **Coulomb friction**, clamped Kelvin–Voigt tangentially with a cone
   bound `|β| ≤ μλ`.
9. **Body-body contact** — SAT plus clipping, the gap Hessian via a
   file-local second-order jet, and a genuinely coupled `∂F/∂Q`. With
   friction between bodies.

So: a multi-body differentiable simulator with friction, body-body
contact, and gradients through whole trajectories. Every derivative is
analytic and finite-difference validated.

Known gaps, in the order they matter: **no joints** (every body is
free-floating with a wrench at its COM), no callable API beyond the C++
headers, and no benchmark.

## Build order

Each step tested and green before the next begins.

10. **Joints.** Bilateral constraints, which is the first thing in this
    project that is not a force. Revolute only to start — it is what
    every 2D benchmark needs, and prismatic and weld are the same
    machinery with a different `c(q)`. Reduced coordinates are *out*:
    they would replace `RigidBodyState`, the stacking convention, and the
    constant diagonal mass matrix all at once. Penalty first, with a
    bilateral constraint solve as a known follow-on — the constraint
    function and its Jacobian are shared either way, so the geometry work
    is not thrown away. See the discussion note below.
11. **Public API and Python bindings.** pybind11. Deliberately after
    joints: they change scene construction, and binding twice is what
    putting bindings last was meant to avoid.

Step 10 is the current milestone.

**Not on the numbered list, and wanted:** a benchmark. `benchmarks/` is
empty, and three performance questions have accumulated with nothing to
answer them — detection allocating several vectors per call, `∂F/∂Q`
being dense when the contact graph is sparse, and the adjoint's `O(B²)`
matvec on a matrix that is block-diagonal whenever bodies are not
touching. Each was deferred with "when a benchmark says it matters."

## Note on joints: penalty first, solve later

A joint is a **bilateral** constraint `c(q) = 0` — an equality, with a
force that pushes or pulls whatever is needed. Contact fitted the
architecture because contact *is* a force; a joint is not, and this is
the first thing that genuinely does not fit.

Three options, one immediately out. **Reduced coordinates** would satisfy
constraints exactly by construction, and would also replace
`RigidBodyState`, `Pack`/`Unpack`, and the constant diagonal mass
matrix — reintroducing the `M(q)` and Coriolis terms
`symplectic_euler.md` spends a section celebrating the absence of. It is
a rewrite wearing a feature's clothes.

That leaves **penalty joints** (a very stiff spring-damper between the
anchor points, fits the existing force path with zero structural change)
and a **bilateral constraint solve** (`(J_c M⁻¹ J_cᵀ)λ = …`, exact, but
a solve in the update loop).

The number that decides it: penalty joints hold to `error = F/k_j`, so a
pendulum at ~10 N wants `k_j ≈ 10⁴` and works at the contact timestep,
while a walker landing at ~500 N wants `5×10⁵` and roughly `dt = 10⁻⁴`.
That is **10× more steps than contact alone** — real, and not
disqualifying.

Penalty first, because both approaches need the same `c(q)` and
`J_c = ∂c/∂q`, so the geometry work carries over; because the step 9 jet
already produces the Hessian the force Jacobian needs; and because it
tells us empirically whether the stiffness limit bites for the tasks that
actually get built.

Worth correcting one thing if it comes up: a bilateral solve is **much
easier than the deferred NCP work below**, not a back door to it. No
complementarity, no active set, no disjunction — it is a linear solve,
its IFT gradient is textbook, and it is *smooth*, so none of the contact
gradient pathology applies.

## Deferred: NCP, IFT gradients, formulation swapping

These were steps 6–8 of the original plan. They are **future ideas, not
current work**, and nothing should be built in anticipation of them:

- **Velocity-level NCP contact solver.** Contact as a complementarity
  constraint solved per step rather than a force — exact non-penetration
  instead of penalty's sub-millimetre sink.
- **IFT gradients through the contact solve**, in the style of Dojo.
  Required because you cannot chain-rule through a solver; this is what
  would make an NCP formulation differentiable at all.
- **Formulation swapping** — running the same scenario through penalty
  and through NCP.

Why deferred: the eventual comparison is only measurable once there is a
control/RL stack to measure it *with*. Building a second formulation
first produces something with nothing to compare on. Note also that
penalty and NCP are different **update rules**, not interchangeable force
laws — penalty is a force, NCP is a post-force projection — so
"swapping" was never going to be a plug-in interface, and no abstraction
should be built pretending otherwise.

What keeps the door open is step 6's seam, not any amount of generality.
Revisit when a consumer repo can actually train something and the
question "does the contact model change what it learns" becomes a
measurement rather than a guess.

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
  representation; each direction has a real cost — see
  `docs/derivations/integrator_jacobians.md`.
- **Multiple bodies are parallel vectors, not a bundled struct.**
  `std::vector<RigidBodyState>` + `std::vector<RigidBodyParams>` +
  `std::vector<BodyShape>`, indexed together. System state is
  `PackSystem`/`UnpackSystem`'s concatenation (body `i` at `[6i, 6i+6)`),
  built on the single-body convention above.
- **The system Jacobian is built alongside every step, not deferred.**
  It is exactly block-diagonal while bodies only touch static scenery —
  each diagonal block is the per-body Jacobian, and penalty contact
  against the plane only enriched those blocks. Body-body contact (step
  9) is the first thing that introduces genuine off-diagonal terms. See
  `docs/derivations/multi_body_system.md`.
- **Strictly deterministic.** Same input, same binary, same output, bit
  for bit. No unordered container iteration, no unstable sorts, no
  order-dependent floating-point accumulation, no uninitialized memory.
- **The rollout path and the gradient path share one physics
  implementation.** Never let them diverge. This is why force laws
  compute their own inputs rather than taking precomputed ones, and why
  every analytic derivative is finite-difference checked against the
  actual forward function.
- **The public surface is for consumers.** Assume another repository is
  calling this one. Prefer explicit arguments over ambient state, keep
  scene construction data-driven, and don't require a caller to
  understand the integrator's internals to run a rollout.

New dependencies beyond Eigen, GoogleTest, and (later) pybind11 need
discussion first. **raylib** is in, for the demo only, behind
`GRIP_BUILD_DEMOS` — it is fetched and linked by `demos/` and by nothing
else. That option defaults ON while there are no consumers; flip it OFF
at the bindings step, since a physics library should not drag a window
toolkit into anything that links it.

## Code style

Pedro's preferences. Follow them in new code; don't reintroduce the old
patterns when editing.

- **Function parameters go on one line**, however long the line gets.
  Never wrap a parameter list one-per-line, in declarations, definitions,
  or calls.
- **No function overloading.** Two functions that take different
  arguments or return different things get different names — reading a
  call site shouldn't require resolving which overload is meant from the
  argument types. This is why the integrator has `step_body` /
  `step_system` (and `step_body_jacobian` / `step_system_jacobian`)
  rather than one overloaded name per pair. When a function gains a
  "same thing, but for N of them" variant, name it; don't overload it.
- **Two blank lines between top-level definitions in `src/`** — functions,
  structs, `using` declarations. One blank line before and after
  `}  // namespace`. Test files use a single blank line between `TEST`
  blocks. Never "clean up" extra blank lines between functions; they are
  deliberate.

## Repository layout

```
grip/
  src/
    core/        types, math helpers, rigid body state
    dynamics/    mass matrices, forces, integrator
    contact/     detection (half-plane and polygon-polygon), formulations
    gradient/    rollout / adjoint path
  tests/
    unit/        per-component tests
    validation/  finite-difference and conservation checks
  demos/         raylib, gated behind GRIP_BUILD_DEMOS
  docs/
    derivations/ the math, in Markdown + LaTeX
  benchmarks/    empty; see the note under the build order
  CMakeLists.txt
```

`bindings/` arrives with step 10.

## Testing

- Every derivative gets a finite-difference test against its analytic
  counterpart. Written alongside the implementation, not after.
- Energy-behavior tests for the integrator: symplectic Euler should show
  bounded energy oscillation, not secular drift. That test is worth more
  than a single-step tolerance check.
- Structural invariants earn their keep over time. `det(dz_dz) = 1` for
  conservative forces was written at step 2 while it was nearly vacuous,
  gained teeth at step 5a with a nonzero `∂f/∂q`, and broke exactly as
  predicted at 5b. Prefer assertions about structure to assertions about
  numbers where both are available.
- Near contact activation boundaries, gradients are *expected* to
  misbehave. Tests there record and characterize the behavior rather
  than enforce a tolerance. That boundary is the interesting part, not a
  bug.
- **Never weaken a tolerance to make a test pass.** Investigate instead.
  If a tolerance was wrong, say why it was wrong — an unjustified
  guess about floating-point behavior is a reason; "it fails otherwise"
  is not.

## Notation

`docs/derivations/notation.md` is the canonical symbol table for the
whole project. **No symbol means two things.** Before introducing a new
quantity in a derivation, add it there and check it against what's
already taken — several collisions (`J` as both a rotation and a
Jacobian, `x` as both a coordinate and the stacked state, `I` as both
inertia and identity) had to be untangled after the fact, which is
exactly what that file exists to prevent. Case is not enough separation:
`N`/`n`, `G`/`g` and `F`/`f` all count as collisions.

When every short candidate collides, spell the name out — `Delassus`,
`eig_max` — rather than inventing a letter.

Math notation belongs in `docs/derivations/`. Code uses descriptive
identifiers — `signed_distance`, not `d`. The Jacobian member names
(`dz_dz`, `dZ_dZ`, …) are the deliberate exception, where matching the
math is clearer than prose.

## Derivations

Non-trivial math — integrator update, mass matrix construction, contact
formulations, frame conventions, the adjoint derivation — lives in
`docs/derivations/` as Markdown with LaTeX. Claude writes these as it
implements. Code comments reference the derivation rather than
re-deriving inline.

If code and derivation disagree, flag it. Don't silently pick one. If a
derivation cites numbers, they must come from something that still runs
in the repo.

## Things to do proactively

- Flag hot-loop allocations, hidden Eigen temporaries, aliasing issues
  (`.noalias()` where appropriate).
- Flag determinism risks.
- Suggest missing edge-case tests, especially near contact activation.
- Keep CMake tidy; new source files get wired into the build and a test
  target in the same change.
- Say when something is out of scope for the current build step, or
  belongs in a consumer repository rather than here.

## Things not to do

- Add abstraction layers, plugin systems, or generality no current test
  needs. Elegance that doesn't unlock a measurement is wasted effort.
- Build toward the deferred NCP work. The seam in step 6 is the whole
  hedge; nothing else should be shaped by a formulation that doesn't
  exist.
- Optimize before a benchmark shows it matters.
- Implement control algorithms, policies, or training loops here.
- Reintroduce research framing into the codebase or its docs.
- Commit commented-out code or TODO placeholders in reviewed paths.

## Standing reminder

If a session drifts toward architecture astronomy, research positioning,
or features that aren't the next numbered step, say so and point back to
the build order. Right now that step is 10: revolute joints as a penalty
force law, on a green suite, before the API surface gets bound.

Keep this file current as steps land. It went stale once already —
rewritten at the replan, then left claiming step 6 was next while 6
through 9 all shipped — which is the kind of drift that makes a session
start by trusting the wrong thing.
