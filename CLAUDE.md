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

The work is planned in three releases, set out under **Release plan**
below. Briefly: **1.0** puts a public API and Python bindings on the
physics that already exists, **2.0** replaces penalty contact with a
velocity-level NCP solve and IFT gradients and adds joints inside that
same solve, and **3.0** is parallelism. Each one changes what the library
*is*. None of them is a research programme.

One question was dropped deliberately, and is recorded here so it doesn't
get reinvented: *how does the choice of contact formulation affect
gradient quality?* The literature has largely answered it — first-order
gradients through contact are not reliably better than zeroth-order
estimators in stiff regimes, and what repairs them is smoothing rather
than the choice of formulation. That collapses the question into a knob
rather than a comparison, which is not enough to build a repository
around. No ontologies, no hypotheses, no publication strategy here.

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
wrong sometimes, and the plan of work in particular has already been
rewritten twice. If following it would produce something worse, say so.

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

Known gaps, in the order they matter: **no callable API** beyond the C++
headers, no joints (every body is free-floating with a wrench at its
COM), and no benchmark. That order changed when joints moved into 2.0.

## Release plan

Each step tested and green before the next begins. Step numbering
continues from the nine already built.

### 1.0 — make it callable

10. **Public API and Python bindings.** pybind11, over the free-body,
    contact-and-friction physics that is already built and green. **No
    new physics ships in 1.0.**

    **The API takes a batch of scenes, not one.** That is the single
    decision at this step that is expensive to reverse: 3.0 needs the
    batched shape, and adding a batch dimension after bindings exist is
    exactly the rebinding this ordering was meant to prevent. A batch of
    one is a fine degenerate case; a scalar API that later grows a batch
    dimension is not.

    One further hedge, much cheaper: shape the scene description so a
    `joints` vector arrives later as an **added field** rather than a
    signature change. Joints land in 2.0, and this is what keeps that
    from forcing a rebind.

Step 10 is the current milestone.

**Joints used to be step 10, and are now deferred into 2.0.** The reason
they sat in front of the API was that they change scene construction —
but a data-driven scene absorbs a joints vector as an added field, which
is a recompile rather than a redesign, and the API decision that is
genuinely expensive to reverse is the batch dimension, which is
orthogonal to joints. Meanwhile penalty joints would have been
scaffolding replaced by 2.0's solver, at a 10× timestep cost for exactly
the articulated systems they exist to enable. See the note below.

### 2.0 — replace penalty contact, and add joints

A velocity-level **NCP contact solve** with **IFT gradients**, in the
style of Dojo. Previously listed here as deferred; it is now the plan.

The argument for it is *not* "NCP gradients are better." It is that
penalty couples two knobs that should be independent.
`penalty_contact.md` states the bind plainly: **the jump in the
derivative is the stiffness**. The knob that improves the physics is the
same knob that degrades the gradient, and no amount of tuning escapes it.
An NCP solve with a central-path relaxation breaks that coupling —
non-penetration is exact regardless of the relaxation parameter, and the
relaxation controls only gradient smoothness. Hard contact and smooth
gradients still trade off against each other; the point is that they
become *separate* knobs.

Two consequences for implementation:

- **The solver choice is the gradient story.** Interior-point with a
  strictly positive relaxation gives smooth IFT derivatives. Projected
  Gauss–Seidel gives an active set, a piecewise derivative, and the same
  pathology relocated. Don't build PGS and expect the gradient benefit.
- **In 2D the friction cone is exact.** It degenerates to a wedge with
  two edges, so there is no polygonalization and none of the
  direction-dependent artefacts 3D engines pay for — already noted in
  `penalty_contact.md`, and it applies to the solve as much as to the
  penalty law.

The other win, larger for anything that trains on this, is the timestep:
penalty needs `dt ≈ 5e-4` to resolve a bounce, an NCP solve runs at
`1e-2`, with exact non-penetration and stable stacking.

**Joints ship here too**, as bilateral rows in the same solve. A
bilateral constraint and a contact constraint land in the same linear
algebra, so building the solver is most of building joints — and they
come out *exact* rather than held to `error = F/k_j`, with no timestep
penalty. That also makes 2.0 the release where the classic 2D benchmarks
(CartPole, Acrobot, Hopper) become possible and well-behaved at the same
time, which is a far larger visible jump than "the same scene with better
contact."

2.0 is therefore three deliverables — solver, IFT gradients, joints — and
that is a genuine risk of a release that never lands. The mitigation is
this repo's usual discipline applied *inside* the version number rather
than between versions: contact solve green, then IFT gradients green,
then bilateral rows green. Sequence it that way when it starts.

### 3.0 — parallelism

CUDA, pursued as an explicit learning goal rather than because a
benchmark demanded it. Three constraints, worth recording now because two
of them reach back into 1.0:

- **Across scenes, not within one.** A 2D scene with ten bodies cannot
  saturate a GPU. The parallelism that pays is a batch of independent
  scenes — which is why 1.0's API takes a batch.
- **CPU threads first.** A thread pool over scenes is roughly a day of
  work, captures most of the win, stays deterministic, and proves out the
  batched shape a GPU backend would need anyway.
- **Determinism is in tension with the GPU.** "Same input, same binary,
  same output, bit for bit" is hard when reductions and atomics are
  order-dependent. Either deterministic reductions get built, or the
  invariant gets narrowed — consciously, and written down here when it
  happens.

### The benchmark, now overdue

`benchmarks/` is empty, and four performance questions have accumulated
with nothing to answer them:

- detection allocating several vectors per call
- `∂F/∂Q` dense when the contact graph is sparse
- the adjoint's `O(B²)` matvec on a matrix that is block-diagonal
  whenever bodies are not touching
- `penalty_forces_system_jacobian` calling five detection entry points
  per pair, each recomputing `ComputePairGeometry` from scratch, with the
  last two rebuilding jets the middle two just built

Each was deferred with "when a benchmark says it matters." Several
decisions now wait on the answer, so it has stopped being optional.

## Note on joints: deferred into 2.0's solve

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

The number that used to decide it: penalty joints hold to
`error = F/k_j`, so a pendulum at ~10 N wants `k_j ≈ 10⁴` and works at
the contact timestep, while a walker landing at ~500 N wants `5×10⁵` and
roughly `dt = 10⁻⁴`. That is **10× more steps than contact alone**.

**The plan was penalty joints first, and that is now dropped.** Not
deferred within 1.0 — dropped. The whole case for building them was that
they were cheap and would unblock the API, and both halves fell over:

- The API doesn't need them. A data-driven scene absorbs a `joints`
  vector as an added field, so binding before joints exist costs a
  recompile, not a redesign. The API decision that *is* expensive to
  reverse is the batch dimension, which has nothing to do with joints.
- They would be replaced almost immediately by 2.0's solver, after
  imposing that 10× timestep cost on precisely the articulated systems
  they exist to enable — right before 3.0, whose entire purpose is
  throughput.

Building something you already know you will delete, which also makes the
simulator slower in the meantime, is not a milestone.

So joints land in 2.0, as bilateral rows in the contact solve. Nothing is
lost by waiting: both approaches need the same `c(q)` and
`J_c = ∂c/∂q`, so the geometry work is identical whenever it happens, and
the step 9 jet already produces the Hessian either form would need.

Scope when it arrives: **revolute only to start.** It is what every 2D
benchmark needs, and prismatic and weld are the same machinery with a
different `c(q)`.

One thing worth keeping straight when 2.0 starts: a bilateral solve is
**much easier than the contact one**, not a back door to it. No
complementarity, no active set, no disjunction — a linear solve, a
textbook IFT gradient, and *smooth*, so none of the contact gradient
pathology applies. Doing joints after the contact solver is doing the
easy half second.

## Penalty stays; modularity does not

2.0 replaces penalty contact as the **default**, not as the only thing in
the repository. Penalty keeps its entry points and its tests.

Three reasons, none of which is generality:

- It is the **validation baseline**. When the solver produces a
  trajectory nobody believes, "does penalty at a tiny timestep agree?" is
  the most useful question available, and there is no substitute for it.
- IFT gradients are **much harder to finite-difference** than force-law
  gradients, because the solve carries its own tolerance floor. Keeping a
  formulation whose derivatives are already validated against the actual
  forward function is worth real money during 2.0.
- It is already written, already green, and costs nothing to leave alone.

**What is not being built is a formulation abstraction.** No interface,
no runtime switch, no plugin, no common base that penalty and NCP both
implement. Penalty is a force; an NCP solve is a post-force projection.
They are different **update rules**, not interchangeable force laws, and
any abstraction over them would be pretending otherwise. An earlier plan
listed "formulation swapping" as a feature; it was never going to be one,
and it is now explicitly dropped rather than merely deferred.

Concretely: two named entry points, in the same spirit as `step_body` /
`step_system`. A caller picks a formulation by calling it, and reading a
call site tells you which one is running. That is the whole mechanism. If
a change starts introducing a type that both formulations satisfy, it has
gone wrong.

What keeps the seam honest is step 6 — integration separated from force
assembly — and nothing else. No further generality is needed and none
should be added.

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
    api/         scene description and batched entry points (step 10)
  tests/
    unit/        per-component tests
    validation/  finite-difference and conservation checks
  demos/         raylib, gated behind GRIP_BUILD_DEMOS
  docs/
    derivations/ the math, in Markdown + LaTeX
  benchmarks/    baseline throughput; see benchmarks/README.md
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
- Build toward 2.0 or 3.0 while 1.0 is unfinished. Step 6's seam and the
  batched API shape are the whole hedge; nothing else should be shaped by
  a formulation or a backend that doesn't exist yet.
- Optimize before a benchmark shows it matters.
- Implement control algorithms, policies, or training loops here.
- Reintroduce research framing into the codebase or its docs.
- Commit commented-out code or TODO placeholders in reviewed paths.

## Standing reminder

If a session drifts toward architecture astronomy, research positioning,
or features that aren't the next numbered step, say so and point back to
the release plan. Right now that step is 10: a batched public API and
Python bindings, over the physics that already exists and on a green
suite. No new physics belongs in 1.0.

2.0 and 3.0 are written down so they aren't re-litigated, and so 1.0's
API doesn't foreclose them — **not** so they get built early. Nothing
should be shaped in anticipation of the NCP solve beyond step 6's seam,
and nothing in anticipation of CUDA beyond the batched API shape.

Keep this file current as steps land. It went stale once already —
rewritten at the replan, then left claiming step 6 was next while 6
through 9 all shipped — which is the kind of drift that makes a session
start by trusting the wrong thing.
