# Penalty contact — spring and damper

Symbols follow `docs/derivations/notation.md`. Built in two increments,
5a and 5b, kept separate on purpose: each is pinned by a structural
invariant that the other cannot express, and shipping them together
would have collapsed both into a single finite-difference check.

## The force law

Detection (`contact_detection.md`) already supplies, per vertex, the gap
`dᵢ = n·pᵢ − o` and the contact Jacobian `Jᵢ = ∂dᵢ/∂q`. A contact is
active when `dᵢ < 0`, and the penetration depth is then just `−dᵢ` — no
separate symbol is needed for it, which is why nothing here is called
`φ`.

**5a, the spring alone:**

```
λᵢ = −k·min(0, dᵢ)
f_c = Σᵢ Jᵢᵀ λᵢ
```

Written with `min` rather than a branch, the one-sided spring is a single
expression and the activation test lives in exactly one place.

**5b adds a damper in the closing rate**, `ḋᵢ = Jᵢ·v` — which detection
already gives us, since `d` is a normal gap so `Jᵢ·v` is already the
normal component with no projection step:

```
λᵢ = dᵢ < 0 ? max(0, −k·dᵢ − b·ḋᵢ) : 0
```

Two gates doing two different jobs. The `dᵢ < 0` test is **activation**:
a separated vertex feels nothing however fast it is moving. The outer
`max` is the **adhesion clamp**, and it is not cosmetic — see below.

## Why `Jᵀ` is the whole force-to-wrench conversion

`λᵢ` is a scalar — a push along `n` applied *at the vertex*. The state is
a frame at the center of mass, so that push has to become both a force
and a moment. Expanding `Jᵢ = [n_x, n_y, n·(pᵢ−c)^⊥]`:

```
Jᵢᵀλᵢ = ( λᵢn_x , λᵢn_y , λᵢ·n·(pᵢ−c)^⊥ )
```

The third component *is* the moment. With `r = pᵢ − c` and `F = λᵢn`, the
2D cross product is `r_x F_y − r_y F_x = λᵢ(r_x n_y − r_y n_x)`, and
`n·r^⊥ = n·(−r_y, r_x) = r_x n_y − r_y n_x`. Identical. So the moment arm
never has to be handled separately — virtual work hands it over, and the
same `Jᵀ` that will map multipliers to generalized forces in step 6 is
already doing it here.

Two consequences the code makes visible:

- **Multiple contacts collapse to one 3-vector.** Four penetrating
  vertices produce four `λᵢ`, but the body only ever feels the sum. This
  is the resultant argument from `contact_detection.md`: the dynamics fix
  the total and the center of pressure, not the individual forces. A flat
  resting box is statically indeterminate, and penalty contact sidesteps
  that only because each spring answers to its own `dᵢ`. The NCP solver
  in step 6 has to confront it directly.
- **A contact directly below the COM produces no torque.** Then
  `r = (0, −h)`, `r^⊥ = (h, 0)`, and `n·r^⊥ = 0` for `n = (0,1)`. That's
  the balanced-on-a-corner case, and it is why the force test uses a
  tilted configuration instead: a vertical moment arm would let a missing
  torque term pass unnoticed.

## The Jacobian is a Hessian

The spring is conservative. With

```
U = Σᵢ (k/2)·min(0, dᵢ)²
```

we have `∂U/∂q = Σᵢ k·min(0, dᵢ)·Jᵢ`, so `f_c = −(∂U/∂q)ᵀ = Σᵢ Jᵢᵀλᵢ`,
matching the force law above. Differentiating once more, and using the
product rule because *both* factors depend on `q`:

```
∂f_c/∂q = −∇²U = −k Σᵢ JᵢᵀJᵢ  +  Σᵢ λᵢ ∂Jᵢᵀ/∂q
                  ^material        ^geometric
```

This is the material/geometric stiffness split from nonlinear structural
mechanics, arriving here for the same reason it does there: a force whose
line of action moves as the body deforms — or in this case rotates.

The geometric term is remarkably confined. `Jᵢ` has constant first and
second components, and its third depends only on `θ`. Since perp applied
twice is negation:

```
∂/∂θ [ n·(R(θ)rᵢ)^⊥ ] = n·((R(θ)rᵢ)^⊥)^⊥ = −n·(pᵢ − c)
```

so `∂Jᵢ/∂q` has **exactly one** nonzero entry, at `(θ, θ)`, and the
geometric term contributes only `−λᵢ·n·(pᵢ − c)` there. Everything else
is material.

`∂f_c/∂v = 0` — the spring reads position only.

### What the damper adds

For contacts that are active *and* unclamped, `λᵢ = −k·dᵢ − b·ḋᵢ`, so

```
∂λᵢ/∂q = −k·Jᵢ − b·∂ḋᵢ/∂q          ∂λᵢ/∂v = −b·Jᵢ
```

and `ḋᵢ = Jᵢ(q)·v` inherits `Jᵢ`'s single `θ` dependence:

```
∂ḋᵢ/∂q = [ 0, 0, −ω·n·(pᵢ − c) ]
```

Assembling:

```
∂f_c/∂q = −k Σᵢ JᵢᵀJᵢ  −  b Σᵢ Jᵢᵀ ∂ḋᵢ/∂q  +  Σᵢ λᵢ ∂Jᵢᵀ/∂q
∂f_c/∂v = −b Σᵢ JᵢᵀJᵢ
```

Two things worth noticing. The geometric term now carries the **full**
`λᵢ`, damping included, because it comes from differentiating `Jᵢᵀ` in
the product `Jᵢᵀλᵢ` — the damping part of the force has a moment arm just
like the spring part does. And the new middle term is confined entirely
to the `θ` column, since `∂ḋᵢ/∂q` has only that one nonzero component.

Clamped and separated contacts contribute nothing at all: both branches
return `λᵢ = 0` identically, so all three terms vanish.

`∂f_c/∂v = −b Σ JᵢᵀJᵢ` is symmetric negative semidefinite by
construction, which is the statement that the damper can only remove
energy: `wᵀ(∂f_c/∂v)w = −b Σ (Jᵢ·w)² ≤ 0`, and at `w = v` that is exactly
the dissipation rate.

### Symmetry is a free correctness check, and the damper expires it

At `b = 0`, `∂f_c/∂q` is `−∇²U`, hence symmetric — and in floating point
symmetric *exactly*: the material term is a sum of outer products
`JᵢᵀJᵢ`, whose `(a,b)` and `(b,a)` entries are the same product of the
same two numbers accumulated in the same order, and the geometric term
touches only a diagonal entry.

Damping ends that. It is non-conservative, so `∂f_c/∂q` stops being a
Hessian, and the term `−b Σ Jᵢᵀ ∂ḋᵢ/∂q` is a column times a row that is
not proportional to `Jᵢ` — rank-1 and asymmetric, living wholly in the
third column. Symmetry is a valid Jacobian check for the spring and must
not be carried past it.

## Continuity at the activation boundary

The boundary is the set where some `dᵢ = 0`. It is a codimension-1
surface depending on `q` alone.

`−k·min(0, d)` is zero from both sides, so **the force is continuous**.
Its slope is `−k` from inside and `0` from outside, so **the derivative
is not**. C⁰, not C¹.

Where the jump lands is the useful part. The geometric term is
proportional to `λᵢ`, which goes to zero as `dᵢ → 0⁻`, so it approaches
the boundary *continuously*. The material term does not. Therefore:

> The entire discontinuity in `∂f_c/∂q` at the boundary is `k JᵀJ`. The
> jump in the derivative **is** the stiffness.

That is the bind penalty contact cannot escape: the knob that makes the
physics good — stiffer contact, less penetration — is the same knob that
makes the gradient discontinuity larger. It is not a tuning problem.

A central difference straddling the boundary returns the average of the
two one-sided slopes: `−k/2` per active contact, matching neither branch
but staying **bounded** as the step shrinks.
`tests/validation/test_contact_boundary.cpp` records that rather than
asserting it away.

At exactly `dᵢ = 0` the implementation reports the separated branch —
zero force, zero stiffness contribution — which is consistent with the
force value there. The derivative genuinely does not exist; this is a
convention, not a computation.

### The damper breaks continuity outright

The spring term dies smoothly into the boundary. The damping term has no
factor that vanishes with `dᵢ`, so approaching from inside,

```
λᵢ → 0 − b·ḋᵢ = b·|ḋᵢ|      whenever ḋᵢ ≠ 0
```

while from outside it is zero. **The force itself jumps** — a step, not a
kink — by `b·|ḋᵢ|` per contact.

Two consequences. First, the jump is proportional to **closing speed**, so
a body grazing tangentially (`ḋᵢ = 0`) crosses the boundary with no
discontinuity at all, while a hard impact crosses a cliff scaled by how
hard. The pathology is not uniform over the boundary surface; it
concentrates exactly where impacts are fast. That is a far more useful
thing to characterize than "gradients are bad near contact."

Second, a central difference straddling a jump returns

```
−k − b·|ḋᵢ|/ε
```

which **grows without bound** as `ε` shrinks, roughly a factor of ten per
decade. That is the sharp discriminator between the two severity classes,
and it is what `test_contact_boundary.cpp` measures: the same sweep that
sits still at `−k` for the spring runs away for the damper.

Because the integrator applies `dt·M⁻¹·f`, a discontinuous force makes
the **one-step map itself** discontinuous, not merely kinked. The jump is
`O(dt)`, so refining the timestep shrinks the cliff without ever
converting it into a kink.

### The clamp fixes the exit and not the entry

This asymmetry is easy to state imprecisely, so: writing the law out,
`λᵢ = max(0, −k·dᵢ − b·ḋᵢ)` while `dᵢ < 0`.

**Leaving** (`ḋᵢ > 0`), the clamp binds as soon as `b·ḋᵢ > −k·dᵢ`, which
happens *while the vertex is still penetrating*. So `λᵢ` reaches zero
strictly before `dᵢ` does, and separation is continuous. Without the
clamp, `λᵢ` would go negative there — the plane pulling a departing body
back down. That is adhesion, and it is not subtle: it fires on every
bounce.

**Entering** (`ḋᵢ < 0`), `−k·dᵢ − b·ḋᵢ` is strictly positive the instant
`dᵢ` goes negative, so the `max` never engages and the force steps.

So "clamping fixes the damper's continuity problem" is half true, and the
false half is the half that fires on impact. The clamp also introduces
its own non-smooth surface at `b·ḋᵢ = −k·dᵢ`, but that one is a
relu-style kink rather than a jump, and it sits strictly *inside* the
contact region rather than on the activation boundary — a milder
pathology in a different place.

### The severity hierarchy

| model | `f_c` at `dᵢ = 0` | one-step map | central difference across it |
|---|---|---|---|
| spring only | C⁰, not C¹ | C⁰, kinked | bounded; blends the two one-sided slopes |
| spring + damper | jump `b·\|ḋᵢ\|` on entry | **jump** `dt·M⁻¹·b·\|ḋᵢ\|·Jᵢᵀ` | diverges as `1/ε` |
| Hunt–Crossley | C⁰, not C¹ | C⁰, kinked | bounded |

Hunt–Crossley, `λᵢ = −dᵢ(k − b·ḋᵢ)`, is listed because it is the obvious
step 8 comparison: multiplying the damping by depth makes it vanish
continuously at `dᵢ = 0` from both sides, buying back exactly the
continuity the damper cost and not one degree more. Nothing in this
family is C¹ at the boundary. That is irreducible, and it is what
"penalty contact differentiates trivially" actually means — the
derivative exists almost everywhere and is cheap, not that it is
continuous.

## The determinant identity

`integrator_jacobians.md` derives `det(dz_dz) = 1` for `∂f/∂v = 0`. That
result generalizes, and the general form is what makes 5a and 5b
distinguishable.

Writing `A = ∂f/∂q` and `D = ∂f/∂v`, the step Jacobian is

```
dz_dz = ⎡ Id + dt²M⁻¹A    dt(Id + dt M⁻¹D) ⎤
        ⎣ dt M⁻¹A          Id + dt M⁻¹D    ⎦
```

Take `S = Id + dt·M⁻¹D` as the lower-right block, so the upper-right
block is exactly `dt·S`. The block formula
`det = det(S)·det(P − Q S⁻¹ R)` gives

```
P − Q S⁻¹ R = Id + dt²M⁻¹A − (dt·S)(S⁻¹)(dt M⁻¹A) = Id + dt²M⁻¹A − dt²M⁻¹A = Id
```

and therefore

```
det(dz_dz) = det(Id + dt·M⁻¹·∂f/∂v)
```

**`∂f/∂q` never affects the determinant at all.** The earlier `det = 1`
was the `D = 0` special case.

For step 5a, `∂f_c/∂v = 0`, so `det(dz_dz) = 1` exactly — but now with a
nonzero, configuration-dependent `A`, so the cancellation is doing real
work rather than being vacuous. That is the single reason the spring
ships as its own increment.

For step 5b, `∂f_c/∂v = −b·J_Aᵀ J_A`, where `J_A` stacks the Jacobians of
the contacts actually carrying force — active *and* unclamped. By
Sylvester's determinant identity,

```
det(dz_dz) = det(Id₃ − dt·b·M⁻¹J_AᵀJ_A) = det(Id_a − dt·b·Delassus)
```

with `Delassus = J_A M⁻¹ J_Aᵀ`. For a single active contact this collapses
to `1 − dt·b·(J M⁻¹ Jᵀ)`.

So the damper's per-step phase-space contraction and step 6's central
operator are the same object measured two ways — the Delassus operator
turns up in step 5 already, and `test_penalty_jacobians.cpp` checks the
identity to `1e-14` by assembling it independently from detection output
and the mass matrix. It is a much sharper assertion than "the determinant
is no longer 1": for a flat-resting unit square with `b = 50`,
`dt = 1e-3`, `Delassus = [[2.5, −0.5], [−0.5, 2.5]]` and the determinant
is exactly `0.765`.

## What actually caps the stiffness

During a contact episode the normal mode obeys `d̈ = J M⁻¹ f_c` (exactly
when `J` is constant, as for a box landing flat; up to a `J̇v` correction
otherwise), so with `f_c = Jᵀλ` and `λ = −k d`:

```
d̈ = −k·Delassus·d
```

The episode is half a period, `π/√(k·Delassus)`, and the number of
integrator steps spent resolving it is

```
N = π / (dt·√(k·Delassus))
```

The stability bound inherited from `symplectic_euler.md` is
`dt²·k·eig_max(Delassus) < 4`, which corresponds to `N ≈ 1.6` — the
bounce is barely sampled. Resolving one properly needs 10–20 steps, which
is roughly **150× more restrictive**, and that, not stability, is what
caps `k` in practice.

For a unit square of unit mass resting flat, with `w = 0.5` and
`I = m(a²+b²)/12 = 1/6`, the active contact Jacobians are `[0, 1, ∓w]`
and

```
Delassus = ⎡ 1/m + w²/I   1/m − w²/I ⎤        eigenvalues:  2/m  and  2w²/I
           ⎣ 1/m − w²/I   1/m + w²/I ⎦
```

which are `2` and `3`. **The rotational mode is the stiff one** — the
naive `k/m` estimate from translation alone understates it by 3/2.

At `dt = 1e-3`:

| steps in contact | `k` | resting penetration `mg/(2k)` |
|---|---|---|
| 1.6 (stability limit) | 1.3×10⁶ | 3.7 µm |
| 10 | 3.3×10⁴ | 0.15 mm |
| 20 | 8.2×10³ | 0.59 mm |
| 50 | 1.3×10³ | 3.7 mm |

Sub-millimetre penetration at a usable timestep is the honest number, and
the standard complaint about penalty methods. It is also the concrete
motivation for step 6.

## Resting equilibrium

Two active corners share the weight, so `2·k·(−d) = mg` and

```
d* = −mg/(2k)
```

At `v = 0` this is a fixed point of the **discrete** map, not merely of
the continuous equations: the net force is identically zero, so `v_{t+1}`
is zero and `q_{t+1} = q_t`. The corner moments cancel by symmetry, so it
holds for any inertia.

With no damping a dropped box never converges here — it bounces
elastically forever, returning to its drop height every time. Reaching
rest requires dissipation, and it is the one behaviour the spring alone
cannot fake.

With the damper on it does converge, and to exactly this point.
`test_energy_behavior.cpp` drops a unit square from `y = 1.2` with
`k = 10⁴`, `b = 50` and checks that it arrives at `|v| < 10⁻⁶` and
`d* = −mg/(2k) = −0.4905 mm` — the same fixed point
`test_penalty_force.cpp` pins exactly by construction, reached here by
simulation instead.

## Energy

With `b = 0` the contact force is `−∇U`, so the conserved quantity for a
bouncing body is kinetic plus gravitational plus stored spring energy.
Symplectic Euler keeps it in a bounded band rather than drifting, as in
`symplectic_euler.md` — with one caveat worth stating: the backward error
analysis that produces that guarantee assumes a **smooth** force, and the
penalty force is C⁰-not-C¹ at the kink. Per-bounce jitter is therefore
real and expected, and depends on where the timesteps happen to land
within the episode. Volume preservation survives regardless, because
`det(dz_dz) = 1` is algebra and does not care about smoothness.

With `b > 0` that same total is no longer conserved but is
**monotonically non-increasing** — gravity and the spring are both
conservative, so the damper is the only thing that can move it, and
`v·f_damp = −b Σ(Jᵢ·v)² ≤ 0`. `test_energy_behavior.cpp` asserts the
total never rises above its starting value and that the box settles.

The clamp matters for that claim. An unclamped Kelvin–Voigt law can
*inject* energy during separation, because a negative `λᵢ` with the body
moving away does positive work on it. Bounding `λᵢ` below by zero removes
that path, which is a second reason for the clamp beyond the obvious
physical absurdity of a sticky plane.
