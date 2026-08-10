# Rollout gradients — the adjoint sweep

Symbols follow `docs/derivations/notation.md`.

## What this is for

Steps 2–6 produce per-step Jacobians: `dZ_dZ = ∂Z_{t+1}/∂Z_t` and
`dZ_dF = ∂Z_{t+1}/∂U_t`. A rollout is `Z₀ → Z₁ → … → Z_H` under controls
`U₀ … U_{H−1}`, and what a caller actually wants is the gradient of a
scalar objective with respect to the things it can choose: where the
trajectory started, and what drove it.

Composing the per-step Jacobians is the whole job. The question is which
direction to compose them in.

**Forward mode** propagates `∂Z_t/∂θ` forward alongside the state. Cost
scales with the number of parameters, and there are `3B·H` of them — three
per body per step — so it is `O(H²)`.

**Reverse mode**, the adjoint, propagates sensitivity backward from the
objective. Cost is `O(H)` regardless of how many parameters there are.
That gap is the entire reason this is a pass of its own rather than a loop
over `step_system_jacobian`.

## Counts

Getting these wrong is the most common way to write a broken adjoint, so
they are worth stating before any algebra:

| object | how many | index range |
|---|---|---|
| states `Z_t` | `H + 1` | `0 … H` |
| controls `U_t` | `H` | `0 … H−1` |
| step Jacobians | `H` | `0 … H−1` |
| adjoints | `H + 1` | `0 … H` |

## Deriving the recursion

Write the objective with running and terminal costs, and treat the
dynamics as constraints rather than substitutions:

```
minimise    J = Σ_{t=0}^{H−1} ℓ_t(Z_t, U_t) + ℓ_H(Z_H)
subject to  Z_{t+1} − F(Z_t, U_t) = 0        for t = 0 … H−1
```

Adjoin each constraint with a multiplier:

```
𝓛 = Σ_{t=0}^{H−1} ℓ_t + ℓ_H + Σ_{t=0}^{H−1} adjoint_{t+1}ᵀ (F(Z_t, U_t) − Z_{t+1})
```

The multiplier on the constraint that *produces* `Z_{t+1}` is indexed
`t+1`. That is a choice; indexing it `t` gives the same answer with a
shift carried through every line below.

The rest is bookkeeping about **where each `Z_t` appears**.

**Interior, `1 ≤ t ≤ H−1`.** Three places: its own stage cost; constraint
`t`, inside `F`; and constraint `t−1`, as the bare `−Z_t`.

```
∂𝓛/∂Z_t = ∂ℓ_t/∂Z_t + (dZ_dZ)ₜᵀ·adjoint_{t+1} − adjoint_t = 0

  ⟹   adjoint_t = ∂ℓ_t/∂Z_t + (dZ_dZ)ₜᵀ·adjoint_{t+1}
```

**Terminal, `t = H`.** Only two places — `ℓ_H`, and constraint `H−1` as
`−Z_H`. There is no constraint `H` for it to sit inside, so the recursion
has nowhere further to reach:

```
∂ℓ_H/∂Z_H − adjoint_H = 0   ⟹   adjoint_H = ∂ℓ_H/∂Z_H
```

**Initial, `t = 0`.** Also two places, but the other two — `ℓ₀`, and
constraint `0` inside `F`. There is no constraint `−1`, so no bare `−Z₀`
term appears:

```
∂𝓛/∂Z₀ = ∂ℓ₀/∂Z₀ + (dZ_dZ)₀ᵀ·adjoint₁
```

and this one is **not set to zero**. `Z₀` is not determined by the
dynamics — it is given, or it is a variable someone may want to optimise —
so its stationarity expression is not a constraint, it is the gradient.
Compare it against the interior formula and they are identical, which is
why

```
dJ/dZ₀ = adjoint₀
```

falls out with no extra work. The asymmetry at `t = 0` is the whole reason
the initial-state gradient is simply the last thing the sweep computes.

**Controls.** `U_t` appears in `ℓ_t` and in constraint `t`, never as a
bare term, so there is nothing to set to zero and the expression is
directly what we want:

```
dJ/dU_t = ∂ℓ_t/∂U_t + (dZ_dF)ₜᵀ·adjoint_{t+1}
```

## The three ways this breaks

Every adjoint bug is one of these, and none of them changes the shape of
the output:

1. **`adjoint_t` where `adjoint_{t+1}` belongs** in the control gradient.
   Dimensionally sound, silently wrong. The implementation reads off
   `dJ_dU[t]` *before* stepping the adjoint back for exactly this reason.
2. **The Jacobian taken at the wrong state.** `(dZ_dZ)ₜ` is evaluated at
   `Z_t` — the state at the *start* of the step it describes.
3. **Off-by-one at either boundary**, from the `H+1` versus `H` counts.

Only a finite-difference check over the whole rollout catches these. A
per-step Jacobian test cannot: every per-step object is already validated,
and the bug is in how they are strung together.

## A worked example

Free flight — gravity alone, so `∂f/∂q = ∂f/∂v = 0` and the step Jacobian
is a constant shear:

```
dz_dz = ⎡ Id   dt·Id ⎤        dz_dzᵀ = ⎡ Id      0  ⎤
        ⎣ 0    Id    ⎦                 ⎣ dt·Id   Id ⎦
```

Seed a terminal-only objective that reads off the final height,
`∂ℓ_H/∂z_H = (0,1,0 | 0,0,0)ᵀ`, and sweep:

```
adjoint_H     = (0, 1, 0 | 0,    0, 0)ᵀ
adjoint_{H−1} = (0, 1, 0 | 0,   dt, 0)ᵀ
adjoint_{H−2} = (0, 1, 0 | 0,  2dt, 0)ᵀ
      ⋮
adjoint₀      = (0, 1, 0 | 0, H·dt, 0)ᵀ
```

Each step leaves the `q` block alone (top-left `Id`) and accumulates `dt`
times it into the `v` block (bottom-left `dt·Id`). So
`∂q_H/∂v₀ = H·dt·Id`, matching `(dz_dz)^H = [[Id, H·dt·Id], [0, Id]]` —
shears compose by adding their off-diagonal blocks. Physically: perturbing
the initial velocity displaces the final position by exactly the elapsed
time.

The control gradient, with `dz_dfᵀ = [dt²M⁻¹ | dt·M⁻¹]`:

```
dJ/du_t = dt²M⁻¹(0,1,0)ᵀ + dt·M⁻¹(0, (H−t−1)dt, 0)ᵀ = dt²(H−t)·M⁻¹(0,1,0)ᵀ
```

which collapses to the single-step `dz_df` at `t = H−1`. At `m = 1`,
`dt = 0.01`, `H = 100` the earliest control has **100× the influence** of
the last on final position, which is just "early pushes have longer to
act" — a cheap check that the sweep is oriented correctly.

Both closed forms are asserted in
`tests/validation/test_rollout_gradients.cpp`. Validating a full-horizon
sweep against exact algebra rather than against finite differences is
unusual, and worth taking where it is available.

## Recompute rather than tape

`step_system_jacobian` is a pure function of the state, so the backward
sweep can rebuild each step's Jacobian from the stored trajectory. Nothing
else needs recording — no operation graph, no tape.

Storage is `48·B` bytes per step for the states, against `(6B)²` doubles
per step for the Jacobians. The compute is a wash: taping makes the
forward pass evaluate Jacobians it may not need, recomputing makes the
backward pass do it, and the totals match whenever gradients are wanted.
Recomputing is strictly cheaper when they are not.

One honest caveat: `step_system` and `step_system_jacobian` each run
detection independently, so recomputing re-runs it too. That is the
per-step allocation issue already flagged at the end of
`contact_detection.md`, and it makes recompute look worse than it is.

## The objective is the caller's

GRIP does not define `J` or `ℓ`. The caller supplies `∂ℓ_t/∂Z_t` and
`∂ℓ_t/∂U_t` — the seeds — and receives `dJ/dZ₀` and `dJ/dU_t`. A
terminal-only objective is the case where every seed but the last is zero.

A cost function is a task definition, and tasks belong to the repositories
that call this one. What requires knowing how the physics works is
propagating a sensitivity backward through the dynamics, and that is
exactly what is here. It also means the sweep never needs to know whether
it is serving trajectory optimisation, policy-gradient learning, or system
identification.

## Cost, and a known inefficiency

Per step: one `(dZ_dZ)ᵀ·adjoint` at `36B²` multiply-adds, one
`(dZ_dF)ᵀ·adjoint` at `18B²`, plus one Jacobian evaluation. Total `O(H·B²)`.

But `dZ_dZ` is block-diagonal while bodies only touch static scenery, so
the true matrix-vector product is `O(B)` — 36 multiply-adds per body.
Assembling a dense `6B × 6B` and multiplying discards that. Real, and
deliberately left alone: the sparsity structure that would fix it follows
the contact graph, which does not exist until body-body contact produces
one. See `multi_body_system.md`.

## What this makes measurable

`∂Z_H/∂Z₀` is the product `∏ₜ (dZ_dZ)ₜ`, and two things follow.

The **determinant** of that product is the product of the per-step
determinants, each of which `integrator_jacobians.md` gives exactly as
`det(Id + dt·M⁻¹·∂f/∂v)`. An undamped contact keeps it at 1 over any
horizon; damping contracts it geometrically in the number of steps spent
in contact.

But determinant is *volume*, not conditioning. A symplectic map preserves
volume and can still be violently chaotic, crushing one direction while
amplifying another. What governs whether a gradient is usable is the
**singular values** of the product, and those are a separate measurement —
one this step makes possible for the first time.

That measurement is where the contact-boundary work from step 5 leads.
`test_contact_boundary.cpp` pins the per-step error exactly: across the
spring's kink a central difference returns a finite wrong number, and
across the damper's jump it diverges like `1/ε`. A single step of a
rollout that straddles an activation boundary contributes a factor that is
wrong in precisely that way. Watching how it compounds over a horizon
needs the product, and the product needs this sweep.
