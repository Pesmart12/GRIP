# Body-body contact detection — polygon against polygon

Symbols follow `docs/derivations/notation.md`. This is step 9a: geometry
and its derivatives, with no force law attached.

## Why the half-plane derivation does not carry over

`contact_detection.md` gets three properties from a half-plane, and
polygon-polygon has none of them:

- **No boundary of its own.** A polygon face is a *segment*, so a contact
  can slide off its end.
- **Constant normal.** `∂n/∂q = 0` dropped out of every derivative there.
  Here the normal rotates with whichever body owns the face, and *which*
  body that is changes discontinuously.
- **Exactly affine signed distance.** `d = n·p − o` needed no case
  analysis. Here the normal must be searched for before `d` means
  anything.

That file also states outright that its completeness argument — vertices
alone suffice — does not extend to body-body, because one body's vertex
can penetrate the other's face with none of the first body's vertices
inside. A new argument is needed, and the separating axis theorem
supplies it.

## Finding the normal

Two convex sets are disjoint iff some axis separates them, and for convex
polygons the face normals of both bodies are a *sufficient* set of
candidate axes. So: for each face of each body, project the other body
onto that normal and record how far the nearest vertex sits outside.

- Any face with positive separation proves disjointness, and the search
  stops.
- Otherwise every face is penetrated, and the **least**-penetrated one is
  the shortest way out. That is the contact normal.

Exact, non-iterative, `O(n·m)`. GJK/EPA would generalize to arbitrary
convex shapes we do not have, at the price of putting an iterative solver
inside detection — the output would be a solver result rather than a
closed form, which is a different thing to differentiate.

## One normal is not a contact

SAT returns a single normal and depth. But two boxes resting face to face
touch along a *segment*, and a single contact point cannot resist
tipping — exactly the argument that made a flat box on the ground need
two contacts.

So the reference face (the winning one) is paired with the **incident
face**: the face of the other body most nearly anti-parallel to the
normal. The incident edge is then clipped to the reference face's extent.

Both clip constraints are linear in the incident edge's own parameter
`t`, so clipping is an interval intersection rather than a geometric
construction:

```
tangent·(i₀ + t·e − r₀) ≥ 0        and        tangent·(i₀ + t·e − r₁) ≤ 0
```

with `e = i₁ − i₀`. Intersecting with `t ∈ [0, 1]` leaves one or two
surviving ends.

**Clipped ends are kept even when their gap is positive.** Clipping
bounds the contact *along* the face, not across it, so the far end of a
tilted incident edge can sit outside the reference plane. Dropping those
would make the contact set change size as a body rocks, adding a
discontinuity for nothing — and the force law gates on `d < 0` anyway.
Same reasoning as `detect_contacts_body` reporting separated vertices.

## The Jacobian

Write the gap as `d = n·(p − r₀)` with `n` the reference face's outward
normal and `r₀` a material point on it. Three shapes appear, and only the
first is familiar:

```
n·(p − c_j)^⊥            the incident body sweeping the contact point —
                         exactly the half-plane form

n^⊥·(p − r₀)             the reference body swinging the face, absent
                         against a half-plane because a fixed plane has
                         ∂n/∂q = 0

(n·e)·∂t/∂q              the contact point sliding along the incident
                         edge as the clip moves — clipped contacts only
```

The second exists because `∂n/∂θ_i = (R n̂)^⊥ = n^⊥`. The full blocks,
for a contact between body `i` (owning the face) and body `j` (owning the
incident edge):

```
∂d/∂c_i = −n                              ∂d/∂c_j = +n
∂d/∂θ_i = n^⊥·(p − r₀) − n·(r₀ − c_i)^⊥   ∂d/∂θ_j = n·(p − c_j)^⊥
```

plus `(n·e)·∂t/∂·` on every block when the contact is clipped.

### Why the choice of `r₀` does not matter

Any material point on the reference face gives the same gap, so it must
give the same derivative. It does, and the cancellation is worth seeing.
For two choices differing by a tangential `Δ = α·n^⊥`:

```
difference = n^⊥·Δ + n·Δ^⊥ = α + (−α) = 0
```

using `(n^⊥)^⊥ = −n`. The two new terms cancel exactly against each
other, which is a decent check that both are right.

### Clipped contacts

A clip point is not a material point of either body — it lies on the
incident edge *and* on a reference side plane, so it moves with both.
With `a = tangent·(i₀ − r_k)` and `b = tangent·e`, the parameter is
`t = −a/b` and

```
∂t/∂q = −( ∂a/∂q + t·∂b/∂q ) / b
```

Every block then picks up `(n·e)·∂t/∂q`. Note the factor: **it vanishes
when the faces are anti-parallel**, since then `e` lies in the reference
face and `n·e = 0`. That is precisely why treating a clip point as a
material vertex looks nearly right — it is exactly right in the resting
configuration and wrong everywhere else.

Throughout all of it the translation blocks stay exact negations of one
another, so **Newton's third law survives clipping**: the generalized
force `Jᵀλ` splits into equal-and-opposite halves without the signs ever
being written down. Same dividend `Jᵀ` paid for the moment arm in step 5.

## The Hessian, and why it is composed rather than assembled

The contact force is `f_c = Jᵀλ`, and *both* factors depend on `q`. So

```
∂f_c/∂q = Jᵀ(∂λ/∂q) + λ(∂Jᵀ/∂q)
```

and since `J = ∂d/∂q`, that second factor is `∂²d/∂q²` — the **Hessian of
the gap**, `6×6` and symmetric. With `ḋ = Jv` giving `∂ḋ/∂q = (Hv)ᵀ`,
everything collapses to

```
∂f_c/∂q = −k·JᵀJ  −  b·Jᵀ(Hv)ᵀ  +  λ·H
∂f_c/∂v = −b·JᵀJ
```

which is exactly the half-plane formula from `penalty_contact.md`, with
`H` standing in for its single nonzero entry. Against a half-plane the
normal was fixed and the contact point material, so `H` had one term.
Here every component of `J` moves.

The velocity block needs no Hessian at all, which is why it was never the
hard part.

### Composed, not hand-assembled

Twenty-one independent Hessian entries through a composition involving a
rotating normal, a material vertex, and a clip parameter that is itself a
ratio of dot products is possible to hand-derive and very hard to trust.
Instead each primitive carries its value, gradient and Hessian together,
and the three product rules

```
dot(a, b)      H = Σₖ [ bₖH_{aₖ} + aₖH_{bₖ} + ∇aₖ⊗∇bₖ + ∇bₖ⊗∇aₖ ]
scale(v, s)    H_k = sH_{vₖ} + vₖH_s + ∇s⊗∇vₖ + ∇vₖ⊗∇s
divide(u, w)   H = [ H_u − fH_w − ∇f⊗∇w − ∇w⊗∇f ] / w,  f = u/w
```

compose them. This is exact closed-form differentiation — no step size,
no truncation — not numerical differencing.

Two things fall out of doing it this way. The **gradient comes free**, and
agrees with the Jacobian derived independently by hand, which is a
genuine cross-check on both. And **material vertices and clipped points
become the same code**: the edge parameter is either a constant jet or a
ratio jet, and everything downstream is identical. The `∂²t/∂q²` terms
that would otherwise need their own derivation are produced by the
quotient rule.

The primitives are few. A vector that rotates rigidly with one body — a
normal, a tangent, an edge — has `∂v/∂θ = v^⊥` and `∂²v/∂θ² = −v`. A
point material in one body follows its translation and sweeps
`(p − c)^⊥`, with `∂²p/∂θ² = −(p − c)`. Everything else is composition.

### Checks worth having

- **Symmetry.** Every entry is produced independently by the composition,
  so symmetry is a real check on the product rules rather than a
  tautology of the storage.
- **Translation-translation blocks vanish**, all four of them, because
  the gap is affine in either body's position. The half-plane asserts the
  same property as its translation block being exactly the normal.
- **Finite differences of the analytic Jacobian**, which is the only
  independent handle on a second derivative. Differencing a first
  derivative costs roughly half the available precision, so the tolerance
  is looser than the Jacobian tests' — still six digits on all
  thirty-six entries.

## Friction between bodies

Slip at a pair contact is the relative tangential velocity of the two
material points currently coincident there, taken second-relative-to-first
to match the normal's orientation:

```
s = n^⊥ · [ (v₂ + ω₂·(p−c₂)^⊥) − (v₁ + ω₁·(p−c₁)^⊥) ]
```

which is linear in `[v₁; v₂]`, so

```
J_perp = [ −n^⊥, −n^⊥·(p−c₁)^⊥ | n^⊥, n^⊥·(p−c₂)^⊥ ]
```

**Simpler than the gap Jacobian**, and worth knowing why: there is no
`n^⊥·(p − r₀)` term. The gap Jacobian has one because it is a gradient
and the normal rotates with the reference body. This is not a gradient —
without a stick anchor there is no tangential gap — so the normal enters
as a coefficient rather than through a derivative. Setting body 1 to
fixed scenery recovers the half-plane form exactly.

The force law is unchanged from `penalty_contact.md`: the same clamped
Kelvin–Voigt normal force, the same Coulomb cone `|β| ≤ μλ`. Each
manifold point carries **its own cone**, bounded by its own `λ`, so a
resting box can stick at one corner while sliding at the other.

### The gradient, and a surprise

The rotating normal reappears one level up, in `∂J_perp/∂q`, which is
what `∂f/∂q` needs. It picks up `∂n^⊥/∂θ_ref = −n` along with the contact
point's motion, clip dependence included.

That object was expected to be a plain non-symmetric `6×6` — `J_perp` is
not a gradient, so nothing required symmetry. **It is symmetric anyway.**
The cross terms match through the perp identity `n^⊥·a = −(n·a^⊥)`:

```
∂J_perp[5]/∂θ₁ = −n·(p−c₂)^⊥            ∂J_perp[2]/∂θ₂ = n^⊥·(p−c₂)
```

the same number written two ways. So `J_perp` is curl-free, and therefore
*is* the gradient of something — a quantity this model never names, since
sliding accumulates no position. Recorded because it was a surprise, and
because it makes symmetry a genuine check on the composition rather than
a property of how the result is stored.

It keeps its own name, `PairSlipGradient`, rather than reusing
`PairHessian`. Both are symmetric, for unrelated reasons, and only one of
them is a second derivative.

### Assembly

Writing `G = ∂J_perp/∂q`, the friction contribution is

```
∂f/∂q = J_perpᵀ(∂β/∂q) + β·G
∂f/∂v = J_perpᵀ(∂β/∂v)
```

with `∂β` taking the two branches of `penalty_contact.md`. Sticking gives
`∂β/∂q = −b_slip·vᵀG` and `∂β/∂v = −b_slip·J_perp`. Sliding pins `β` to
`∓μλ`, so its whole state dependence runs through the **normal** force
and reuses the gap Hessian already computed.

## Three new non-smooth surfaces

Body-body contact adds more than the force law did, and one of them is a
class the model did not previously contain.

**The reference flips.** Which body owns the face changes discontinuously.
The gap is continuous across the flip — both faces measure the same
distance — but the Jacobian is not, because the `n^⊥·(p − r₀)` term
belongs to whichever body owns the face.

Worse, the commonest resting configuration sits exactly on it. When two
faces are parallel and overlapping, both give *exactly* the same
penetration, so the ownership is a tie. Tilting either way hands it to
the same body, so exact parallel is not a crossing between two regimes —
it is an **isolated point whose tie-break disagrees with its entire
neighbourhood**. The analytic Jacobian there is not the limit from either
side, and finite-differencing at squarely stacked boxes compares two
different formulas. `test_pair_detection.cpp` pins this, and the
validation configurations deliberately tilt clear of it.

**The manifold changes size**, two points to one, as a body tips onto a
corner. Contact forces redistribute discontinuously.

**The normal itself jumps** when the winning axis changes between
non-parallel faces — by the angle between them.

That last one is a genuinely new severity class. The damper's jump at
step 5b is proportional to closing speed, so it vanishes for grazing
contact and the pathology at least concentrates predictably where impacts
are fast. A normal flip's jump is proportional to *penetration depth* and
does not vanish as any velocity goes to zero. It is the first
discontinuity in the model not controlled by something going to zero.

## What is out of scope

- **Broad phase.** Every pair is tested, `O(B²)`. Fine at these scales;
  spatial partitioning is an optimization awaiting a benchmark.
- **Continuous collision detection.** A fast thin body can pass through
  another between steps. Penalty contact has no CCD.
- **Non-convex shapes.** SAT requires convexity, as does the whole
  argument above.
