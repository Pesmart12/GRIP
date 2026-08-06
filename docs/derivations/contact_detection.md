# Contact detection — convex polygon against a static half-plane

Symbols follow `docs/derivations/notation.md`.

## The obstacle

A half-plane is an infinite line with everything on one side of it kept
solid. In Hesse normal form, free space is `{ p : n·p ≥ o }`, with `n` a
**unit** normal pointing *into* free space — the direction a penetrating
body must move to escape — and `o` the scalar offset from the origin
along that normal. Ground at `y = 0` is `n = (0,1)`, `o = 0`, consistent
with the y-up gravity convention in `symplectic_euler.md`.

One scalar, one degree of freedom: a half-plane's only positional
freedom is how far along its normal it sits. `‖n‖ = 1` is a genuine
precondition rather than a convention — `n·p − o` is a true distance
only for a unit normal, and a scaled one would quietly scale every
penetration depth and every contact force derived from it.

Three properties make this the right first obstacle, and all three are
things a finite or curved obstacle would take away:

- **No boundary of its own.** A finite ground segment has endpoints where
  the normal flips discontinuously as a body slides off. An infinite
  half-plane has one surface and one normal, everywhere.
- **Constant normal.** `n` doesn't vary with position, so `∂n/∂q = 0` and
  it drops out of every derivative below.
- **Exactly affine signed distance.** `d = n·p − o` is a dot product and
  a subtraction. No square roots, no minimum over surface features, no
  case analysis.

The obstacle is *static scenery*, not a body: no mass, no state, no entry
in the system state vector, and no contribution to any Jacobian.

## Body geometry

A body is a convex polygon given by its vertices in the body frame,
`rᵢ`, fixed relative to the center of mass (`core/rigid_body.hpp`,
`BodyShape`). The world position of vertex `i` is

```
pᵢ(q) = c + R(θ)·rᵢ,      c = q.head<2>(),  R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]]
```

so vertices move both when the body translates (through `c`) and when it
rotates (through `R(θ)`). That second dependence is why contact is the
first physics in this project that reads `θ` at all: gravity and a COM
wrench are orientation-blind.

**Why a polygon rather than a disc.** A disc's contact geometry is
rotation-invariant, giving `∂d/∂θ ≡ 0`, and its normal force points
through the center of mass, so it produces no torque. Both the `θ` row
and column of the eventual contact Jacobian would be structurally zero —
a finite-difference test would pass while validating nothing across a
third of the state space. A polygon makes both nonzero immediately. It
also means a box resting flat has **two** active contacts, so
multi-contact is the ordinary case from the start rather than a later
surprise.

## Signed distance

For each vertex,

```
dᵢ = n·pᵢ − o
```

with `d > 0` separated (by that distance), `d = 0` touching, `d < 0`
penetrating (by that depth). Signed, not absolute: a force law needs to
know which side it's on and how far past, and `d` crossing zero is the
activation boundary the project ultimately exists to characterize.

## Why vertices alone are sufficient (not an approximation)

Only vertices are tested, but for a convex polygon against an infinite
half-plane nothing is missed between them. `d` is affine in position and
the polygon is the convex hull of its vertices, so any interior point
`p = Σλᵢpᵢ` (with `λᵢ ≥ 0`, `Σλᵢ = 1`) has

```
d(p) = n·(Σλᵢpᵢ) − o = Σλᵢ(n·pᵢ) − o = Σλᵢ(n·pᵢ − o) = Σλᵢdᵢ
```

The gap at *any* point of the body is the same convex combination of the
vertex gaps. No point can be deeper than the deepest vertex, and an edge
midpoint is exactly the average of its two endpoints.

The flush case — a box resting flat, where a whole edge is in contact —
is physically a distributed pressure rather than two point loads. Rigid
body dynamics responds only to the resultant, though: a total force and
the position of its line of action. Two normal forces at the edge's
endpoints produce a resultant whose center of pressure sweeps the entire
edge as those two forces vary over non-negative values, so the vertex
model spans exactly the achievable resultants of the distributed load.
What it doesn't reproduce is the shape of the pressure distribution,
which affects stress and deformation but not a rigid body's trajectory.

This is also the clearest way to see the static indeterminacy of a flat
resting box: the dynamics determine the *sum* of the two corner forces
and their center of pressure, but not the individual forces. Penalty
contact resolves that by construction (each spring responds only to its
own `d`); the NCP solver in step 6 will have to confront it directly.

Completeness here depends on the obstacle being an infinite plane. It
does **not** carry over to body-body contact (step 6+), where checking
one body's vertices against the other misses the case where the second
body's vertex penetrates the first body's face with none of the first
body's vertices inside — that needs vertex-face checks in both
directions plus edge-edge handling. Nor does it cover curved shapes,
which have no vertices, or a finite ground segment, where a vertex can
hang past the end while the edge still overlaps.

## Every vertex is reported, including separated ones

`detect_contacts_body` returns one `Contact` per vertex, in vertex order,
whether or not it's penetrating. Two reasons:

1. **Detection stays smooth.** Each `dᵢ` is linear in `pᵢ` with no
   branching, so `detect_contacts_body` is differentiable everywhere.
   The non-smoothness enters exactly one step later, when a force law
   decides which `dᵢ < 0` are active. Putting the discontinuity at the
   activation test rather than inside detection is the right place for
   it, and keeps the boundary isolated and inspectable.
2. **Margins come free.** Distance-to-activation for a separated vertex
   is just its `d`, with no separate query.

Contact `i` is always vertex `i` — fixed iteration order, no sorting, no
filtering — so the output is deterministic by construction.

## The contact Jacobian J = ∂d/∂q

Differentiate `pᵢ(q) = c + R(θ)·rᵢ`. The translation part is immediate.
For the rotation part, use the perp operator `a^⊥ = (−a_y, aₓ)` (rotation
by +90°), for which `(dR/dθ)·r = (R(θ)·r)^⊥`:

```
∂pᵢ/∂θ = (dR/dθ)·rᵢ = (R(θ)·rᵢ)^⊥ = (pᵢ − c)^⊥
```

and since `dᵢ = n·pᵢ − o` is affine in `pᵢ`:

```
Jᵢ ≡ ∂dᵢ/∂q = [ n_x,  n_y,  n·(pᵢ − c)^⊥ ]
```

`(pᵢ − c)^⊥` is the moment arm rotated 90°, which is exactly the
direction that vertex moves under unit angular velocity — so the third
component asks how much of the vertex's swing lies along the escape
direction.

Two consequences worth stating, both of which are asserted directly in
`tests/validation/test_contact_jacobians.cpp` rather than left to the
finite-difference comparison:

- The translation block is *exactly* the plane normal, for every vertex
  at any orientation, because `d` is linear in `c`.
- `∂dᵢ/∂θ = 0` precisely when the vertex is at an extremum of height —
  e.g. a square balanced on a corner has `∂d/∂θ = 0` for both the bottom
  and top corners, while the two side corners are moving fastest, at
  `±` half the diagonal. A sign error or a dropped rotation in the
  `(p − c)^⊥` term would not reproduce that pattern.

`Jᵢ` will do more than measure the gap's sensitivity. In step 5 it also
maps generalized velocity to closing rate (`ḋᵢ = Jᵢ·v`) and maps normal
force back to generalized force (`f_c,i = Jᵢᵀλᵢ`) — the virtual-work
duality that makes the contact Jacobian the central object of every
contact formulation.

## Note for later

`detect_contacts_body` allocates and returns a `std::vector` per call.
That's fine now — no benchmark exists, and CLAUDE.md is explicit about
not optimizing ahead of one — but it is a per-step heap allocation on
what will become the hot path, so it's the first thing to revisit when
rollout throughput starts being measured.
