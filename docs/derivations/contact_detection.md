# Contact detection — convex polygon against a static half-plane

## The obstacle

A half-plane is an infinite line with everything on one side of it kept
solid. Free space is `{ p : n·p ≥ d }`, with `n` a **unit** normal
pointing *into* free space — the direction a penetrating body must move
to escape — and `d` a scalar offset. Ground at `y = 0` is `n = (0,1)`,
`d = 0`, consistent with the y-up gravity convention in
`symplectic_euler.md`.

Three properties make this the right first obstacle, and all three are
things a finite or curved obstacle would take away:

- **No boundary of its own.** A finite ground segment has endpoints where
  the normal flips discontinuously as a body slides off. An infinite
  half-plane has one surface and one normal, everywhere.
- **Constant normal.** `n` doesn't vary with position, so `∂n/∂q = 0` and
  it drops out of every derivative below.
- **Exactly linear signed distance.** `φ = n·p − d` is a dot product. No
  square roots, no minimum over surface features, no case analysis.

The obstacle is *static scenery*, not a body: no mass, no state, no entry
in the system state vector, and no contribution to any Jacobian.

## Body geometry

A body is a convex polygon given by its vertices in the body frame,
`r_i`, fixed relative to the center of mass (`core/rigid_body.hpp`,
`BodyShape`). The world position of vertex `i` is

```
p_i(q) = c + R(θ)·r_i,      c = q.head<2>(),  R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]]
```

so vertices move both when the body translates (through `c`) and when it
rotates (through `R(θ)`). That second dependence is why contact is the
first physics in this project that reads `θ` at all: gravity and a COM
wrench are orientation-blind.

**Why a polygon rather than a disc.** A disc's contact geometry is
rotation-invariant, giving `∂φ/∂θ ≡ 0`, and its normal force points
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
φ_i = n·p_i − d
```

with `φ > 0` separated (by that distance), `φ = 0` touching, `φ < 0`
penetrating (by that depth). Signed, not absolute: a force law needs to
know which side it's on and how far past, and `φ` crossing zero is the
activation boundary the project ultimately exists to characterize.

## Why vertices alone are sufficient (not an approximation)

Only vertices are tested, but for a convex polygon against an infinite
half-plane nothing is missed between them. `φ` is affine in position and
the polygon is the convex hull of its vertices, so any interior point
`p = Σλᵢpᵢ` (with `λᵢ ≥ 0`, `Σλᵢ = 1`) has

```
φ(p) = n·(Σλᵢpᵢ) − d = Σλᵢ(n·pᵢ − d) = Σλᵢφᵢ
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
own `φ`); the NCP solver in step 6 will have to confront it directly.

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

1. **Detection stays smooth.** Each `φ_i` is linear in `p_i` with no
   branching, so `detect_contacts_body` is differentiable everywhere.
   The non-smoothness enters exactly one step later, when a force law
   decides which `φ_i < 0` are active. Putting the discontinuity at the
   activation test rather than inside detection is the right place for
   it, and keeps the boundary isolated and inspectable.
2. **Margins come free.** Distance-to-activation for a separated vertex
   is just its `φ`, with no separate query.

Contact `i` is always vertex `i` — fixed iteration order, no sorting, no
filtering — so the output is deterministic by construction.

## The Jacobian ∂φ/∂q

Differentiate `p_i(q) = c + R(θ)·r_i`. The translation part is immediate.
For the rotation part, use `dR/dθ = J·R(θ)` where `J = [[0,−1],[1,0]]` is
rotation by +90°:

```
∂p_i/∂θ = (dR/dθ)·r_i = J·R(θ)·r_i = J·(p_i − c)
```

and since `φ_i = n·p_i − d` is linear in `p_i`:

```
∂φ_i/∂q = [ n_x,  n_y,  n·(J·(p_i − c)) ]
```

`J·(p_i − c)` is the moment arm rotated 90°, which is exactly the
direction that vertex moves under unit angular velocity — so the third
component asks how much of the vertex's swing lies along the escape
direction.

Two consequences worth stating, both of which are asserted directly in
`tests/validation/test_contact_jacobians.cpp` rather than left to the
finite-difference comparison:

- The translation block is *exactly* the plane normal, for every vertex
  at any orientation, because `φ` is linear in `c`.
- `∂φ_i/∂θ = 0` precisely when the vertex is at an extremum of height —
  e.g. a square balanced on a corner has `∂φ/∂θ = 0` for both the bottom
  and top corners, while the two side corners are moving fastest, at
  `±` half the diagonal. A sign error or a dropped rotation in the
  `J·(p−c)` term would not reproduce that pattern.

## Note for later

`detect_contacts_body` allocates and returns a `std::vector` per call.
That's fine now — no benchmark exists, and CLAUDE.md is explicit about
not optimizing ahead of one — but it is a per-step heap allocation on
what will become the hot path, so it's the first thing to revisit when
rollout throughput starts being measured.
