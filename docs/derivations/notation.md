# Notation

Canonical symbols for the whole project. Every other file in
`docs/derivations/` uses these and nothing else. **No symbol means two
things.** If a new quantity needs a letter, add it here first.

## State and dynamics

| symbol | meaning | shape | code |
|---|---|---|---|
| `q` | configuration `(x, y, θ)` | 3-vector | `RigidBodyState::q` |
| `v` | generalized velocity `(vₓ, v_y, ω)` | 3-vector | `RigidBodyState::v` |
| `z` | stacked state `(q, v)` | 6-vector | `StateVector` |
| `Z` | stacked system state, body `i` at `[6i, 6i+6)` | 6B-vector | `SystemStateVector` |
| `u` | control wrench `(fₓ, f_y, τ)` at the COM | 3-vector | `u` |
| `f` | total generalized force | 3-vector | — |
| `M` | mass matrix `diag(m, m, I)` | 3×3 | — |
| `m` | mass | scalar | `RigidBodyParams::mass` |
| `I` | moment of inertia about the COM | scalar | `RigidBodyParams::inertia` |
| `g` | gravitational acceleration | scalar | `kDefaultGravity` |
| `dt` | timestep | scalar | `dt` |
| `B` | number of bodies | scalar | — |
| `T` | kinetic energy | scalar | — |
| `Id` | identity matrix | square | `Matrix3d::Identity()` |

`z` rather than `x` for the stacked state: `q = (x, y, θ)` already uses
`x` for a scalar coordinate, and `∂x/∂x` meaning "6×6 state Jacobian"
next to `x` meaning "horizontal position" is a genuine misreading.
`z = (q, v)` is the standard control-theory spelling and removes the
clash. Jacobians therefore read `∂z_{t+1}/∂z_t`, in code `dz_dz`.

`Id` rather than `I` for the identity, because `I` is the moment of
inertia in `M = diag(m, m, I)`, which has no comfortable alternative
spelling.

`B` rather than `N` for the body count, so `n` unambiguously means the
contact normal.

## Geometry

| symbol | meaning | shape | code |
|---|---|---|---|
| `c` | body center of mass, world | 2-vector | `q.head<2>()` |
| `θ` | body orientation | scalar | `q.z()` |
| `R(θ)` | body→world rotation | 2×2 | `Rotation2Dd` |
| `rᵢ` | vertex `i` in the body frame | 2-vector | `BodyShape::vertices[i]` |
| `pᵢ` | vertex `i` in the world, `= c + R(θ)rᵢ` | 2-vector | `Contact::point` |
| `n` | half-plane normal, unit, into free space | 2-vector | `HalfPlane::normal` |
| `o` | half-plane offset from the origin along `n` | scalar | `HalfPlane::offset` |
| `a^⊥` | perp operator, `(−a_y, aₓ)`; `a` is a placeholder, not a quantity | 2-vector | `Perp(a)` |
| `ρ` | body-frame offset of a reference point from the COM | 2-vector | — |

The perp operator replaces what an earlier draft called `J` (the 90°
rotation matrix `[[0,−1],[1,0]]`). That collided head-on with `J` for
the contact Jacobian — the two appeared in the *same* expression. `a^⊥`
needs no symbol of its own and says what it does, leaving `J` to mean
Jacobian everywhere.

`o` is a **scalar**, not a point. An intermediate draft defined the plane
by a point `p₀` on its boundary, which collided with vertex 0 (vertices
being `pᵢ`) in the one expression where it mattered most. The scalar
offset avoids that, and is better anyway: a half-plane has exactly one
positional degree of freedom, so storing a point would be redundant —
two different points describe the same plane. This is the standard Hesse
normal form.

`ρ` is `0` everywhere in this project — the configuration's translational
part *is* the center of mass, and `BodyShape::vertices` are stored relative
to it. It has a symbol only because `symplectic_euler.md` needs to write the
counterfactual `ρ ≠ 0` mass matrix to show what the COM choice buys: the
off-diagonal coupling block, the `θ` dependence, and the parallel-axis term
that all vanish at `ρ = 0`. Deliberately not `r`, which is taken by
`rᵢ` (vertex `i` in the body frame) — `r₀` would collide with vertex 0 the
same way the discarded `p₀` did.

## Contact

| symbol | meaning | shape | code |
|---|---|---|---|
| `dᵢ` | signed distance (gap) at vertex `i` | scalar | `Contact::signed_distance` |
| `ḋᵢ` | closing rate, `= Jᵢ·v` | scalar | — |
| `Jᵢ` | contact Jacobian `∂dᵢ/∂q` | 1×3 row | — |
| `J_perp,ᵢ` | slip Jacobian, `sᵢ = J_perp,ᵢ·v` | 1×3 row | `perp_jacobian` |
| `sᵢ` | slip rate at contact `i`, signed along `n^⊥` | scalar | — |
| `λᵢ` | normal force magnitude at contact `i` | scalar | — |
| `βᵢ` | tangential force magnitude at contact `i`, signed | scalar | — |
| `k`, `b` | contact stiffness, normal damping | scalars | `PenaltyParams` |
| `b_slip` | slip damping | scalar | `PenaltyParams::slip_damping` |
| `μ` | Coulomb friction coefficient | scalar | `PenaltyParams::friction` |
| `U` | penalty potential, `= Σᵢ (k/2)·min(0, dᵢ)²` | scalar | — |
| `J_A` | active contact Jacobians, stacked one row per active contact | a×3 | — |
| `Delassus` | inverse effective mass at the contacts, `= J_A M⁻¹ J_Aᵀ` | a×a | `delassus` |
| `eig_max(·)` | largest eigenvalue | operator | — |

`d` rather than `φ` for the gap. `φ` is the contact-dynamics convention
(Stewart–Trinkle, Anitescu, Dojo), but this project uses `θ` for body
orientation, and `θ`/`φ` is *the* canonical angle pair — pairing them
while making one emphatically not an angle is the worst available
choice. `d` reads as "distance" with no gloss. It is free because the
half-plane's offset is called `o`:

```
free space = { p : n·p ≥ o }
dᵢ = n·pᵢ − o
```

Both formulas assume `‖n‖ = 1`. With a scaled normal `d` is still
correctly signed and still zero on the boundary, but it is no longer a
distance — every penetration depth, and therefore every contact force,
is silently scaled with it.

`λ` for the normal force magnitude, not `N`: `N` differs from `n`
(normal direction) only by case, and the two always appear together as
`λn`. `β` is its tangential counterpart, and is signed rather than
non-negative — friction pushes either way along the surface.

**The tangential direction still has no symbol.** It is written `n^⊥`
wherever it appears, per the reservation on `t`. `J_perp` needs a name
only because it appears in every line of a derivation; `J^⊥` was
rejected because `V^⊥` conventionally means an orthogonal complement,
which is a real ambiguity for something that is itself a row vector.

Sign convention, worth pinning because `−n^⊥` would have been equally
defensible: `n^⊥ = (−n_y, n_x)`, so for the default ground plane
`n = (0, 1)` the surface direction points in **−x**, and a body sliding
in `+x` has *negative* `s`. Self-consistent, but it reads backwards, so
`test_contact_jacobians.cpp` pins it directly rather than leaving it to
be rediscovered.

`s` is the signed *rate* along `n^⊥`; `J_perp` is the row that produces
it from `v`. One is a speed, the other a map — they share a word only
because both are about slip.

Nothing is called `W`. "Wrench" is a useful word for a
force-plus-torque 3-vector, but `W` is conventionally *work*, and the
duality argument for `Jᵀ` is itself a virtual-work argument. Contact
contributes `f_c` to the same `f` the integrator already sums.

**`Delassus` is spelled out rather than lettered**, and so is
`eig_max(·)`. Every short candidate collides. The contact literature
variously uses `W` (banned above), `G`, `D`, `A`, or `Λ` — but `G`/`g`
duplicates gravitational acceleration, `D`/`d` duplicates the gap, and
`Λ`/`λ` duplicates the normal force magnitude, all by the same
case-difference rule that already ruled out `N` for normal force.
`eig_max` exists for the same reason: `λ_max` is the natural spelling for
a largest eigenvalue everywhere else in mathematics, and here it reads as
"the largest contact force." A name that says what it is beats a letter
you have to look up — the same reasoning that gave `a^⊥` its spelling.
The subscript `A` in `J_A` marks the *active* contacts (`a` of them),
since the Delassus operator is assembled only over contacts that are
actually carrying force.

## Gradients

| symbol | meaning | shape | code |
|---|---|---|---|
| `H` | rollout horizon, in steps | scalar | `horizon` |
| `adjoint` | costate, `= dJ/dZ_t` | 6B-vector | `adjoint` |
| `J` | the caller's scalar objective | scalar | — |
| `ℓ_t` | the caller's stage cost at step `t` | scalar | — |

`H` for the horizon, not `T`: `T` is kinetic energy in
`symplectic_euler.md`, which is where the mass matrix comes from. `N` is
out for the same reason it was rejected as the body count — it differs
from `n`, the contact normal, only by case.

`adjoint` is spelled out rather than lettered. Optimal control calls it
`λ` or `p`, and both are taken here — `λᵢ` is the normal force magnitude,
`pᵢ` is a world-frame vertex — while `μ` is reserved for friction. Same
resolution as `Delassus`: when every short candidate collides, use the
name.

`J` and `ℓ` are the **consumer's**, not GRIP's. They appear in
`adjoint.md` only as scaffolding, to show where the seeds `∂ℓ/∂Z` and
`∂ℓ/∂U` come from. The API never sees a cost function — a cost is a task
definition, and tasks live in the repositories that call this one.

## Dots and subscripts

A dot is a **time** derivative: `ḋ = dd/dt`. Distinguish it from the
configuration derivative `∂d/∂q`; they're linked by the chain rule,
`ḋ = (∂d/∂q)·q̇ = J·v`, since `q̇ = v`.

Subscript `i` indexes contacts (equivalently vertices — contact `i` is
always vertex `i`). Subscript `t` indexes timesteps. A bare symbol
where the indexed one exists means the sum or stack over the index:
`f_c = Σᵢ f_c,i`.

## Reserved

Letters deliberately left unspent, because a known upcoming quantity has
a strong claim on them:

| symbol | reserved for |
|---|---|
| `t` | time (never a tangential direction — use `n^⊥`) |

`s` was the reason the stacked state is `z` and not `s`, despite `s`
being the more mnemonic choice for "state": friction's slip rate had the
better claim, and `t` and `v` were already taken by time and velocity, so
slip had nowhere else obvious to go. It is now spent, on exactly that.

The tangential direction needs no symbol at all — in 2D it is just
`n^⊥`, written out wherever it appears.

## Code correspondence

Math notation is for `docs/derivations/`. Code uses descriptive
identifiers — `signed_distance`, not `d`; `normal_force_magnitude`, not
`lambda`. The exceptions are the Jacobian member names (`dz_dz`,
`dz_du`, `dZ_dZ`, `dZ_dU`), where matching the math directly is clearer
than any prose spelling.
