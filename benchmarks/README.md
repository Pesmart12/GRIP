# Baseline

Throughput for GRIP 1.0 — penalty contact, single-threaded, one scene at
a time. Taken before 2.0 replaces the contact formulation, because after
that this number can only be reconstructed rather than measured.

```
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target grip_baseline_benchmark
./build-release/benchmarks/grip_baseline_benchmark
```

**Release only.** Unoptimized Eigen is slower by more than an order of
magnitude, so a debug build produces nothing worth recording. The binary
says so loudly if `NDEBUG` is absent, because the two builds look
identical once the output is pasted somewhere.

## Recorded run

Intel Core i7-14700F, MSVC 19.4x `/O2 /fp:precise`, 2026-08-18.
Penalty at `dt = 5e-4`, `k = 1e4`, `b = 50`, `b_slip = 200`, `mu = 0.5`.

| scene | active | fwd ns/step | fwd µs/step | steps/sec | s per sim s | adj µs/step |
|---|---|---|---|---|---|---|
| free flight, 1 body | 0 | 306 | 0.31 | 3,264,083 | 0.003 | 0.80 |
| resting pair, 2 bodies | 4 | 3,282 | 3.28 | 304,695 | 0.01 | 6.24 |
| stack, 10 bodies | 20 | 71,987 | 71.99 | 13,891 | 0.14 | 127.58 |

`active` counts contacts with `d < 0`, plane vertices plus pair points —
reported so the table says what it measured rather than what it intended
to. `s per sim s` is wall-clock to simulate one second. `adj µs/step` is
one backward step including the Jacobian the adjoint recomputes rather
than tapes.

## What it says

**Contact costs an order of magnitude.** One free body is 306 ns; two
bodies in resting contact are 3.3 µs. The gap is the pair path — SAT,
clipping, and the second-order jets — and it is entered per pair per
call.

**Everything is faster than real time.** A ten-box stack simulates one
second in 0.14 s of wall clock, at a timestep chosen for contact
resolution rather than stability. Penalty's cost is the *timestep*, not
the per-step work.

**The backward sweep is about 2× the forward step**, consistently across
scene sizes (2.6×, 1.9×, 1.8×). Recomputing the Jacobian rather than
taping it is not the expensive choice it might look like.

**The Python boundary will dominate small scenes.** A pybind11 round trip
is order 1 µs against a 0.3–3.3 µs step, so for the one- and two-body
scenes that a control task actually uses, crossing per step would spend
25–75% of the time in the binding. Step 10 needs to expose *n* substeps
per call, and a batch of scenes per call — not because of 3.0, but
because 1.0 is unusable at these scene sizes otherwise.

## What it deliberately does not say

Where the time goes inside a step. The four optimization questions in
CLAUDE.md — detection allocations, dense `∂F/∂Q`, the adjoint's `O(B²)`
matvec, and the redundant `ComputePairGeometry` calls — live mostly in
code that 2.0 replaces. Measuring them now would be work with a known
expiry. Re-ask after the contact solve lands.
