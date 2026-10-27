#pragma once

#include <cstddef>
#include <vector>

#include "api/scene.hpp"

namespace grip {

// ===========================================================================
// Batched entry points.
//
// Two things are batched here, for two different reasons, and they are
// independent:
//
//   substeps      integration steps per control step. A policy runs at a
//                 fixed control rate while penalty contact needs a much
//                 finer integration step, so one call advances many. The
//                 baseline benchmark is what forced this: a step costs
//                 0.3-3.3 us against a language-boundary crossing of order
//                 1 us, so crossing per integration step would spend a
//                 quarter to three quarters of the runtime in the binding.
//
//   environments  independent scenes advanced together, in parallel under
//                 OpenMP when the build has it (GRIP_USE_OPENMP, on by
//                 default; OMP_NUM_THREADS sets the thread count). Each
//                 environment is computed start to finish by one thread, so
//                 there is no cross-environment accumulation to reorder and
//                 the result is bit-identical for any thread count, serial
//                 included.
//
// No new physics. Every one of these is a loop over step_system and
// step_system_jacobian, which remain the single implementation the rollout
// and gradient paths share.
// ===========================================================================

// Advance every environment by `substeps` integration steps under one held
// control wrench.
//
// In place, because the caller holds a state buffer it steps repeatedly and
// returning a fresh one would allocate per call. The state is still an
// explicit argument rather than anything ambient -- Scene stays pure
// configuration.
//
// `controls` carries a single control step, so `controls.steps` must be 1.
void step_batch(const std::vector<Scene>& scenes, StateBatch& state, const ControlBatch& controls, std::size_t substeps);


// Roll out every environment for controls.steps control steps, recording one
// state per control step plus the initial one.
//
// `trajectory` is resized to fit. Sized (steps + 1, environments, bodies, 6),
// which is the layout numpy wraps without copying.
void rollout_batch(const std::vector<Scene>& scenes, const StateBatch& initial, const ControlBatch& controls, std::size_t substeps, TrajectoryBatch& trajectory);


// Total derivatives of a caller-defined objective, batched.
//
// dobjective_dZ0 is shaped like a StateBatch, dobjective_dU like the ControlBatch that
// produced the trajectory -- so a learner reads gradients in exactly the
// arrays it supplied controls in.
struct RolloutGradientBatch {
  StateBatch dobjective_dZ0;
  ControlBatch dobjective_dU;
};


// Reverse-mode sweep over a trajectory produced by rollout_batch.
//
// GRIP does not define the objective. The caller supplies its partials --
// dl_dZ shaped like the trajectory, dl_dU shaped like the controls -- and
// receives total derivatives. A terminal-only objective is the case where
// every dl_dZ entry but the last is zero.
//
// The recursion is adjoint_system's (gradient/rollout.hpp), called once per
// environment -- including how substeps accumulate into one control
// gradient, and how each control step is re-run forward to rebuild the
// states inside it. This function only converts between the batched arrays
// and the per-environment form. Likewise step_batch and rollout_batch call
// rollout_system, so substeps are composed in exactly one place.
RolloutGradientBatch adjoint_batch(const std::vector<Scene>& scenes, const TrajectoryBatch& trajectory, const ControlBatch& controls, std::size_t substeps, const TrajectoryBatch& dl_dZ, const ControlBatch& dl_dU);

}  // namespace grip
