#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "contact/half_plane.hpp"
#include "contact/penalty.hpp"
#include "core/rigid_body.hpp"
#include "dynamics/forces.hpp"

namespace grip {

// ===========================================================================
// The public scene description.
//
// This layer exists because a consumer calls the library across a language
// boundary, where the C++ habit of threading params, shapes, plane, penalty,
// dt and gravity through every signature stops being free -- it means
// re-converting a vector of polygons out of Python lists on every call.
//
// Nothing here is new physics. It is the existing per-scene functions viewed
// as something an outside caller can hold onto.
// ===========================================================================

// Everything about one environment that does not change while it is stepped.
//
// Deliberately *configuration*, not a simulator: state is passed in and
// returned out, never held. That preserves the statelessness the assembly
// layer already has, and it is what makes environments in a batch trivially
// independent -- and therefore deterministic whatever order they are stepped
// in, including in parallel.
//
// Joints arrive in 2.0 as one more field on this struct. That is the whole of
// the forward hedge: adding a field is a recompile, where changing a call
// signature would be a rebinding.
struct Scene {
  std::vector<RigidBodyParams> params;
  std::vector<BodyShape> shapes;
  HalfPlane plane;
  PenaltyParams penalty;
  double dt = 1.0e-3;
  double gravity = kDefaultGravity;
};


// Values per body in the packed layout below: (x, y, theta, vx, vy, omega).
inline constexpr std::size_t kStateValuesPerBody = 6;


// A batch of environment states, laid out contiguously as
// (environments, bodies, 6) in row-major order.
//
// The last axis is exactly Pack's ordering from core/rigid_body.hpp, and one
// environment's (bodies, 6) plane is exactly PackSystem's concatenation --
// body i at [6i, 6i+6). No new stacking convention is introduced here; this
// is the existing one viewed as an array, which is what lets numpy reshape
// into it rather than translate it.
//
// Contiguous, and that is load-bearing rather than tidy. A
// std::vector<std::vector<RigidBodyState>> cannot be handed to numpy without
// copying, and the trajectories this carries are tens of megabytes at the
// horizons a first-order learner uses.
//
// Rectangular, so every environment in a batch must carry the same body
// count. Nothing else has to match: shapes, masses, plane and contact
// parameters all live in the per-environment Scene and may differ freely,
// which is what lets a caller randomize geometry across a batch.
struct StateBatch {
  std::size_t environments = 0;
  std::size_t bodies = 0;
  std::vector<double> values;
};


// Index of the first value belonging to (environment, body). The one place
// the array layout above is encoded, for the same reason Pack/Unpack are the
// one place the 6-vector layout is.
inline std::size_t state_batch_offset(const StateBatch& batch, std::size_t environment, std::size_t body) {
  return kStateValuesPerBody * (batch.bodies * environment + body);
}


inline StateBatch make_state_batch(std::size_t environments, std::size_t bodies) {
  StateBatch batch;
  batch.environments = environments;
  batch.bodies = bodies;
  batch.values.assign(environments * bodies * kStateValuesPerBody, 0.0);
  return batch;
}


// One environment's states, in the form the physics reads.
//
// Allocates, and is meant to be called once per environment per API call
// rather than once per substep: the batched step converts in, runs every
// substep in this representation, and converts back out, so the cost is
// amortized over the substep count instead of paid per integration step.
inline std::vector<RigidBodyState> read_environment_state(const StateBatch& batch, std::size_t environment) {
  std::vector<RigidBodyState> states(batch.bodies);
  for (std::size_t body = 0; body < batch.bodies; ++body) {
    const std::size_t offset = state_batch_offset(batch, environment, body);
    states[body].q = Eigen::Vector3d(batch.values[offset + 0], batch.values[offset + 1], batch.values[offset + 2]);
    states[body].v = Eigen::Vector3d(batch.values[offset + 3], batch.values[offset + 4], batch.values[offset + 5]);
  }
  return states;
}


inline void write_environment_state(StateBatch& batch, std::size_t environment, const std::vector<RigidBodyState>& states) {
  for (std::size_t body = 0; body < batch.bodies; ++body) {
    const std::size_t offset = state_batch_offset(batch, environment, body);
    batch.values[offset + 0] = states[body].q.x();
    batch.values[offset + 1] = states[body].q.y();
    batch.values[offset + 2] = states[body].q.z();
    batch.values[offset + 3] = states[body].v.x();
    batch.values[offset + 4] = states[body].v.y();
    batch.values[offset + 5] = states[body].v.z();
  }
}


// Values per body in a control wrench: (fx, fy, tau).
inline constexpr std::size_t kControlValuesPerBody = 3;


// Control wrenches for a batched rollout, laid out as
// (steps, environments, bodies, 3) row-major.
//
// Indexed by *control* step, not by integration step. One entry is held
// constant across however many substeps the caller asks for -- a zero-order
// hold, which is what a policy running at a fixed control rate above a finer
// integration rate actually does. The substep count is where penalty
// contact's small timestep goes, and keeping it out of this array is what
// stops a caller from having to materialize twenty identical wrenches.
struct ControlBatch {
  std::size_t steps = 0;
  std::size_t environments = 0;
  std::size_t bodies = 0;
  std::vector<double> values;
};


inline std::size_t control_batch_offset(const ControlBatch& batch, std::size_t step, std::size_t environment, std::size_t body) {
  return kControlValuesPerBody * ((step * batch.environments + environment) * batch.bodies + body);
}


inline ControlBatch make_control_batch(std::size_t steps, std::size_t environments, std::size_t bodies) {
  ControlBatch batch;
  batch.steps = steps;
  batch.environments = environments;
  batch.bodies = bodies;
  batch.values.assign(steps * environments * bodies * kControlValuesPerBody, 0.0);
  return batch;
}


inline std::vector<Eigen::Vector3d> read_environment_control(const ControlBatch& batch, std::size_t step, std::size_t environment) {
  std::vector<Eigen::Vector3d> controls(batch.bodies);
  for (std::size_t body = 0; body < batch.bodies; ++body) {
    const std::size_t offset = control_batch_offset(batch, step, environment, body);
    controls[body] = Eigen::Vector3d(batch.values[offset + 0], batch.values[offset + 1], batch.values[offset + 2]);
  }
  return controls;
}


inline void write_environment_control(ControlBatch& batch, std::size_t step, std::size_t environment, const std::vector<Eigen::Vector3d>& controls) {
  for (std::size_t body = 0; body < batch.bodies; ++body) {
    const std::size_t offset = control_batch_offset(batch, step, environment, body);
    batch.values[offset + 0] = controls[body].x();
    batch.values[offset + 1] = controls[body].y();
    batch.values[offset + 2] = controls[body].z();
  }
}


// States over a rollout, laid out as (steps + 1, environments, bodies, 6).
//
// One state per *control* step, plus the initial one -- the same indexing as
// ControlBatch, so a learner computing rewards at control frequency reads
// this without striding. States between control steps are not recorded: the
// backward sweep rebuilds them by re-running the substeps, which is the same
// recompute-rather-than-tape trade docs/derivations/adjoint.md already makes
// one level up, and it keeps this buffer smaller by the substep count.
//
// This is the object contiguity was for. At the horizons a first-order
// learner uses it is tens of megabytes, and a vector of vectors of
// RigidBodyState would have to be copied to reach numpy.
struct TrajectoryBatch {
  std::size_t steps = 0;
  std::size_t environments = 0;
  std::size_t bodies = 0;
  std::vector<double> values;
};


inline std::size_t trajectory_batch_offset(const TrajectoryBatch& batch, std::size_t step, std::size_t environment, std::size_t body) {
  return kStateValuesPerBody * ((step * batch.environments + environment) * batch.bodies + body);
}


inline TrajectoryBatch make_trajectory_batch(std::size_t steps, std::size_t environments, std::size_t bodies) {
  TrajectoryBatch batch;
  batch.steps = steps;
  batch.environments = environments;
  batch.bodies = bodies;
  batch.values.assign((steps + 1) * environments * bodies * kStateValuesPerBody, 0.0);
  return batch;
}


inline std::vector<RigidBodyState> read_trajectory_state(const TrajectoryBatch& batch, std::size_t step, std::size_t environment) {
  std::vector<RigidBodyState> states(batch.bodies);
  for (std::size_t body = 0; body < batch.bodies; ++body) {
    const std::size_t offset = trajectory_batch_offset(batch, step, environment, body);
    states[body].q = Eigen::Vector3d(batch.values[offset + 0], batch.values[offset + 1], batch.values[offset + 2]);
    states[body].v = Eigen::Vector3d(batch.values[offset + 3], batch.values[offset + 4], batch.values[offset + 5]);
  }
  return states;
}


inline void write_trajectory_state(TrajectoryBatch& batch, std::size_t step, std::size_t environment, const std::vector<RigidBodyState>& states) {
  for (std::size_t body = 0; body < batch.bodies; ++body) {
    const std::size_t offset = trajectory_batch_offset(batch, step, environment, body);
    batch.values[offset + 0] = states[body].q.x();
    batch.values[offset + 1] = states[body].q.y();
    batch.values[offset + 2] = states[body].q.z();
    batch.values[offset + 3] = states[body].v.x();
    batch.values[offset + 4] = states[body].v.y();
    batch.values[offset + 5] = states[body].v.z();
  }
}


// Throws rather than asserts, which is a deliberate exception to the project's
// no-exceptions habit and worth being explicit about.
//
// An assert compiles away under NDEBUG, and the whole point of this layer is
// that the caller is on the far side of a language boundary -- a Python caller
// passing a mismatched array would get undefined behaviour in a release build
// rather than an error. Validation runs once per API call, not per integration
// step, so this is not the hot loop the rule is about. pybind11 translates
// std::invalid_argument into a Python exception with no adapter.
inline void validate_scene_batch(const std::vector<Scene>& scenes, const StateBatch& batch) {
  if (scenes.size() != batch.environments) {
    throw std::invalid_argument("scene count " + std::to_string(scenes.size()) + " does not match state batch environment count " + std::to_string(batch.environments));
  }
  if (batch.values.size() != batch.environments * batch.bodies * kStateValuesPerBody) {
    throw std::invalid_argument("state batch holds " + std::to_string(batch.values.size()) + " values, expected " + std::to_string(batch.environments * batch.bodies * kStateValuesPerBody));
  }
  for (std::size_t environment = 0; environment < scenes.size(); ++environment) {
    const Scene& scene = scenes[environment];
    if (scene.params.size() != scene.shapes.size()) {
      throw std::invalid_argument("scene " + std::to_string(environment) + " has " + std::to_string(scene.params.size()) + " params and " + std::to_string(scene.shapes.size()) + " shapes");
    }
    // The rectangularity requirement, and the only cross-environment one.
    if (scene.params.size() != batch.bodies) {
      throw std::invalid_argument("scene " + std::to_string(environment) + " has " + std::to_string(scene.params.size()) + " bodies, batch is rectangular at " + std::to_string(batch.bodies));
    }
    if (scene.dt <= 0.0) {
      throw std::invalid_argument("scene " + std::to_string(environment) + " has non-positive dt");
    }
  }
}

}  // namespace grip
