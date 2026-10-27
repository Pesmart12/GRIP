#include "api/simulate.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

#include "gradient/rollout.hpp"

// The batched loops run one environment per iteration under OpenMP when it
// is available. Each iteration reads only shared const inputs and writes only
// its own environment's slice of the outputs, all of which are allocated
// before the loop, so there is nothing to synchronize -- and since each
// environment is computed start to finish by one thread running the same
// serial code, the result is bit-identical for any thread count. Dynamic
// scheduling because environments differ in cost with their contact count.
// The index is signed because MSVC's OpenMP 2.0 rejects unsigned ones.
#if defined(_OPENMP)
#define GRIP_PARALLEL_FOR_ENVIRONMENTS _Pragma("omp parallel for schedule(dynamic, 1)")
#else
#define GRIP_PARALLEL_FOR_ENVIRONMENTS
#endif

namespace grip {
namespace {

void RequireSubsteps(std::size_t substeps) {
  if (substeps == 0) {
    throw std::invalid_argument("substeps must be at least 1");
  }
}


void RequireControlShape(const ControlBatch& controls, std::size_t environments, std::size_t bodies, std::size_t steps, const char* what) {
  if (controls.environments != environments || controls.bodies != bodies || controls.steps != steps) {
    throw std::invalid_argument(std::string(what) + " is (" + std::to_string(controls.steps) + ", " + std::to_string(controls.environments) + ", " + std::to_string(controls.bodies) + "), expected (" + std::to_string(steps) + ", " + std::to_string(environments) + ", " + std::to_string(bodies) + ")");
  }
  if (controls.values.size() != steps * environments * bodies * kControlValuesPerBody) {
    throw std::invalid_argument(std::string(what) + " holds the wrong number of values for its shape");
  }
}


void RequireTrajectoryShape(const TrajectoryBatch& trajectory, std::size_t environments, std::size_t bodies, std::size_t steps, const char* what) {
  if (trajectory.environments != environments || trajectory.bodies != bodies || trajectory.steps != steps) {
    throw std::invalid_argument(std::string(what) + " is (" + std::to_string(trajectory.steps) + " + 1, " + std::to_string(trajectory.environments) + ", " + std::to_string(trajectory.bodies) + "), expected (" + std::to_string(steps) + " + 1, " + std::to_string(environments) + ", " + std::to_string(bodies) + ")");
  }
  if (trajectory.values.size() != (steps + 1) * environments * bodies * kStateValuesPerBody) {
    throw std::invalid_argument(std::string(what) + " holds the wrong number of values for its shape");
  }
}


// One environment's controls, in the per-step form rollout_system reads.
std::vector<std::vector<Eigen::Vector3d>> ReadEnvironmentControls(const ControlBatch& controls, std::size_t environment) {
  std::vector<std::vector<Eigen::Vector3d>> sequence(controls.steps);
  for (std::size_t step = 0; step < controls.steps; ++step) {
    sequence[step] = read_environment_control(controls, step, environment);
  }
  return sequence;
}

}  // namespace

void step_batch(const std::vector<Scene>& scenes, StateBatch& state, const ControlBatch& controls, std::size_t substeps) {
  RequireSubsteps(substeps);
  validate_scene_batch(scenes, state);
  RequireControlShape(controls, state.environments, state.bodies, 1, "controls");

  // One control step is a one-step rollout, so substeps are composed in
  // exactly one place: rollout_system.
  const auto environment_count = static_cast<std::ptrdiff_t>(state.environments);
  GRIP_PARALLEL_FOR_ENVIRONMENTS
  for (std::ptrdiff_t index = 0; index < environment_count; ++index) {
    const auto environment = static_cast<std::size_t>(index);
    const Scene& scene = scenes[environment];
    const std::vector<std::vector<RigidBodyState>> stepped = rollout_system(read_environment_state(state, environment), scene.params, scene.shapes, scene.plane, scene.penalty, ReadEnvironmentControls(controls, environment), substeps, scene.dt, scene.gravity);
    write_environment_state(state, environment, stepped.back());
  }
}


void rollout_batch(const std::vector<Scene>& scenes, const StateBatch& initial, const ControlBatch& controls, std::size_t substeps, TrajectoryBatch& trajectory) {
  RequireSubsteps(substeps);
  validate_scene_batch(scenes, initial);
  RequireControlShape(controls, initial.environments, initial.bodies, controls.steps, "controls");

  trajectory = make_trajectory_batch(controls.steps, initial.environments, initial.bodies);

  const auto environment_count = static_cast<std::ptrdiff_t>(initial.environments);
  GRIP_PARALLEL_FOR_ENVIRONMENTS
  for (std::ptrdiff_t index = 0; index < environment_count; ++index) {
    const auto environment = static_cast<std::size_t>(index);
    const Scene& scene = scenes[environment];
    const std::vector<std::vector<RigidBodyState>> states = rollout_system(read_environment_state(initial, environment), scene.params, scene.shapes, scene.plane, scene.penalty, ReadEnvironmentControls(controls, environment), substeps, scene.dt, scene.gravity);
    for (std::size_t step = 0; step <= controls.steps; ++step) {
      write_trajectory_state(trajectory, step, environment, states[step]);
    }
  }
}


RolloutGradientBatch adjoint_batch(const std::vector<Scene>& scenes, const TrajectoryBatch& trajectory, const ControlBatch& controls, std::size_t substeps, const TrajectoryBatch& dl_dZ, const ControlBatch& dl_dU) {
  RequireSubsteps(substeps);
  const std::size_t environments = trajectory.environments;
  const std::size_t bodies = trajectory.bodies;
  const std::size_t horizon = trajectory.steps;

  validate_scenes(scenes, environments, bodies);
  RequireTrajectoryShape(trajectory, environments, bodies, horizon, "trajectory");
  RequireControlShape(controls, environments, bodies, horizon, "controls");
  RequireControlShape(dl_dU, environments, bodies, horizon, "dl_dU");
  RequireTrajectoryShape(dl_dZ, environments, bodies, horizon, "dl_dZ");

  RolloutGradientBatch gradients;
  gradients.dobjective_dZ0 = make_state_batch(environments, bodies);
  gradients.dobjective_dU = make_control_batch(horizon, environments, bodies);

  // The recursion itself is adjoint_system's -- this only converts each
  // environment's slice of the batch into the form it reads and back.
  const auto environment_count = static_cast<std::ptrdiff_t>(environments);
  GRIP_PARALLEL_FOR_ENVIRONMENTS
  for (std::ptrdiff_t index = 0; index < environment_count; ++index) {
    const auto environment = static_cast<std::size_t>(index);
    const Scene& scene = scenes[environment];

    std::vector<std::vector<RigidBodyState>> states(horizon + 1);
    std::vector<SystemStateVector> state_seeds(horizon + 1);
    for (std::size_t step = 0; step <= horizon; ++step) {
      states[step] = read_trajectory_state(trajectory, step, environment);
      state_seeds[step] = PackSystem(read_trajectory_state(dl_dZ, step, environment));
    }
    std::vector<SystemControlVector> control_seeds(horizon);
    for (std::size_t step = 0; step < horizon; ++step) {
      control_seeds[step] = PackControls(read_environment_control(dl_dU, step, environment));
    }

    const RolloutGradients sweep = adjoint_system(states, scene.params, scene.shapes, scene.plane, scene.penalty, ReadEnvironmentControls(controls, environment), substeps, state_seeds, control_seeds, scene.dt, scene.gravity);

    write_environment_state(gradients.dobjective_dZ0, environment, UnpackSystem(sweep.dobjective_dZ0, bodies));
    for (std::size_t step = 0; step < horizon; ++step) {
      write_environment_control(gradients.dobjective_dU, step, environment, UnpackControls(sweep.dobjective_dU[step], bodies));
    }
  }
  return gradients;
}

}  // namespace grip
