#include "api/simulate.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

#include "dynamics/integrator.hpp"

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


// One control step: the same wrench applied for `substeps` integration steps.
// The conversion in and out of RigidBodyState happens once around the whole
// macro step rather than once per substep, which is what api/scene.hpp means
// about amortizing it.
std::vector<RigidBodyState> AdvanceMacroStep(const Scene& scene, const std::vector<RigidBodyState>& start, const std::vector<Eigen::Vector3d>& control, std::size_t substeps) {
  std::vector<RigidBodyState> states = start;
  for (std::size_t k = 0; k < substeps; ++k) {
    states = step_system(states, scene.params, scene.shapes, scene.plane, scene.penalty, control, scene.dt, scene.gravity);
  }
  return states;
}

}  // namespace

void step_batch(const std::vector<Scene>& scenes, StateBatch& state, const ControlBatch& controls, std::size_t substeps) {
  RequireSubsteps(substeps);
  validate_scene_batch(scenes, state);
  RequireControlShape(controls, state.environments, state.bodies, 1, "controls");

  for (std::size_t environment = 0; environment < state.environments; ++environment) {
    const std::vector<RigidBodyState> start = read_environment_state(state, environment);
    const std::vector<Eigen::Vector3d> control = read_environment_control(controls, 0, environment);
    write_environment_state(state, environment, AdvanceMacroStep(scenes[environment], start, control, substeps));
  }
}


void rollout_batch(const std::vector<Scene>& scenes, const StateBatch& initial, const ControlBatch& controls, std::size_t substeps, TrajectoryBatch& trajectory) {
  RequireSubsteps(substeps);
  validate_scene_batch(scenes, initial);
  RequireControlShape(controls, initial.environments, initial.bodies, controls.steps, "controls");

  trajectory = make_trajectory_batch(controls.steps, initial.environments, initial.bodies);

  for (std::size_t environment = 0; environment < initial.environments; ++environment) {
    std::vector<RigidBodyState> states = read_environment_state(initial, environment);
    write_trajectory_state(trajectory, 0, environment, states);
    for (std::size_t step = 0; step < controls.steps; ++step) {
      states = AdvanceMacroStep(scenes[environment], states, read_environment_control(controls, step, environment), substeps);
      write_trajectory_state(trajectory, step + 1, environment, states);
    }
  }
}


RolloutGradientBatch adjoint_batch(const std::vector<Scene>& scenes, const TrajectoryBatch& trajectory, const ControlBatch& controls, std::size_t substeps, const TrajectoryBatch& dl_dZ, const ControlBatch& dl_dU) {
  RequireSubsteps(substeps);
  const std::size_t environments = trajectory.environments;
  const std::size_t bodies = trajectory.bodies;
  const std::size_t horizon = trajectory.steps;

  if (scenes.size() != environments) {
    throw std::invalid_argument("scene count " + std::to_string(scenes.size()) + " does not match trajectory environment count " + std::to_string(environments));
  }
  RequireControlShape(controls, environments, bodies, horizon, "controls");
  RequireControlShape(dl_dU, environments, bodies, horizon, "dl_dU");
  RequireTrajectoryShape(dl_dZ, environments, bodies, horizon, "dl_dZ");

  RolloutGradientBatch gradients;
  gradients.dJ_dZ0 = make_state_batch(environments, bodies);
  gradients.dJ_dU = make_control_batch(horizon, environments, bodies);

  const auto state_size = static_cast<Eigen::Index>(kStateValuesPerBody * bodies);

  for (std::size_t environment = 0; environment < environments; ++environment) {
    const Scene& scene = scenes[environment];

    // Terminal condition: Z_H sits in its own stage cost and in the
    // constraint that produced it, but in no constraint's dynamics.
    SystemStateVector adjoint = PackSystem(read_trajectory_state(dl_dZ, horizon, environment));

    for (std::size_t step = horizon; step-- > 0;) {
      const std::vector<Eigen::Vector3d> control = read_environment_control(controls, step, environment);

      // Rebuild the states inside this macro step. They were not recorded,
      // and every substep's Jacobian is taken at the state that substep
      // starts from -- getting that wrong is adjoint.md's failure mode 2.
      std::vector<std::vector<RigidBodyState>> inner(substeps + 1);
      inner[0] = read_trajectory_state(trajectory, step, environment);
      for (std::size_t k = 0; k < substeps; ++k) {
        inner[k + 1] = step_system(inner[k], scene.params, scene.shapes, scene.plane, scene.penalty, control, scene.dt, scene.gravity);
      }

      // Every substep reads the same wrench, so the control gradient
      // accumulates over the whole macro step instead of being read off
      // once. Both lines consume the adjoint at the END of their substep,
      // so dJ_dU is taken before the adjoint is stepped back -- the same
      // ordering the single-scene sweep depends on.
      SystemControlVector control_gradient = PackControls(read_environment_control(dl_dU, step, environment));
      for (std::size_t k = substeps; k-- > 0;) {
        const SystemStepJacobians jac = step_system_jacobian(inner[k], scene.params, scene.shapes, scene.plane, scene.penalty, scene.dt, scene.gravity);
        control_gradient.noalias() += jac.dZ_dF.transpose() * adjoint;
        adjoint = jac.dZ_dZ.transpose() * adjoint;
      }
      write_environment_control(gradients.dJ_dU, step, environment, UnpackControls(control_gradient, bodies));

      // Stage-cost seeds enter only at macro boundaries, which is where a
      // caller's stage cost is defined -- there is no cost inside a control
      // step to seed with.
      adjoint += PackSystem(read_trajectory_state(dl_dZ, step, environment));
    }

    if (adjoint.size() != state_size) {
      throw std::invalid_argument("adjoint size does not match the trajectory's body count");
    }
    write_environment_state(gradients.dJ_dZ0, environment, UnpackSystem(adjoint, bodies));
  }
  return gradients;
}

}  // namespace grip
