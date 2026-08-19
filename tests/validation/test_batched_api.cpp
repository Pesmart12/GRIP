#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "api/scene.hpp"
#include "api/simulate.hpp"
#include "core/rigid_body.hpp"
#include "dynamics/integrator.hpp"

namespace grip {
namespace {

BodyShape UnitSquare() {
  return BodyShape{{{-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}}};
}

Scene MakeScene(std::size_t bodies) {
  Scene scene;
  scene.params.assign(bodies, RigidBodyParams{1.0, 1.0 / 6.0});
  scene.shapes.assign(bodies, UnitSquare());
  scene.penalty = PenaltyParams{1.0e4, 50.0, 200.0, 0.5};
  scene.dt = 1.0e-3;
  return scene;
}

// Low enough to be in sustained contact, tilted enough to keep the pair off
// the parallel-face tie-break that pair_detection.md warns about.
std::vector<RigidBodyState> ContactingStates() {
  std::vector<RigidBodyState> states(2);
  states[0].q = Eigen::Vector3d(0.0, 0.499, 0.0);
  states[1].q = Eigen::Vector3d(0.13, 1.47, 0.21);
  states[1].v = Eigen::Vector3d(0.0, -0.4, 0.1);
  return states;
}

StateBatch BatchOf(const std::vector<std::vector<RigidBodyState>>& per_environment) {
  StateBatch batch = make_state_batch(per_environment.size(), per_environment.front().size());
  for (std::size_t i = 0; i < per_environment.size(); ++i) {
    write_environment_state(batch, i, per_environment[i]);
  }
  return batch;
}

ControlBatch VaryingControls(std::size_t steps, std::size_t environments, std::size_t bodies) {
  ControlBatch controls = make_control_batch(steps, environments, bodies);
  for (std::size_t s = 0; s < steps; ++s) {
    for (std::size_t e = 0; e < environments; ++e) {
      std::vector<Eigen::Vector3d> wrenches(bodies);
      for (std::size_t b = 0; b < bodies; ++b) {
        const double seed = 0.1 * static_cast<double>(s) + 0.7 * static_cast<double>(e) + 1.3 * static_cast<double>(b);
        wrenches[b] = Eigen::Vector3d(0.3 * seed, -0.2 * seed, 0.05 * seed);
      }
      write_environment_control(controls, s, e, wrenches);
    }
  }
  return controls;
}

TEST(BatchedRollout, BatchOfIdenticalScenesEqualsSeparateSingleScenesBitForBit) {
  // The load-bearing test for everything batched. Environments never
  // interact, so running four of them together must give exactly what
  // running each alone gives -- not to a tolerance, bit for bit. That pins
  // correctness and determinism at once, and it is the test that will catch
  // a parallel backend if one ever reorders something it should not.
  const std::size_t environments = 4;
  const std::vector<RigidBodyState> single = ContactingStates();
  const std::vector<Scene> scenes(environments, MakeScene(single.size()));
  const StateBatch initial = BatchOf(std::vector<std::vector<RigidBodyState>>(environments, single));
  const ControlBatch controls = VaryingControls(12, environments, single.size());

  TrajectoryBatch batched;
  rollout_batch(scenes, initial, controls, 5, batched);

  for (std::size_t environment = 0; environment < environments; ++environment) {
    const std::vector<Scene> alone(1, scenes[environment]);
    const StateBatch initial_alone = BatchOf({single});
    ControlBatch controls_alone = make_control_batch(controls.steps, 1, single.size());
    for (std::size_t s = 0; s < controls.steps; ++s) {
      write_environment_control(controls_alone, s, 0, read_environment_control(controls, s, environment));
    }

    TrajectoryBatch solo;
    rollout_batch(alone, initial_alone, controls_alone, 5, solo);

    for (std::size_t step = 0; step <= controls.steps; ++step) {
      const std::vector<RigidBodyState> from_batch = read_trajectory_state(batched, step, environment);
      const std::vector<RigidBodyState> from_solo = read_trajectory_state(solo, step, 0);
      for (std::size_t body = 0; body < single.size(); ++body) {
        EXPECT_EQ(from_batch[body].q, from_solo[body].q) << "env " << environment << " step " << step << " body " << body;
        EXPECT_EQ(from_batch[body].v, from_solo[body].v) << "env " << environment << " step " << step << " body " << body;
      }
    }
  }
}

TEST(BatchedRollout, SubstepsComposeExactlyLikeSteppingByHand) {
  // A control step of K substeps must be K applications of step_system under
  // the same held wrench -- exactly, since it is literally that loop. Pins
  // the zero-order hold: a bug that advanced the control index per substep
  // would still produce plausible motion.
  const std::size_t substeps = 7;
  const std::vector<RigidBodyState> single = ContactingStates();
  const Scene scene = MakeScene(single.size());
  const std::vector<Scene> scenes(1, scene);
  const ControlBatch controls = VaryingControls(4, 1, single.size());

  TrajectoryBatch trajectory;
  rollout_batch(scenes, BatchOf({single}), controls, substeps, trajectory);

  std::vector<RigidBodyState> states = single;
  for (std::size_t step = 0; step < controls.steps; ++step) {
    const std::vector<Eigen::Vector3d> held = read_environment_control(controls, step, 0);
    for (std::size_t k = 0; k < substeps; ++k) {
      states = step_system(states, scene.params, scene.shapes, scene.plane, scene.penalty, held, scene.dt, scene.gravity);
    }
    const std::vector<RigidBodyState> recorded = read_trajectory_state(trajectory, step + 1, 0);
    for (std::size_t body = 0; body < single.size(); ++body) {
      EXPECT_EQ(recorded[body].q, states[body].q) << "step " << step << " body " << body;
      EXPECT_EQ(recorded[body].v, states[body].v) << "step " << step << " body " << body;
    }
  }
}

TEST(BatchedStep, MatchesOneMacroStepOfTheRollout) {
  const std::vector<RigidBodyState> single = ContactingStates();
  const std::vector<Scene> scenes(1, MakeScene(single.size()));
  const ControlBatch controls = VaryingControls(1, 1, single.size());

  TrajectoryBatch trajectory;
  rollout_batch(scenes, BatchOf({single}), controls, 6, trajectory);

  StateBatch state = BatchOf({single});
  step_batch(scenes, state, controls, 6);

  const std::vector<RigidBodyState> stepped = read_environment_state(state, 0);
  const std::vector<RigidBodyState> rolled = read_trajectory_state(trajectory, 1, 0);
  for (std::size_t body = 0; body < single.size(); ++body) {
    EXPECT_EQ(stepped[body].q, rolled[body].q) << "body " << body;
    EXPECT_EQ(stepped[body].v, rolled[body].v) << "body " << body;
  }
}

// Total derivative of the final y of body 0 with respect to one component of
// one control wrench, by central difference through the whole batched
// rollout. The only independent handle on the substep accumulation.
double FiniteDifferenceControlGradient(const std::vector<Scene>& scenes, const StateBatch& initial, const ControlBatch& controls, std::size_t substeps, std::size_t step, std::size_t body, int component, double eps) {
  double values[2] = {0.0, 0.0};
  for (int side = 0; side < 2; ++side) {
    ControlBatch perturbed = controls;
    perturbed.values[control_batch_offset(perturbed, step, 0, body) + static_cast<std::size_t>(component)] += (side == 0 ? eps : -eps);
    TrajectoryBatch trajectory;
    rollout_batch(scenes, initial, perturbed, substeps, trajectory);
    values[side] = read_trajectory_state(trajectory, trajectory.steps, 0)[0].q.y();
  }
  return (values[0] - values[1]) / (2.0 * eps);
}

TEST(BatchedAdjoint, ControlGradientMatchesCentralFiniteDifferenceAcrossSubsteps) {
  // The test the substep accumulation exists to pass. A control wrench is
  // held across every substep of its macro step, so its gradient is a sum
  // over all of them; reading it off once would be dimensionally sound and
  // silently wrong, which is exactly adjoint.md's failure mode 1 wearing a
  // new hat. Only a finite difference over the whole rollout catches it.
  const std::size_t substeps = 4;
  const std::vector<RigidBodyState> single = ContactingStates();
  const std::vector<Scene> scenes(1, MakeScene(single.size()));
  const StateBatch initial = BatchOf({single});
  const ControlBatch controls = VaryingControls(6, 1, single.size());

  TrajectoryBatch trajectory;
  rollout_batch(scenes, initial, controls, substeps, trajectory);

  TrajectoryBatch dl_dZ = make_trajectory_batch(controls.steps, 1, single.size());
  dl_dZ.values[trajectory_batch_offset(dl_dZ, controls.steps, 0, 0) + 1] = 1.0;
  const ControlBatch dl_dU = make_control_batch(controls.steps, 1, single.size());

  const RolloutGradientBatch gradients = adjoint_batch(scenes, trajectory, controls, substeps, dl_dZ, dl_dU);

  double largest = 0.0;
  for (std::size_t step = 0; step < controls.steps; ++step) {
    for (std::size_t body = 0; body < single.size(); ++body) {
      for (int component = 0; component < 3; ++component) {
        const double analytic = gradients.dJ_dU.values[control_batch_offset(gradients.dJ_dU, step, 0, body) + static_cast<std::size_t>(component)];
        const double numeric = FiniteDifferenceControlGradient(scenes, initial, controls, substeps, step, body, component, 1.0e-6);
        EXPECT_NEAR(analytic, numeric, 1.0e-6 + 1.0e-5 * std::abs(numeric)) << "step " << step << " body " << body << " component " << component;
        largest = std::max(largest, std::abs(analytic));
      }
    }
  }

  // Without this the comparison above would pass on an adjoint that
  // returned zeros, since the finite difference of a control with no
  // influence is zero too. An early push has the most time to act, so the
  // largest entry is comfortably clear of noise.
  EXPECT_GT(largest, 1.0e-6);
}

TEST(BatchedAdjoint, InitialStateGradientMatchesCentralFiniteDifference) {
  const std::size_t substeps = 3;
  const std::vector<RigidBodyState> single = ContactingStates();
  const std::vector<Scene> scenes(1, MakeScene(single.size()));
  const StateBatch initial = BatchOf({single});
  const ControlBatch controls = VaryingControls(5, 1, single.size());

  TrajectoryBatch trajectory;
  rollout_batch(scenes, initial, controls, substeps, trajectory);

  TrajectoryBatch dl_dZ = make_trajectory_batch(controls.steps, 1, single.size());
  dl_dZ.values[trajectory_batch_offset(dl_dZ, controls.steps, 0, 0) + 1] = 1.0;
  const ControlBatch dl_dU = make_control_batch(controls.steps, 1, single.size());

  const RolloutGradientBatch gradients = adjoint_batch(scenes, trajectory, controls, substeps, dl_dZ, dl_dU);

  for (std::size_t body = 0; body < single.size(); ++body) {
    for (std::size_t component = 0; component < kStateValuesPerBody; ++component) {
      double values[2] = {0.0, 0.0};
      for (int side = 0; side < 2; ++side) {
        StateBatch perturbed = initial;
        perturbed.values[state_batch_offset(perturbed, 0, body) + component] += (side == 0 ? 1.0e-6 : -1.0e-6);
        TrajectoryBatch moved;
        rollout_batch(scenes, perturbed, controls, substeps, moved);
        values[side] = read_trajectory_state(moved, moved.steps, 0)[0].q.y();
      }
      const double numeric = (values[0] - values[1]) / 2.0e-6;
      const double analytic = gradients.dJ_dZ0.values[state_batch_offset(gradients.dJ_dZ0, 0, body) + component];
      EXPECT_NEAR(analytic, numeric, 1.0e-6 + 1.0e-5 * std::abs(numeric)) << "body " << body << " component " << component;
    }
  }
}

TEST(BatchedAdjoint, EnvironmentsDoNotCouple) {
  // Same property as the rollout test, one level up. A gradient seeded in
  // one environment must leave every other environment's gradient exactly
  // zero -- not small, zero.
  const std::size_t environments = 3;
  const std::vector<RigidBodyState> single = ContactingStates();
  const std::vector<Scene> scenes(environments, MakeScene(single.size()));
  const StateBatch initial = BatchOf(std::vector<std::vector<RigidBodyState>>(environments, single));
  const ControlBatch controls = VaryingControls(4, environments, single.size());

  TrajectoryBatch trajectory;
  rollout_batch(scenes, initial, controls, 3, trajectory);

  TrajectoryBatch dl_dZ = make_trajectory_batch(controls.steps, environments, single.size());
  dl_dZ.values[trajectory_batch_offset(dl_dZ, controls.steps, 1, 0) + 1] = 1.0;
  const ControlBatch dl_dU = make_control_batch(controls.steps, environments, single.size());

  const RolloutGradientBatch gradients = adjoint_batch(scenes, trajectory, controls, 3, dl_dZ, dl_dU);

  for (const std::size_t environment : {std::size_t{0}, std::size_t{2}}) {
    for (std::size_t body = 0; body < single.size(); ++body) {
      for (std::size_t component = 0; component < kStateValuesPerBody; ++component) {
        EXPECT_EQ(gradients.dJ_dZ0.values[state_batch_offset(gradients.dJ_dZ0, environment, body) + component], 0.0) << "env " << environment;
      }
    }
    for (std::size_t step = 0; step < controls.steps; ++step) {
      for (std::size_t body = 0; body < single.size(); ++body) {
        for (std::size_t component = 0; component < kControlValuesPerBody; ++component) {
          EXPECT_EQ(gradients.dJ_dU.values[control_batch_offset(gradients.dJ_dU, step, environment, body) + component], 0.0) << "env " << environment << " step " << step;
        }
      }
    }
  }

  // The seeded environment is not trivially zero, or the check above proves
  // nothing.
  double seeded = 0.0;
  for (std::size_t component = 0; component < kStateValuesPerBody; ++component) {
    seeded += std::abs(gradients.dJ_dZ0.values[state_batch_offset(gradients.dJ_dZ0, 1, 0) + component]);
  }
  EXPECT_GT(seeded, 1.0e-6);
}

TEST(BatchedApi, RejectsMalformedShapes) {
  const std::vector<RigidBodyState> single = ContactingStates();
  const std::vector<Scene> scenes(2, MakeScene(single.size()));
  StateBatch state = BatchOf(std::vector<std::vector<RigidBodyState>>(2, single));

  EXPECT_THROW(step_batch(scenes, state, make_control_batch(1, 2, single.size()), 0), std::invalid_argument);
  EXPECT_THROW(step_batch(scenes, state, make_control_batch(2, 2, single.size()), 1), std::invalid_argument);
  EXPECT_THROW(step_batch(scenes, state, make_control_batch(1, 3, single.size()), 1), std::invalid_argument);

  TrajectoryBatch trajectory;
  rollout_batch(scenes, state, make_control_batch(3, 2, single.size()), 2, trajectory);
  const ControlBatch good = make_control_batch(3, 2, single.size());
  const TrajectoryBatch good_seed = make_trajectory_batch(3, 2, single.size());
  EXPECT_THROW(adjoint_batch(scenes, trajectory, good, 2, make_trajectory_batch(2, 2, single.size()), good), std::invalid_argument);
  EXPECT_THROW(adjoint_batch(scenes, trajectory, good, 2, good_seed, make_control_batch(2, 2, single.size())), std::invalid_argument);
}

}  // namespace
}  // namespace grip
