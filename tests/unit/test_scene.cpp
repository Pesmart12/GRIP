#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "api/scene.hpp"
#include "core/rigid_body.hpp"

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
  scene.dt = 5.0e-4;
  return scene;
}

// Deliberately irregular, so a layout bug cannot hide behind a value that
// happens to be right in the wrong slot.
std::vector<RigidBodyState> DistinctStates(std::size_t bodies, double seed) {
  std::vector<RigidBodyState> states(bodies);
  for (std::size_t i = 0; i < bodies; ++i) {
    const double base = seed + 10.0 * static_cast<double>(i);
    states[i].q = Eigen::Vector3d(base + 1.0, base + 2.0, base + 3.0);
    states[i].v = Eigen::Vector3d(base + 4.0, base + 5.0, base + 6.0);
  }
  return states;
}

TEST(StateBatch, IsZeroedAndCorrectlySizedOnConstruction) {
  const StateBatch batch = make_state_batch(3, 2);
  EXPECT_EQ(batch.environments, 3u);
  EXPECT_EQ(batch.bodies, 2u);
  ASSERT_EQ(batch.values.size(), 3u * 2u * 6u);
  for (const double value : batch.values) {
    EXPECT_EQ(value, 0.0);
  }
}

TEST(StateBatch, RoundTripsEveryEnvironmentExactly) {
  // Exact equality, not a tolerance: this is a copy, and anything less than
  // bit-for-bit would mean the conversion is doing arithmetic it should not.
  StateBatch batch = make_state_batch(4, 3);
  std::vector<std::vector<RigidBodyState>> written;
  for (std::size_t environment = 0; environment < batch.environments; ++environment) {
    written.push_back(DistinctStates(batch.bodies, 100.0 * static_cast<double>(environment)));
    write_environment_state(batch, environment, written.back());
  }

  for (std::size_t environment = 0; environment < batch.environments; ++environment) {
    const std::vector<RigidBodyState> read = read_environment_state(batch, environment);
    ASSERT_EQ(read.size(), batch.bodies);
    for (std::size_t body = 0; body < batch.bodies; ++body) {
      EXPECT_EQ(read[body].q, written[environment][body].q) << "env " << environment << " body " << body;
      EXPECT_EQ(read[body].v, written[environment][body].v) << "env " << environment << " body " << body;
    }
  }
}

TEST(StateBatch, OneEnvironmentPlaneIsExactlyPackSystemsConcatenation) {
  // The claim the layout rests on: no new stacking convention was invented
  // here. An environment's slice of the array is byte-identical to what
  // PackSystem already produces, so numpy reshapes into this rather than
  // translating it. If this fails, the array layout and the physics layout
  // have diverged and every gradient handed to a caller is transposed.
  const std::size_t bodies = 3;
  const std::vector<RigidBodyState> states = DistinctStates(bodies, 7.0);

  StateBatch batch = make_state_batch(2, bodies);
  write_environment_state(batch, 1, states);

  const SystemStateVector packed = PackSystem(states);
  ASSERT_EQ(static_cast<std::size_t>(packed.size()), bodies * 6u);
  const std::size_t start = state_batch_offset(batch, 1, 0);
  for (std::size_t i = 0; i < static_cast<std::size_t>(packed.size()); ++i) {
    EXPECT_EQ(batch.values[start + i], packed(static_cast<Eigen::Index>(i))) << "value " << i;
  }
}

TEST(StateBatch, EnvironmentsDoNotAliasOneAnother) {
  StateBatch batch = make_state_batch(3, 2);
  write_environment_state(batch, 1, DistinctStates(2, 50.0));

  // Neighbours on both sides stay exactly zero, which is the property that
  // makes stepping environments in any order -- or in parallel -- safe.
  for (const std::size_t environment : {std::size_t{0}, std::size_t{2}}) {
    const std::vector<RigidBodyState> read = read_environment_state(batch, environment);
    for (std::size_t body = 0; body < batch.bodies; ++body) {
      EXPECT_TRUE(read[body].q.isZero(0.0)) << "env " << environment;
      EXPECT_TRUE(read[body].v.isZero(0.0)) << "env " << environment;
    }
  }
}

TEST(ValidateSceneBatch, AcceptsAWellFormedBatch) {
  const std::vector<Scene> scenes(3, MakeScene(2));
  const StateBatch batch = make_state_batch(3, 2);
  EXPECT_NO_THROW(validate_scene_batch(scenes, batch));
}

TEST(ValidateSceneBatch, AcceptsScenesThatDifferInEverythingButBodyCount) {
  // The point of a per-environment Scene. Only rectangularity is required,
  // so a caller can randomize geometry, mass, ground angle and contact
  // parameters across a batch -- which is exactly what a domain-randomized
  // task needs.
  std::vector<Scene> scenes(2, MakeScene(2));
  scenes[1].shapes[0] = BodyShape{{{-0.25, -0.1}, {0.25, -0.1}, {0.25, 0.1}, {-0.25, 0.1}}};
  scenes[1].params[0] = RigidBodyParams{3.0, 0.5};
  scenes[1].plane = HalfPlane{Eigen::Vector2d(-0.342, 0.940), 0.0};
  scenes[1].penalty = PenaltyParams{2.0e4, 10.0, 5.0, 0.9};
  scenes[1].dt = 1.0e-3;
  scenes[1].gravity = 3.71;

  const StateBatch batch = make_state_batch(2, 2);
  EXPECT_NO_THROW(validate_scene_batch(scenes, batch));
}

TEST(ValidateSceneBatch, RejectsSceneCountNotMatchingTheBatch) {
  const std::vector<Scene> scenes(2, MakeScene(2));
  const StateBatch batch = make_state_batch(3, 2);
  EXPECT_THROW(validate_scene_batch(scenes, batch), std::invalid_argument);
}

TEST(ValidateSceneBatch, RejectsARaggedBodyCount) {
  std::vector<Scene> scenes(2, MakeScene(2));
  scenes[1] = MakeScene(3);
  const StateBatch batch = make_state_batch(2, 2);
  EXPECT_THROW(validate_scene_batch(scenes, batch), std::invalid_argument);
}

TEST(ValidateSceneBatch, RejectsParamsAndShapesDisagreeingWithinAScene) {
  std::vector<Scene> scenes(1, MakeScene(2));
  scenes[0].shapes.pop_back();
  const StateBatch batch = make_state_batch(1, 2);
  EXPECT_THROW(validate_scene_batch(scenes, batch), std::invalid_argument);
}

TEST(ValidateSceneBatch, RejectsAMisSizedValueBuffer) {
  const std::vector<Scene> scenes(1, MakeScene(2));
  StateBatch batch = make_state_batch(1, 2);
  batch.values.pop_back();
  EXPECT_THROW(validate_scene_batch(scenes, batch), std::invalid_argument);
}

TEST(ValidateSceneBatch, RejectsNonPositiveTimestep) {
  std::vector<Scene> scenes(1, MakeScene(2));
  scenes[0].dt = 0.0;
  const StateBatch batch = make_state_batch(1, 2);
  EXPECT_THROW(validate_scene_batch(scenes, batch), std::invalid_argument);
}

}  // namespace
}  // namespace grip
