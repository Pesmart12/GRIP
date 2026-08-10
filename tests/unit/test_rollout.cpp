#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "contact/half_plane.hpp"
#include "contact/penalty.hpp"
#include "core/rigid_body.hpp"
#include "dynamics/integrator.hpp"
#include "gradient/rollout.hpp"

namespace grip {
namespace {

BodyShape UnitSquare() {
  return BodyShape{{{-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}}};
}

TEST(RolloutSystem, RecordsOneMoreStateThanThereAreControls) {
  // H controls take you through H transitions, so the trajectory holds H+1
  // states with the initial one first. Every adjoint off-by-one starts by
  // getting this count wrong.
  const std::vector<RigidBodyState> initial(2);
  const std::vector<RigidBodyParams> params(2);
  const std::vector<BodyShape> shapes(2);
  const std::size_t horizon = 7;
  const std::vector<std::vector<Eigen::Vector3d>> controls(horizon, std::vector<Eigen::Vector3d>(2, Eigen::Vector3d::Zero()));

  const std::vector<std::vector<RigidBodyState>> trajectory = rollout_system(initial, params, shapes, HalfPlane{}, PenaltyParams{}, controls, 0.01);

  ASSERT_EQ(trajectory.size(), horizon + 1);
  for (const std::vector<RigidBodyState>& states : trajectory) {
    EXPECT_EQ(states.size(), 2u);
  }
  EXPECT_TRUE(trajectory.front()[0].q.isApprox(initial[0].q, 0.0));
  EXPECT_TRUE(trajectory.front()[0].v.isApprox(initial[0].v, 0.0));
}

TEST(RolloutSystem, MatchesSteppingByHand) {
  // The rollout must be exactly repeated step_system and nothing else --
  // no re-evaluation, no reordering, no drift.
  std::vector<RigidBodyState> initial(1);
  initial[0].q = Eigen::Vector3d(0.0, 0.9, 0.2);
  initial[0].v = Eigen::Vector3d(0.3, -0.5, 0.4);
  const std::vector<RigidBodyParams> params(1, RigidBodyParams{1.0, 1.0 / 6.0});
  const std::vector<BodyShape> shapes(1, UnitSquare());
  const HalfPlane ground;
  const PenaltyParams penalty{/*stiffness=*/1.0e4, /*damping=*/50.0};
  const double dt = 1.0e-3;

  std::vector<std::vector<Eigen::Vector3d>> controls;
  for (int t = 0; t < 40; ++t) {
    controls.push_back({Eigen::Vector3d(0.1 * t, -0.2, 0.05)});
  }

  const std::vector<std::vector<RigidBodyState>> trajectory = rollout_system(initial, params, shapes, ground, penalty, controls, dt);

  std::vector<RigidBodyState> by_hand = initial;
  for (std::size_t t = 0; t < controls.size(); ++t) {
    by_hand = step_system(by_hand, params, shapes, ground, penalty, controls[t], dt);
    EXPECT_TRUE(trajectory[t + 1][0].q.isApprox(by_hand[0].q, 0.0)) << "step " << t;
    EXPECT_TRUE(trajectory[t + 1][0].v.isApprox(by_hand[0].v, 0.0)) << "step " << t;
  }
}

TEST(RolloutSystem, EmptyControlSequenceReturnsOnlyTheInitialState) {
  const std::vector<RigidBodyState> initial(1);
  const std::vector<RigidBodyParams> params(1);
  const std::vector<BodyShape> shapes(1);

  const std::vector<std::vector<RigidBodyState>> trajectory = rollout_system(initial, params, shapes, HalfPlane{}, PenaltyParams{}, {}, 0.01);

  EXPECT_EQ(trajectory.size(), 1u);
}

TEST(PackControls, RoundTripsThroughTheStackedForm) {
  // Body i at [3i, 3i+3), matching PackSystem's convention for states. The
  // gradient path needs controls as a point in R^{3B}; forward simulation
  // wants them as named wrenches.
  const std::vector<Eigen::Vector3d> u = {Eigen::Vector3d(1.0, 2.0, 3.0), Eigen::Vector3d(-4.0, 5.0, -6.0)};

  const SystemControlVector packed = PackControls(u);

  ASSERT_EQ(packed.size(), 6);
  EXPECT_DOUBLE_EQ(packed(0), 1.0);
  EXPECT_DOUBLE_EQ(packed(3), -4.0);

  const std::vector<Eigen::Vector3d> restored = UnpackControls(packed, 2);
  ASSERT_EQ(restored.size(), 2u);
  EXPECT_TRUE(restored[0].isApprox(u[0], 0.0));
  EXPECT_TRUE(restored[1].isApprox(u[1], 0.0));
}

}  // namespace
}  // namespace grip
