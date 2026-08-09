#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "contact/detection.hpp"
#include "contact/half_plane.hpp"
#include "contact/penalty.hpp"
#include "core/rigid_body.hpp"
#include "dynamics/forces.hpp"
#include "dynamics/integrator.hpp"

namespace grip {
namespace {

// Unit square centered on the COM, corners at (+-0.5, +-0.5). Same
// helper and same vertex order as the detection tests.
BodyShape UnitSquare() {
  return BodyShape{{{-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}}};
}

TEST(PenaltyForce, SeparatedBodyFeelsNothing) {
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.6, 0.0);  // bottom corners at y = 0.1
  const PenaltyParams penalty{/*stiffness=*/1.0e4};

  const Eigen::Vector3d force = penalty_force_body(state, UnitSquare(), HalfPlane{}, penalty);

  // Exactly zero, not approximately: every d_i > 0, so min(0, d_i) is an
  // identical zero and nothing accumulates.
  EXPECT_TRUE(force.isZero(0.0));
}

TEST(PenaltyForce, ZeroStiffnessFeelsNothingEvenWhenPenetrating) {
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.3, 0.0);  // bottom corners 0.2 through the floor

  const Eigen::Vector3d force = penalty_force_body(state, UnitSquare(), HalfPlane{}, PenaltyParams{});

  EXPECT_TRUE(force.isZero(0.0));
}

TEST(PenaltyForce, SinglePenetratingVertexProducesForceAndTorque) {
  // Tilted so exactly one corner is through the floor, and so that
  // corner is NOT directly below the COM -- otherwise the moment arm
  // would be vertical and the torque would vanish, which is the one case
  // a missing torque term could still reproduce.
  const double theta = 0.3;
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.5, theta);
  const double stiffness = 100.0;
  const PenaltyParams penalty{stiffness};
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;

  const std::vector<Contact> contacts = detect_contacts_body(state, shape, ground);
  ASSERT_LT(contacts[0].signed_distance, 0.0);
  for (std::size_t i = 1; i < contacts.size(); ++i) {
    ASSERT_GT(contacts[i].signed_distance, 0.0) << "vertex " << i << " should be clear";
  }

  const Eigen::Vector3d force = penalty_force_body(state, shape, ground, penalty);

  const double normal_force = -stiffness * contacts[0].signed_distance;
  const Eigen::Vector2d arm = contacts[0].point - state.q.head<2>();

  // Force is the normal times the magnitude -- ground normal is +y.
  EXPECT_DOUBLE_EQ(force.x(), 0.0);
  EXPECT_DOUBLE_EQ(force.y(), normal_force);

  // Torque cross-checked against the physicist's moment, r x F, computed
  // here as a 2D cross product rather than via the n . arm^perp form the
  // implementation uses. Same number, independent expression.
  const Eigen::Vector2d applied = normal_force * ground.normal;
  const double moment = arm.x() * applied.y() - arm.y() * applied.x();
  EXPECT_DOUBLE_EQ(force.z(), moment);

  // The contact sits left of the COM and pushes up, so the box tips
  // clockwise -- negative torque in the counterclockwise-positive
  // convention.
  EXPECT_LT(arm.x(), 0.0);
  EXPECT_LT(force.z(), 0.0);
}

TEST(PenaltyForce, SymmetricFlatContactSumsToPureNormalForce) {
  // Two active corners at equal depth. The individual moments are equal
  // and opposite, so the resultant is a pure upward force -- the body
  // only ever feels the sum, never the individual lambda_i.
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.4, 0.0);  // bottom corners at d = -0.1
  const double stiffness = 100.0;

  const Eigen::Vector3d force = penalty_force_body(state, UnitSquare(), HalfPlane{}, PenaltyParams{stiffness});

  EXPECT_DOUBLE_EQ(force.x(), 0.0);
  EXPECT_DOUBLE_EQ(force.y(), 2.0 * stiffness * 0.1);
  EXPECT_DOUBLE_EQ(force.z(), 0.0);
}

TEST(PenaltyForce, RespectsNonDefaultPlane) {
  // A wall: free space is x >= -1, so the escape direction is +x and the
  // penalty force pushes horizontally, not up.
  HalfPlane wall;
  wall.normal = Eigen::Vector2d(1.0, 0.0);
  wall.offset = -1.0;

  RigidBodyState state;
  state.q = Eigen::Vector3d(-0.8, 0.0, 0.0);  // left corners 0.3 past the wall
  const double stiffness = 100.0;

  const Eigen::Vector3d force = penalty_force_body(state, UnitSquare(), wall, PenaltyParams{stiffness});

  EXPECT_DOUBLE_EQ(force.x(), 2.0 * stiffness * 0.3);
  EXPECT_DOUBLE_EQ(force.y(), 0.0);
  EXPECT_DOUBLE_EQ(force.z(), 0.0);  // symmetric about the wall normal
}

// Flat box with both bottom corners 0.1 through the floor, so the spring
// alone would give k*0.1 per corner. Velocity is what the damper reads.
RigidBodyState FlatAndPenetrating(double vy, double omega) {
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.4, 0.0);
  state.v = Eigen::Vector3d(0.0, vy, omega);
  return state;
}

TEST(PenaltyForce, DamperAddsForceWhileApproaching) {
  // Closing at 2 m/s: ddot = -2 at both corners, so each contributes
  // -k*d - b*ddot = 100*0.1 + 10*2 = 30 rather than the spring's 10.
  const PenaltyParams penalty{/*stiffness=*/100.0, /*damping=*/10.0};

  const Eigen::Vector3d force = penalty_force_body(FlatAndPenetrating(-2.0, 0.0), UnitSquare(), HalfPlane{}, penalty);

  EXPECT_DOUBLE_EQ(force.y(), 2.0 * 30.0);
  EXPECT_DOUBLE_EQ(force.z(), 0.0);  // symmetric, so the moments cancel
}

TEST(PenaltyForce, DamperRemovesForceWhileSeparating) {
  // Pulling away at 0.5 m/s: 100*0.1 - 10*0.5 = 5 per corner. Less than
  // the spring alone, but still a push -- the clamp has not engaged.
  const PenaltyParams penalty{/*stiffness=*/100.0, /*damping=*/10.0};

  const Eigen::Vector3d force = penalty_force_body(FlatAndPenetrating(0.5, 0.0), UnitSquare(), HalfPlane{}, penalty);

  EXPECT_DOUBLE_EQ(force.y(), 2.0 * 5.0);
  EXPECT_GT(force.y(), 0.0);
}

TEST(PenaltyForce, ClampPreventsAdhesionOnSeparation) {
  // Pulling away at 2 m/s: 100*0.1 - 10*2 = -10 per corner. Unclamped
  // that is a negative normal force -- the plane pulling a departing
  // body back down, which is the failure the outer max exists to stop.
  const PenaltyParams penalty{/*stiffness=*/100.0, /*damping=*/10.0};
  const RigidBodyState state = FlatAndPenetrating(2.0, 0.0);

  const Eigen::Vector3d force = penalty_force_body(state, UnitSquare(), HalfPlane{}, penalty);

  // Exactly zero, not merely small or non-negative.
  EXPECT_TRUE(force.isZero(0.0));

  // Confirm the unclamped law really would have gone negative here, so
  // this test cannot pass for the wrong reason.
  const std::vector<Contact> contacts = detect_contacts_body(state, UnitSquare(), HalfPlane{});
  const double unclamped = -penalty.stiffness * contacts[0].signed_distance - penalty.damping * state.v.y();
  EXPECT_LT(unclamped, 0.0);
}

TEST(PenaltyForce, DampingIsIgnoredWhileSeparated) {
  // Well clear of the plane and diving at it hard. Activation is on d,
  // not on the closing rate, so nothing fires until contact.
  const PenaltyParams penalty{/*stiffness=*/100.0, /*damping=*/10.0};
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.6, 0.0);
  state.v = Eigen::Vector3d(0.0, -5.0, 0.0);

  EXPECT_TRUE(penalty_force_body(state, UnitSquare(), HalfPlane{}, penalty).isZero(0.0));
}

TEST(PenaltyForce, DamperOpposesRotation) {
  // Spinning counterclockwise with no translation: the left corner is
  // driven down and the right corner lifted, so ddot differs between
  // them and the damper loads them unequally.
  //
  // ddot = J_i . v with J = [0, 1, -+0.5] and v = (0, 0, 1), giving
  // -0.5 and +0.5, hence lambda = 10 + 5 = 15 and 10 - 5 = 5.
  const PenaltyParams penalty{/*stiffness=*/100.0, /*damping=*/10.0};

  const Eigen::Vector3d force = penalty_force_body(FlatAndPenetrating(0.0, 1.0), UnitSquare(), HalfPlane{}, penalty);

  // The two damping contributions cancel in the normal direction...
  EXPECT_DOUBLE_EQ(force.y(), 2.0 * 10.0);
  // ...but not in the moment: -0.5*15 + 0.5*5 = -5, a torque that
  // opposes the rotation, which is what a dissipative term must do.
  EXPECT_DOUBLE_EQ(force.z(), -5.0);
}

TEST(PenaltyForce, RestingBoxIsAnExactFixedPointOfTheDiscreteMap) {
  // Static equilibrium: two active corners share the weight, so
  // 2*k*(-d) = mg and the resting penetration is d* = -mg/(2k).
  //
  // Every quantity below is a dyadic rational, so the balance is exact in
  // binary rather than merely close: g = 8, m = 1, k = 1024 give
  // d* = -8/2048 = -1/256, and 0.5 - 1/256 is exactly representable, so
  // the round trip through the vertex position loses nothing.
  const double gravity = 8.0;
  const double stiffness = 1024.0;
  const RigidBodyParams params{/*mass=*/1.0, /*inertia=*/1.0 / 6.0};
  const double resting_depth = -params.mass * gravity / (2.0 * stiffness);  // -1/256
  ASSERT_DOUBLE_EQ(resting_depth, -1.0 / 256.0);

  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.5 + resting_depth, 0.0);
  const BodyShape shape = UnitSquare();
  const PenaltyParams penalty{stiffness};

  // The two forces cancel identically, not approximately.
  const Eigen::Vector3d contact = penalty_force_body(state, shape, HalfPlane{}, penalty);
  const Eigen::Vector3d weight = gravity_force(params, gravity);
  EXPECT_DOUBLE_EQ(contact.y(), params.mass * gravity);
  EXPECT_TRUE((contact + weight).isZero(0.0));

  // So a step is the identity: zero net force leaves v at zero, and zero
  // velocity leaves q where it was. A fixed point of the discrete map,
  // not just of the continuous equations.
  const RigidBodyState next = step_body(state, params, shape, HalfPlane{}, penalty, Eigen::Vector3d::Zero(), 1.0e-3, gravity);
  EXPECT_TRUE(next.v.isZero(0.0));
  EXPECT_TRUE(next.q.isApprox(state.q, 0.0));
}

}  // namespace
}  // namespace grip
