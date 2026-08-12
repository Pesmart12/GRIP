#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
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
RigidBodyState FlatAndPenetrating(double vy, double omega, double vx = 0.0) {
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.4, 0.0);
  state.v = Eigen::Vector3d(vx, vy, omega);
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

TEST(PenaltyForce, StickingFrictionOpposesSliding) {
  // Flat box 0.1 into the floor, sliding in +x at 2 m/s. With
  // n = (0, 1) the surface direction is n^perp = (-1, 0), so both bottom
  // corners have J_perp = [-1, 0, -0.5] and slip s = -2.
  //
  // lambda = 100*0.1 = 10 each, so the cone allows up to mu*lambda = 5.
  // The unclamped demand is |b_slip*s| = 4, inside the bound, so this is
  // the sticking branch: beta = -b_slip*s = +4 each. J_perp^T carries
  // that to a force in -x plus a moment, since friction acts half a
  // metre below the COM and tips the box forward.
  const double slip_damping = 2.0;
  const PenaltyParams penalty{/*stiffness=*/100.0, /*damping=*/0.0, slip_damping, /*friction=*/0.5};

  const Eigen::Vector3d force = penalty_force_body(FlatAndPenetrating(0.0, 0.0, 2.0), UnitSquare(), HalfPlane{}, penalty);

  EXPECT_DOUBLE_EQ(force.x(), -4.0 * slip_damping);
  EXPECT_DOUBLE_EQ(force.y(), 2.0 * 100.0 * 0.1);  // normal force is untouched
  EXPECT_DOUBLE_EQ(force.z(), -2.0 * slip_damping);
  EXPECT_LT(force.x(), 0.0) << "friction must oppose motion in +x";
}

TEST(PenaltyForce, FrictionVanishesWithoutSlip) {
  // Pressed in but not moving sideways: nothing to resist. Continuous at
  // s = 0, unlike a sign() formulation, which would jump the full
  // 2*mu*lambda across exactly this state.
  const PenaltyParams penalty{/*stiffness=*/100.0, /*damping=*/0.0, /*slip_damping=*/2.0, /*friction=*/0.5};

  const Eigen::Vector3d force = penalty_force_body(FlatAndPenetrating(0.0, 0.0, 0.0), UnitSquare(), HalfPlane{}, penalty);

  EXPECT_DOUBLE_EQ(force.x(), 0.0);
  EXPECT_DOUBLE_EQ(force.z(), 0.0);
}

TEST(PenaltyForce, FrictionIsIgnoredWhileSeparated) {
  // Clear of the plane and sliding fast. No normal force means no
  // friction -- it gates on lambda, not on d_i directly.
  const PenaltyParams penalty{/*stiffness=*/100.0, /*damping=*/0.0, /*slip_damping=*/2.0, /*friction=*/0.5};
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.6, 0.0);
  state.v = Eigen::Vector3d(5.0, 0.0, 0.0);

  EXPECT_TRUE(penalty_force_body(state, UnitSquare(), HalfPlane{}, penalty).isZero(0.0));
}

TEST(PenaltyForce, FrictionIsSuppressedByTheAdhesionClamp) {
  // Penetrating but leaving fast enough that the clamp zeroes lambda.
  // With no normal force there is no friction either, even though the
  // body is still sliding sideways at the time.
  const PenaltyParams penalty{/*stiffness=*/100.0, /*damping=*/10.0, /*slip_damping=*/2.0, /*friction=*/0.5};

  // 100*0.1 - 10*2 = -10 < 0, so lambda clamps to zero.
  const Eigen::Vector3d force = penalty_force_body(FlatAndPenetrating(2.0, 0.0, 3.0), UnitSquare(), HalfPlane{}, penalty);

  EXPECT_TRUE(force.isZero(0.0));
}

TEST(PenaltyForce, ZeroFrictionProducesNoTangentialForceAtAll) {
  // mu = 0 collapses the cone to a line, so no tangential force exists
  // however hard the body slides or however large b_slip is. This is why
  // every test written before friction still passes untouched.
  const PenaltyParams frictionless{/*stiffness=*/100.0, /*damping=*/0.0, /*slip_damping=*/1000.0, /*friction=*/0.0};

  const Eigen::Vector3d force = penalty_force_body(FlatAndPenetrating(0.0, 1.5, 4.0), UnitSquare(), HalfPlane{}, frictionless);

  EXPECT_DOUBLE_EQ(force.x(), 0.0);
  EXPECT_DOUBLE_EQ(force.z(), 0.0);
  EXPECT_GT(force.y(), 0.0) << "the normal spring must be unaffected";
}

TEST(PenaltyForce, SlidingForceSaturatesExactlyAtTheCone) {
  // Sliding fast enough that the unclamped demand exceeds mu*lambda, so
  // each contact is pinned to the cone boundary. The magnitude is then
  // known rather than computed from slip: exactly mu*lambda, opposing.
  const double stiffness = 100.0;
  const double friction = 0.5;
  const PenaltyParams penalty{stiffness, /*damping=*/0.0, /*slip_damping=*/50.0, friction};
  const double normal_force = stiffness * 0.1;  // per corner, at d = -0.1

  for (const double vx : {1.0, 5.0, 100.0}) {
    const Eigen::Vector3d force = penalty_force_body(FlatAndPenetrating(0.0, 0.0, vx), UnitSquare(), HalfPlane{}, penalty);
    // Two corners, each at the bound, each carried through J_perp,x = -1.
    EXPECT_DOUBLE_EQ(force.x(), -2.0 * friction * normal_force) << "vx " << vx;
  }
}

TEST(PenaltyForce, ConeBoundIsNeverExceeded) {
  // |beta_i| <= mu*lambda_i at every contact, at any slip speed. Unlike
  // sticking, this one is exact rather than approximate -- the clamp
  // enforces it by construction, which is the mirror of lambda >= 0
  // holding exactly in the normal direction.
  const double stiffness = 100.0;
  const double friction = 0.5;
  const PenaltyParams penalty{stiffness, /*damping=*/0.0, /*slip_damping=*/50.0, friction};
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;

  for (const double vx : {-40.0, -1.0, -0.01, 0.0, 0.01, 1.0, 40.0}) {
    for (const double omega : {-3.0, 0.0, 3.0}) {
      const RigidBodyState state = FlatAndPenetrating(0.0, omega, vx);
      const std::vector<Contact> contacts = detect_contacts_body(state, shape, ground);
      const std::vector<Eigen::RowVector3d> perp = detect_contacts_body_perp_jacobian(state, shape, ground);

      for (std::size_t i = 0; i < contacts.size(); ++i) {
        if (contacts[i].signed_distance >= 0.0) {
          continue;
        }
        const double normal_force = -stiffness * contacts[i].signed_distance;
        const double slip_rate = perp[i].dot(state.v);
        const double beta = -std::clamp(penalty.slip_damping * slip_rate, -friction * normal_force, friction * normal_force);
        EXPECT_LE(std::abs(beta), friction * normal_force + 1e-15) << "vx " << vx << ", omega " << omega << ", contact " << i;
      }
    }
  }
}

TEST(PenaltyForce, FrictionDissipatesInBothRegimes) {
  // v . f <= 0 for the tangential part, across sticking and sliding
  // alike. This is the property that SURVIVES the cone -- unlike the
  // Jacobian's negative semidefiniteness, which does not.
  const PenaltyParams penalty{/*stiffness=*/100.0, /*damping=*/0.0, /*slip_damping=*/10.0, /*friction=*/0.5};
  const PenaltyParams frictionless{/*stiffness=*/100.0, /*damping=*/0.0, /*slip_damping=*/0.0, /*friction=*/0.0};
  const BodyShape shape = UnitSquare();

  for (const double vx : {-3.0, -0.5, -0.05, 0.0, 0.05, 0.5, 3.0}) {
    for (const double omega : {-2.0, 0.0, 2.0}) {
      const RigidBodyState state = FlatAndPenetrating(0.0, omega, vx);
      const Eigen::Vector3d tangential = penalty_force_body(state, shape, HalfPlane{}, penalty) - penalty_force_body(state, shape, HalfPlane{}, frictionless);
      EXPECT_LE(state.v.dot(tangential), 0.0) << "vx " << vx << ", omega " << omega;
    }
  }
}

TEST(PenaltyForce, FrictionAngleHoldsBelowAndReleasesAbove) {
  // The test that proves the sign conventions are right, because nothing
  // else is as physically legible: a body rests on a slope exactly while
  // tan(theta) <= mu.
  //
  // Rather than tilting the plane and rotating the body to match, tilt
  // GRAVITY -- a box on a slope in vertical gravity is the same problem
  // as a box on flat ground in gravity rotated by theta. Applied through
  // the control wrench with the integrator's own gravity switched off.
  //
  // mu = 0.5 gives a friction angle of atan(0.5) = 26.6 degrees, so 20
  // degrees must hold and 35 must run.
  const RigidBodyParams params{/*mass=*/1.0, /*inertia=*/1.0 / 6.0};
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;
  const double stiffness = 1.0e4;
  const PenaltyParams penalty{stiffness, /*damping=*/50.0, /*slip_damping=*/1000.0, /*friction=*/0.5};
  const double gravity = 9.81;
  const double dt = 1.0e-4;
  const int steps = 5000;  // 0.5 s

  const auto slide_speed_after = [&](double degrees) {
    const double theta = degrees * std::numbers::pi / 180.0;
    // Gravity rotated by theta, and the body started at the resting
    // penetration for the normal component so it does not bounce first.
    const Eigen::Vector3d tilted(params.mass * gravity * std::sin(theta), -params.mass * gravity * std::cos(theta), 0.0);
    RigidBodyState state;
    state.q = Eigen::Vector3d(0.0, 0.5 - params.mass * gravity * std::cos(theta) / (2.0 * stiffness), 0.0);

    for (int i = 0; i < steps; ++i) {
      state = step_body(state, params, shape, ground, penalty, tilted, dt, /*gravity=*/0.0);
    }
    return std::abs(state.v.x());
  };

  const double holding = slide_speed_after(20.0);
  const double running = slide_speed_after(35.0);

  EXPECT_LT(holding, 1.0e-2) << "below the friction angle the body should only creep";
  EXPECT_GT(running, 0.5) << "above the friction angle it should be sliding freely";
  EXPECT_GT(running / holding, 50.0) << "the two regimes should be unmistakable";
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
