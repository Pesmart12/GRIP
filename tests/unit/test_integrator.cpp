#include <gtest/gtest.h>

#include "contact/half_plane.hpp"
#include "contact/penalty.hpp"
#include "core/rigid_body.hpp"
#include "dynamics/integrator.hpp"

namespace grip {
namespace {

// These are free-flight tests: BodyShape{} has no vertices, so there are
// no contacts and the penalty term is identically zero regardless of the
// plane or the stiffness.
TEST(SymplecticEulerStep, GravityFromRestSingleStep) {
  RigidBodyState state;    // q = v = 0
  RigidBodyParams params;  // mass = inertia = 1
  const double dt = 0.01;
  const double g = 9.81;

  const RigidBodyState next = step_body(state, params, BodyShape{}, HalfPlane{}, PenaltyParams{}, Eigen::Vector3d::Zero(), dt, g);

  const double expected_vy = -dt * g;
  const double expected_qy = dt * expected_vy;  // = -dt^2 * g

  EXPECT_DOUBLE_EQ(next.v.y(), expected_vy);
  EXPECT_DOUBLE_EQ(next.q.y(), expected_qy);
  EXPECT_DOUBLE_EQ(next.v.x(), 0.0);
  EXPECT_DOUBLE_EQ(next.v.z(), 0.0);
  EXPECT_DOUBLE_EQ(next.q.x(), 0.0);
  EXPECT_DOUBLE_EQ(next.q.z(), 0.0);
}

TEST(SymplecticEulerStep, GravityFromRestMultiStepClosedForm) {
  RigidBodyState state;
  RigidBodyParams params;
  const double dt = 0.01;
  const double g = 9.81;
  const double a = -g;
  const int n = 10;

  for (int i = 0; i < n; ++i) {
    state = step_body(state, params, BodyShape{}, HalfPlane{}, PenaltyParams{}, Eigen::Vector3d::Zero(), dt, g);
  }

  // Recursion v_i = v_{i-1} + dt*a, q_i = q_{i-1} + dt*v_i, from rest,
  // has closed form v_n = n*dt*a, q_n = dt^2 * a * n*(n+1)/2. This is
  // NOT the continuous free-fall solution -- symplectic Euler's
  // position update carries an O(dt) offset from it even for constant
  // acceleration. See docs/derivations/symplectic_euler.md.
  const double expected_vy = n * dt * a;
  const double expected_qy = dt * dt * a * (n * (n + 1) / 2.0);

  EXPECT_NEAR(state.v.y(), expected_vy, 1e-12);
  EXPECT_NEAR(state.q.y(), expected_qy, 1e-12);
}

TEST(SymplecticEulerStep, ConstantWrenchDecouplesAcrossAxes) {
  RigidBodyState state;
  RigidBodyParams params;
  params.mass = 2.0;
  params.inertia = 0.5;
  const double dt = 0.01;
  const double gravity = 0.0;              // isolate the wrench
  const Eigen::Vector3d u(4.0, 0.0, 1.0);  // fx and tau only, no fy

  const RigidBodyState next = step_body(state, params, BodyShape{}, HalfPlane{}, PenaltyParams{}, u, dt, gravity);

  const double expected_vx = dt * (u.x() / params.mass);
  const double expected_omega = dt * (u.z() / params.inertia);

  EXPECT_DOUBLE_EQ(next.v.x(), expected_vx);
  EXPECT_DOUBLE_EQ(next.v.z(), expected_omega);
  EXPECT_DOUBLE_EQ(next.v.y(), 0.0);
  EXPECT_DOUBLE_EQ(next.q.x(), dt * expected_vx);
  EXPECT_DOUBLE_EQ(next.q.z(), dt * expected_omega);
  EXPECT_DOUBLE_EQ(next.q.y(), 0.0);
}

}  // namespace
}  // namespace grip
