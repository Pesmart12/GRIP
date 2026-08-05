#include <vector>

#include <gtest/gtest.h>

#include "core/rigid_body.hpp"
#include "dynamics/integrator.hpp"

namespace grip {
namespace {

TEST(SystemStep, IndependentBodiesMatchOwnClosedForm) {
  // Three bodies, distinct mass/inertia/control, so each one's closed
  // form actually depends on its own parameters rather than being
  // indistinguishable free-fall. Reuses the recursion derived in
  // test_integrator.cpp: from rest under a constant force/torque a,
  // v_n = n*dt*a, x_n = dt^2*a*n(n+1)/2.
  std::vector<RigidBodyState> states(3);
  std::vector<RigidBodyParams> params(3);
  std::vector<Eigen::Vector3d> u(3, Eigen::Vector3d::Zero());

  params[0] = RigidBodyParams{/*mass=*/1.0, /*inertia=*/1.0};  // gravity only

  params[1] = RigidBodyParams{/*mass=*/2.0, /*inertia=*/0.5};
  u[1] = Eigen::Vector3d(3.0, 0.0, 0.0);  // horizontal wrench

  params[2] = RigidBodyParams{/*mass=*/0.5, /*inertia=*/2.0};
  u[2] = Eigen::Vector3d(0.0, 0.0, 1.0);  // torque only

  const double dt = 0.01;
  const double g = 9.81;
  const int n = 10;

  for (int i = 0; i < n; ++i) {
    states = step_system(states, params, u, dt, g);
  }

  const double gravity_vy = n * dt * (-g);
  const double gravity_qy = dt * dt * (-g) * (n * (n + 1) / 2.0);

  // All three bodies fell under the same gravity, independent of mass.
  EXPECT_NEAR(states[0].v.y(), gravity_vy, 1e-12);
  EXPECT_NEAR(states[0].q.y(), gravity_qy, 1e-12);
  EXPECT_NEAR(states[1].v.y(), gravity_vy, 1e-12);
  EXPECT_NEAR(states[1].q.y(), gravity_qy, 1e-12);
  EXPECT_NEAR(states[2].v.y(), gravity_vy, 1e-12);
  EXPECT_NEAR(states[2].q.y(), gravity_qy, 1e-12);

  // Body 1's horizontal motion uses its own mass (2.0).
  const double ax = u[1].x() / params[1].mass;
  EXPECT_NEAR(states[1].v.x(), n * dt * ax, 1e-12);
  EXPECT_NEAR(states[1].q.x(), dt * dt * ax * (n * (n + 1) / 2.0), 1e-12);

  // Body 2's rotation uses its own inertia (2.0).
  const double alpha = u[2].z() / params[2].inertia;
  EXPECT_NEAR(states[2].v.z(), n * dt * alpha, 1e-12);
  EXPECT_NEAR(states[2].q.z(), dt * dt * alpha * (n * (n + 1) / 2.0), 1e-12);
}

TEST(SystemStep, PerturbingOneBodyDoesNotAffectAnother) {
  // Regression target for an indexing bug (e.g. a wrong block offset):
  // body B's output must be bit-identical whether or not body A's
  // initial state is perturbed, since the bodies don't couple.
  const std::vector<RigidBodyParams> params(2, RigidBodyParams{1.5, 0.7});
  const std::vector<Eigen::Vector3d> u(2, Eigen::Vector3d(0.2, -0.1, 0.3));
  const double dt = 0.02;

  std::vector<RigidBodyState> baseline(2);
  baseline[0].q = Eigen::Vector3d(0.0, 1.0, 0.0);
  baseline[0].v = Eigen::Vector3d(0.5, 0.0, 0.1);
  baseline[1].q = Eigen::Vector3d(2.0, -1.0, 0.4);
  baseline[1].v = Eigen::Vector3d(-0.3, 0.2, 0.0);

  std::vector<RigidBodyState> perturbed = baseline;
  perturbed[0].q.x() += 0.1;  // only body 0 changes

  const std::vector<RigidBodyState> baseline_next = step_system(baseline, params, u, dt);
  const std::vector<RigidBodyState> perturbed_next = step_system(perturbed, params, u, dt);

  EXPECT_TRUE(baseline_next[1].q.isApprox(perturbed_next[1].q, 0.0));
  EXPECT_TRUE(baseline_next[1].v.isApprox(perturbed_next[1].v, 0.0));
  EXPECT_FALSE(baseline_next[0].q.isApprox(perturbed_next[0].q, 0.0));
}

}  // namespace
}  // namespace grip
