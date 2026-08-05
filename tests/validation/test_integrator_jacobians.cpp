#include <functional>

#include <gtest/gtest.h>

#include "core/rigid_body.hpp"
#include "dynamics/integrator.hpp"
#include "utils/finite_difference.hpp"

namespace grip {
namespace {

TEST(IntegratorJacobians, MatchCentralFiniteDifference) {
  // A generic, non-trivial operating point -- nonzero everywhere, mass
  // and inertia both != 1, so a bug that mixes up M^-1's two distinct
  // entries (mass vs. inertia) would show up.
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.3, -0.2, 0.7);
  state.v = Eigen::Vector3d(1.1, -0.4, 0.9);
  RigidBodyParams params;
  params.mass = 2.3;
  params.inertia = 0.6;
  const Eigen::Vector3d u(0.5, -0.3, 0.2);
  const double dt = 0.05;
  const double gravity = kDefaultGravity;

  const StepJacobians analytic =
      symplectic_euler_step_jacobian(state, params, u, dt, gravity);

  const std::function<StateVector(const StateVector&)> step_from_state =
      [&](const StateVector& x) {
        return Pack(symplectic_euler_step(Unpack(x), params, u, dt, gravity));
      };
  const StateJacobian fd_dx_dx =
      testutil::CentralDifferenceJacobian<6, 6>(step_from_state, Pack(state));

  const std::function<StateVector(const Eigen::Vector3d&)> step_from_control =
      [&](const Eigen::Vector3d& u_perturbed) {
        return Pack(
            symplectic_euler_step(state, params, u_perturbed, dt, gravity));
      };
  const ControlJacobian fd_dx_du =
      testutil::CentralDifferenceJacobian<6, 3>(step_from_control, u);

  constexpr double kTol = 1e-6;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      EXPECT_NEAR(analytic.dx_dx(i, j), fd_dx_dx(i, j), kTol)
          << "dx_dx mismatch at (" << i << ", " << j << ")";
    }
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(analytic.dx_du(i, j), fd_dx_du(i, j), kTol)
          << "dx_du mismatch at (" << i << ", " << j << ")";
    }
  }
}

TEST(IntegratorJacobians, GravityContributesZeroStateCoupling) {
  // Gravity is independent of q and v, so d(v_{t+1})/dq_t must be
  // exactly zero -- not just small. This is the trivial case flagged
  // in docs/derivations/integrator_jacobians.md as worth checking
  // explicitly, precisely because it should come out as an exact zero.
  RigidBodyState state;
  state.q = Eigen::Vector3d(1.0, 2.0, 3.0);
  state.v = Eigen::Vector3d(-1.0, 0.5, -0.2);
  RigidBodyParams params;
  const Eigen::Vector3d u = Eigen::Vector3d::Zero();
  const double dt = 0.02;

  const StepJacobians jac =
      symplectic_euler_step_jacobian(state, params, u, dt, kDefaultGravity);

  const Eigen::Matrix3d dv_dq_block = jac.dx_dx.block<3, 3>(3, 0);
  EXPECT_TRUE(dv_dq_block.isZero(0.0));
}

}  // namespace
}  // namespace grip
