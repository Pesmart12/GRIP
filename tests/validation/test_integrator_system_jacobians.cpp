#include <cstddef>
#include <functional>
#include <vector>

#include <gtest/gtest.h>

#include "core/rigid_body.hpp"
#include "dynamics/integrator.hpp"
#include "utils/finite_difference.hpp"

namespace grip {
namespace {

// Test-local flatten/unflatten for the control input -- not production
// code. The production system step takes std::vector<Eigen::Vector3d>
// directly; a flat vector is only needed here so the FD helper (which
// operates on R^n -> R^m) can perturb it component-wise.
Eigen::VectorXd FlattenControls(const std::vector<Eigen::Vector3d>& u) {
  Eigen::VectorXd flat(3 * static_cast<Eigen::Index>(u.size()));
  for (std::size_t i = 0; i < u.size(); ++i) {
    flat.segment<3>(static_cast<Eigen::Index>(3 * i)) = u[i];
  }
  return flat;
}

std::vector<Eigen::Vector3d> UnflattenControls(const Eigen::VectorXd& flat, std::size_t num_bodies) {
  std::vector<Eigen::Vector3d> u(num_bodies);
  for (std::size_t i = 0; i < num_bodies; ++i) {
    u[i] = flat.segment<3>(static_cast<Eigen::Index>(3 * i));
  }
  return u;
}

TEST(SystemJacobians, MatchCentralFiniteDifference) {
  std::vector<RigidBodyState> states(2);
  states[0].q = Eigen::Vector3d(0.3, -0.2, 0.7);
  states[0].v = Eigen::Vector3d(1.1, -0.4, 0.9);
  states[1].q = Eigen::Vector3d(-0.5, 0.8, -0.1);
  states[1].v = Eigen::Vector3d(0.2, 0.6, -0.3);

  std::vector<RigidBodyParams> params(2);
  params[0] = RigidBodyParams{2.3, 0.6};
  params[1] = RigidBodyParams{0.9, 1.4};

  const std::vector<Eigen::Vector3d> u = {Eigen::Vector3d(0.5, -0.3, 0.2), Eigen::Vector3d(-0.4, 0.1, 0.6)};
  const double dt = 0.05;
  const double gravity = kDefaultGravity;
  const std::size_t num_bodies = states.size();

  const SystemStepJacobians analytic = step_system_jacobian(states, params, u, dt, gravity);

  const std::function<Eigen::VectorXd(const Eigen::VectorXd&)> step_from_state = [&](const Eigen::VectorXd& x) {
    return PackSystem(step_system(UnpackSystem(x, num_bodies), params, u, dt, gravity));
  };
  const Eigen::MatrixXd fd_dX_dX = testutil::CentralDifferenceJacobianXd(step_from_state, PackSystem(states));

  const std::function<Eigen::VectorXd(const Eigen::VectorXd&)> step_from_control = [&](const Eigen::VectorXd& flat_u) {
    return PackSystem(step_system(states, params, UnflattenControls(flat_u, num_bodies), dt, gravity));
  };
  const Eigen::MatrixXd fd_dX_dU = testutil::CentralDifferenceJacobianXd(step_from_control, FlattenControls(u));

  constexpr double kTol = 1e-6;
  ASSERT_EQ(analytic.dX_dX.rows(), fd_dX_dX.rows());
  ASSERT_EQ(analytic.dX_dX.cols(), fd_dX_dX.cols());
  ASSERT_EQ(analytic.dX_dU.rows(), fd_dX_dU.rows());
  ASSERT_EQ(analytic.dX_dU.cols(), fd_dX_dU.cols());

  for (Eigen::Index i = 0; i < analytic.dX_dX.rows(); ++i) {
    for (Eigen::Index j = 0; j < analytic.dX_dX.cols(); ++j) {
      EXPECT_NEAR(analytic.dX_dX(i, j), fd_dX_dX(i, j), kTol)
          << "dX_dX mismatch at (" << i << ", " << j << ")";
    }
    for (Eigen::Index j = 0; j < analytic.dX_dU.cols(); ++j) {
      EXPECT_NEAR(analytic.dX_dU(i, j), fd_dX_dU(i, j), kTol)
          << "dX_dU mismatch at (" << i << ", " << j << ")";
    }
  }
}

TEST(SystemJacobians, OffDiagonalBlocksAreExactlyZero) {
  // No coupling between bodies yet, so body i's output must not depend
  // on body j's state/control at all -- not approximately, exactly.
  std::vector<RigidBodyState> states(2);
  states[0].q = Eigen::Vector3d(1.0, 2.0, 3.0);
  states[1].q = Eigen::Vector3d(-1.0, 0.5, -0.2);
  const std::vector<RigidBodyParams> params(2, RigidBodyParams{1.2, 0.8});
  const std::vector<Eigen::Vector3d> u(2, Eigen::Vector3d::Zero());
  const double dt = 0.02;

  const SystemStepJacobians jac = step_system_jacobian(states, params, u, dt, kDefaultGravity);

  const Eigen::MatrixXd off_diag_01 = jac.dX_dX.block<6, 6>(0, 6);
  const Eigen::MatrixXd off_diag_10 = jac.dX_dX.block<6, 6>(6, 0);
  const Eigen::MatrixXd off_diag_control_01 = jac.dX_dU.block<6, 3>(0, 3);
  const Eigen::MatrixXd off_diag_control_10 = jac.dX_dU.block<6, 3>(6, 0);

  EXPECT_TRUE(off_diag_01.isZero(0.0));
  EXPECT_TRUE(off_diag_10.isZero(0.0));
  EXPECT_TRUE(off_diag_control_01.isZero(0.0));
  EXPECT_TRUE(off_diag_control_10.isZero(0.0));
}

}  // namespace
}  // namespace grip
