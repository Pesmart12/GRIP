#include <functional>

#include <Eigen/LU>
#include <gtest/gtest.h>

#include "contact/half_plane.hpp"
#include "contact/penalty.hpp"
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

  const StepJacobians analytic = step_body_jacobian(state, params, BodyShape{}, HalfPlane{}, PenaltyParams{}, u, dt, gravity);

  const std::function<StateVector(const StateVector&)> step_from_state = [&](const StateVector& x) {
    return Pack(step_body(Unpack(x), params, BodyShape{}, HalfPlane{}, PenaltyParams{}, u, dt, gravity));
  };
  const StateJacobian fd_dz_dz = testutil::CentralDifferenceJacobian<6, 6>(step_from_state, Pack(state));

  const std::function<StateVector(const Eigen::Vector3d&)> step_from_control = [&](const Eigen::Vector3d& u_perturbed) {
    return Pack(step_body(state, params, BodyShape{}, HalfPlane{}, PenaltyParams{}, u_perturbed, dt, gravity));
  };
  const ControlJacobian fd_dz_du = testutil::CentralDifferenceJacobian<6, 3>(step_from_control, u);

  constexpr double kTol = 1e-6;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      EXPECT_NEAR(analytic.dz_dz(i, j), fd_dz_dz(i, j), kTol)
          << "dz_dz mismatch at (" << i << ", " << j << ")";
    }
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(analytic.dz_du(i, j), fd_dz_du(i, j), kTol)
          << "dz_du mismatch at (" << i << ", " << j << ")";
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

  const StepJacobians jac = step_body_jacobian(state, params, BodyShape{}, HalfPlane{}, PenaltyParams{}, u, dt, kDefaultGravity);

  const Eigen::Matrix3d dv_dq_block = jac.dz_dz.block<3, 3>(3, 0);
  EXPECT_TRUE(dv_dq_block.isZero(0.0));
}

TEST(IntegratorJacobians, ConservativeStepPreservesPhaseSpaceVolume) {
  // A symplectic map preserves phase-space volume, so its Jacobian has
  // determinant exactly 1. Block-triangular algebra gives this for any
  // force law with df/dv = 0, not just gravity: the h^2*M^-1*df/dq term
  // in the position row cancels against the velocity row's contribution.
  //
  // This checks the integrator's *structure* rather than its numbers --
  // a different failure mode from the finite-difference tests above.
  // It gains teeth at step 5 (holds for an undamped contact spring) and
  // should correctly stop holding once damping is added, since
  // dissipation contracts phase-space volume. See
  // docs/derivations/integrator_jacobians.md.
  const double dt = 0.03;

  RigidBodyState state;
  state.q = Eigen::Vector3d(0.4, -1.2, 0.9);
  state.v = Eigen::Vector3d(-0.7, 0.3, 1.4);

  // Determinant is 1 regardless of body parameters, timestep, or
  // operating point -- vary them to document that invariance.
  for (const RigidBodyParams& params : {RigidBodyParams{1.0, 1.0}, RigidBodyParams{3.7, 0.2}}) {
    const StepJacobians jac = step_body_jacobian(state, params, BodyShape{}, HalfPlane{}, PenaltyParams{}, Eigen::Vector3d(0.5, -0.2, 0.8), dt, kDefaultGravity);
    EXPECT_NEAR(jac.dz_dz.determinant(), 1.0, 1e-14) << "mass " << params.mass << ", inertia " << params.inertia;
  }
}

}  // namespace
}  // namespace grip
