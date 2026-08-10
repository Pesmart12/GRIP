#include <functional>
#include <vector>

#include <Eigen/LU>
#include <gtest/gtest.h>

#include "contact/half_plane.hpp"
#include "contact/penalty.hpp"
#include "core/rigid_body.hpp"
#include "dynamics/integrator.hpp"
#include "dynamics/mass.hpp"
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

  const StepJacobians analytic = step_body_jacobian(state, params, BodyShape{}, HalfPlane{}, PenaltyParams{}, dt, gravity);

  const std::function<StateVector(const StateVector&)> step_from_state = [&](const StateVector& x) {
    return Pack(step_body(Unpack(x), params, BodyShape{}, HalfPlane{}, PenaltyParams{}, u, dt, gravity));
  };
  const StateJacobian fd_dz_dz = testutil::CentralDifferenceJacobian<6, 6>(step_from_state, Pack(state));

  // Perturbing u and comparing against dz_df is a check on the claim that
  // makes them the same object: u enters additively, so df/du = Id and the
  // integrator never needs to see u at all.
  const std::function<StateVector(const Eigen::Vector3d&)> step_from_control = [&](const Eigen::Vector3d& u_perturbed) {
    return Pack(step_body(state, params, BodyShape{}, HalfPlane{}, PenaltyParams{}, u_perturbed, dt, gravity));
  };
  const ForceSensitivity fd_dz_du = testutil::CentralDifferenceJacobian<6, 3>(step_from_control, u);

  constexpr double kTol = 1e-6;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      EXPECT_NEAR(analytic.dz_dz(i, j), fd_dz_dz(i, j), kTol)
          << "dz_dz mismatch at (" << i << ", " << j << ")";
    }
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(analytic.dz_df(i, j), fd_dz_du(i, j), kTol)
          << "dz_df vs. finite-difference dz_du mismatch at (" << i << ", " << j << ")";
    }
  }
}

TEST(IntegratorJacobians, AssemblyMatchesIntegratingTheSummedForceByHand) {
  // step_body exists so callers cannot evaluate the force and its Jacobian at
  // different states. It must be exactly the two-step version and nothing
  // more -- if it ever grows a term of its own, this catches it.
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.2, 0.3);
  state.v = Eigen::Vector3d(0.5, -1.4, 0.8);
  const RigidBodyParams params{/*mass=*/1.0, /*inertia=*/1.0 / 6.0};
  const BodyShape shape{{{-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}}};
  const HalfPlane ground;
  const PenaltyParams penalty{/*stiffness=*/100.0, /*damping=*/12.0};
  const Eigen::Vector3d u(0.3, -0.7, 0.1);
  const double dt = 1.0e-3;

  const Eigen::Vector3d force = gravity_force(params, kDefaultGravity) + penalty_force_body(state, shape, ground, penalty) + u;
  const RigidBodyState by_hand = integrate_body(state, params, force, dt);
  const RigidBodyState assembled = step_body(state, params, shape, ground, penalty, u, dt, kDefaultGravity);

  EXPECT_TRUE(assembled.q.isApprox(by_hand.q, 0.0));
  EXPECT_TRUE(assembled.v.isApprox(by_hand.v, 0.0));

  ForceJacobian summed = gravity_force_jacobian(state.q, state.v, params, kDefaultGravity);
  const ForceJacobian contact = penalty_force_body_jacobian(state, shape, ground, penalty);
  summed.df_dq += contact.df_dq;
  summed.df_dv += contact.df_dv;

  const StepJacobians by_hand_jac = integrate_body_jacobian(params, summed, dt);
  const StepJacobians assembled_jac = step_body_jacobian(state, params, shape, ground, penalty, dt, kDefaultGravity);

  EXPECT_TRUE(assembled_jac.dz_dz.isApprox(by_hand_jac.dz_dz, 0.0));
  EXPECT_TRUE(assembled_jac.dz_df.isApprox(by_hand_jac.dz_df, 0.0));
}

TEST(IntegratorJacobians, DeterminantIsOneForAnyForceWithNoVelocityDependence) {
  // det(dz_dz) = det(Id + dt*M^-1*df/dv), so df/dq cancels out entirely and
  // any force with df/dv = 0 gives determinant exactly 1.
  //
  // Now that the Jacobian takes a ForceJacobian instead of a state, this can
  // be asserted for ARBITRARY df/dq rather than at whatever operating points
  // a particular force law happens to produce. Including asymmetric ones --
  // the cancellation never needed df/dq to be a Hessian.
  const double dt = 0.03;

  std::vector<Eigen::Matrix3d> df_dq_cases;
  df_dq_cases.push_back(Eigen::Matrix3d::Zero());
  Eigen::Matrix3d symmetric;
  symmetric << -12.0, 3.5, -1.0, 3.5, -8.0, 2.25, -1.0, 2.25, -19.0;
  df_dq_cases.push_back(symmetric);
  Eigen::Matrix3d asymmetric;
  asymmetric << 4.0, -17.0, 6.5, 2.0, -3.0, 11.0, -8.5, 0.25, 13.0;
  df_dq_cases.push_back(asymmetric);

  for (const RigidBodyParams& params : {RigidBodyParams{1.0, 1.0}, RigidBodyParams{3.7, 0.2}}) {
    for (const Eigen::Matrix3d& df_dq : df_dq_cases) {
      ForceJacobian force_jac;
      force_jac.df_dq = df_dq;  // df_dv stays zero

      const StepJacobians jac = integrate_body_jacobian(params, force_jac, dt);

      EXPECT_NEAR(jac.dz_dz.determinant(), 1.0, 1e-14) << "mass " << params.mass << ", df_dq row 0 " << df_dq.row(0);
    }
  }
}

TEST(IntegratorJacobians, DeterminantTracksTheVelocityBlockExactly) {
  // The general identity, asserted directly rather than through a force law:
  // whatever df/dv is, the determinant is det(Id + dt*M^-1*df/dv), and df/dq
  // has no say in it.
  const double dt = 0.03;
  const RigidBodyParams params{/*mass=*/2.5, /*inertia=*/0.4};
  const Eigen::Matrix3d m_inv = inverse_mass_diagonal(params).asDiagonal();

  Eigen::Matrix3d df_dq;
  df_dq << 4.0, -17.0, 6.5, 2.0, -3.0, 11.0, -8.5, 0.25, 13.0;

  std::vector<Eigen::Matrix3d> df_dv_cases;
  df_dv_cases.push_back(Eigen::Matrix3d::Zero());
  Eigen::Matrix3d dissipative;  // symmetric negative semidefinite, like a damper
  dissipative << -6.0, 1.5, 0.0, 1.5, -4.0, -0.5, 0.0, -0.5, -2.0;
  df_dv_cases.push_back(dissipative);
  Eigen::Matrix3d skewed;
  skewed << -1.0, 5.0, -2.0, 0.0, -3.0, 1.0, 4.0, -1.0, -7.0;
  df_dv_cases.push_back(skewed);

  for (const Eigen::Matrix3d& df_dv : df_dv_cases) {
    ForceJacobian force_jac;
    force_jac.df_dq = df_dq;
    force_jac.df_dv = df_dv;

    const StepJacobians jac = integrate_body_jacobian(params, force_jac, dt);
    const Eigen::Matrix3d predicted = Eigen::Matrix3d::Identity() + dt * m_inv * df_dv;

    EXPECT_NEAR(jac.dz_dz.determinant(), predicted.determinant(), 1e-14) << "df_dv row 0 " << df_dv.row(0);
  }
}

}  // namespace
}  // namespace grip
