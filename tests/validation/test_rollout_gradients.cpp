#include <algorithm>
#include <cstddef>
#include <functional>
#include <vector>

#include <gtest/gtest.h>

#include "contact/detection.hpp"
#include "contact/half_plane.hpp"
#include "contact/penalty.hpp"
#include "core/rigid_body.hpp"
#include "dynamics/integrator.hpp"
#include "gradient/rollout.hpp"
#include "utils/finite_difference.hpp"

namespace grip {
namespace {

BodyShape UnitSquare() {
  return BodyShape{{{-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}}};
}

// A deliberately linear objective: J = sum_t a_t . Z_t + sum_t b_t . U_t.
//
// Linear in the trajectory, not in the parameters -- the dynamics are still
// nonlinear once contact is active, so dJ/dZ0 genuinely depends on the path.
// What linearity buys is that the seeds ARE a_t and b_t, with no chain rule
// on the caller's side to get wrong, so a finite-difference disagreement can
// only be the adjoint's fault.
double Objective(const std::vector<std::vector<RigidBodyState>>& trajectory, const std::vector<std::vector<Eigen::Vector3d>>& controls, const std::vector<SystemStateVector>& dl_dZ, const std::vector<SystemControlVector>& dl_dU) {
  double total = 0.0;
  for (std::size_t t = 0; t < trajectory.size(); ++t) {
    total += dl_dZ[t].dot(PackSystem(trajectory[t]));
  }
  for (std::size_t t = 0; t < controls.size(); ++t) {
    total += dl_dU[t].dot(PackControls(controls[t]));
  }
  return total;
}

// Test-local flattening of a whole control sequence into one vector, so the
// finite-difference helper can perturb it component-wise. Not production
// shape -- rollout_system takes controls per step, per body.
Eigen::VectorXd FlattenSequence(const std::vector<std::vector<Eigen::Vector3d>>& controls) {
  const auto per_step = static_cast<Eigen::Index>(3 * (controls.empty() ? 0 : controls[0].size()));
  Eigen::VectorXd flat(per_step * static_cast<Eigen::Index>(controls.size()));
  for (std::size_t t = 0; t < controls.size(); ++t) {
    flat.segment(per_step * static_cast<Eigen::Index>(t), per_step) = PackControls(controls[t]);
  }
  return flat;
}

std::vector<std::vector<Eigen::Vector3d>> UnflattenSequence(const Eigen::VectorXd& flat, std::size_t horizon, std::size_t num_bodies) {
  const auto per_step = static_cast<Eigen::Index>(3 * num_bodies);
  std::vector<std::vector<Eigen::Vector3d>> controls(horizon);
  for (std::size_t t = 0; t < horizon; ++t) {
    controls[t] = UnpackControls(flat.segment(per_step * static_cast<Eigen::Index>(t), per_step), num_bodies);
  }
  return controls;
}

// Seeds that are nonzero at several interior steps as well as the terminal
// one, so the running-cost path is exercised rather than only the terminal
// case that a first implementation tends to get right by accident.
void MakeSeeds(std::size_t horizon, std::size_t num_bodies, std::vector<SystemStateVector>* dl_dZ, std::vector<SystemControlVector>* dl_dU) {
  const auto state_size = static_cast<Eigen::Index>(6 * num_bodies);
  const auto control_size = static_cast<Eigen::Index>(3 * num_bodies);

  dl_dZ->assign(horizon + 1, SystemStateVector::Zero(state_size));
  dl_dU->assign(horizon, SystemControlVector::Zero(control_size));

  for (std::size_t t = 0; t <= horizon; ++t) {
    for (Eigen::Index i = 0; i < state_size; ++i) {
      (*dl_dZ)[t](i) = (t % 3 == 0) ? 0.3 * static_cast<double>(i + 1) - 0.4 : 0.0;
    }
  }
  (*dl_dZ)[horizon].setConstant(1.25);  // a real terminal cost on top

  for (std::size_t t = 0; t < horizon; ++t) {
    for (Eigen::Index i = 0; i < control_size; ++i) {
      (*dl_dU)[t](i) = (t % 2 == 0) ? 0.1 * static_cast<double>(i) - 0.15 : 0.0;
    }
  }
}

void ExpectGradientsMatchFiniteDifference(const std::vector<RigidBodyState>& initial, const std::vector<RigidBodyParams>& params, const std::vector<BodyShape>& shapes, const HalfPlane& plane, const PenaltyParams& penalty, const std::vector<std::vector<Eigen::Vector3d>>& controls, double dt, double tolerance) {
  const std::size_t horizon = controls.size();
  const std::size_t num_bodies = initial.size();

  std::vector<SystemStateVector> dl_dZ;
  std::vector<SystemControlVector> dl_dU;
  MakeSeeds(horizon, num_bodies, &dl_dZ, &dl_dU);

  const std::vector<std::vector<RigidBodyState>> trajectory = rollout_system(initial, params, shapes, plane, penalty, controls, dt);
  const RolloutGradients analytic = adjoint_system(trajectory, params, shapes, plane, penalty, dl_dZ, dl_dU, dt);

  const std::function<Eigen::VectorXd(const Eigen::VectorXd&)> objective_of_initial = [&](const Eigen::VectorXd& z0) {
    const std::vector<std::vector<RigidBodyState>> perturbed = rollout_system(UnpackSystem(z0, num_bodies), params, shapes, plane, penalty, controls, dt);
    Eigen::VectorXd out(1);
    out(0) = Objective(perturbed, controls, dl_dZ, dl_dU);
    return out;
  };
  const Eigen::MatrixXd fd_dJ_dZ0 = testutil::CentralDifferenceJacobianXd(objective_of_initial, PackSystem(initial));

  ASSERT_EQ(fd_dJ_dZ0.cols(), analytic.dJ_dZ0.size());
  for (Eigen::Index i = 0; i < analytic.dJ_dZ0.size(); ++i) {
    EXPECT_NEAR(analytic.dJ_dZ0(i), fd_dJ_dZ0(0, i), tolerance) << "dJ_dZ0 mismatch at " << i;
  }

  const std::function<Eigen::VectorXd(const Eigen::VectorXd&)> objective_of_controls = [&](const Eigen::VectorXd& flat) {
    const std::vector<std::vector<Eigen::Vector3d>> perturbed_controls = UnflattenSequence(flat, horizon, num_bodies);
    const std::vector<std::vector<RigidBodyState>> perturbed = rollout_system(initial, params, shapes, plane, penalty, perturbed_controls, dt);
    Eigen::VectorXd out(1);
    out(0) = Objective(perturbed, perturbed_controls, dl_dZ, dl_dU);
    return out;
  };
  const Eigen::MatrixXd fd_dJ_dU = testutil::CentralDifferenceJacobianXd(objective_of_controls, FlattenSequence(controls));

  const auto per_step = static_cast<Eigen::Index>(3 * num_bodies);
  for (std::size_t t = 0; t < horizon; ++t) {
    for (Eigen::Index i = 0; i < per_step; ++i) {
      const Eigen::Index column = per_step * static_cast<Eigen::Index>(t) + i;
      EXPECT_NEAR(analytic.dJ_dU[t](i), fd_dJ_dU(0, column), tolerance) << "dJ_dU mismatch at step " << t << ", component " << i;
    }
  }
}

TEST(RolloutGradients, MatchCentralFiniteDifferenceInFreeFlight) {
  // No vertices, so no contact: the dynamics are linear and every Jacobian is
  // the same constant shear. Any disagreement here is pure index bookkeeping.
  std::vector<RigidBodyState> initial(2);
  initial[0].q = Eigen::Vector3d(0.3, 1.2, 0.4);
  initial[0].v = Eigen::Vector3d(-0.2, 0.5, 0.3);
  initial[1].q = Eigen::Vector3d(-0.7, 0.8, -0.2);
  initial[1].v = Eigen::Vector3d(0.4, -0.1, 0.6);

  std::vector<RigidBodyParams> params(2);
  params[0] = RigidBodyParams{2.3, 0.6};
  params[1] = RigidBodyParams{0.9, 1.4};
  const std::vector<BodyShape> shapes(2);

  std::vector<std::vector<Eigen::Vector3d>> controls;
  for (int t = 0; t < 12; ++t) {
    controls.push_back({Eigen::Vector3d(0.1 * t, -0.3, 0.2), Eigen::Vector3d(-0.2, 0.05 * t, -0.1)});
  }

  ExpectGradientsMatchFiniteDifference(initial, params, shapes, HalfPlane{}, PenaltyParams{}, controls, 0.02, 1.0e-6);
}

TEST(RolloutGradients, MatchCentralFiniteDifferenceInSustainedContact) {
  // Held against the plane by a downward wrench for the whole horizon, so the
  // contact set never changes and the finite difference is measuring the
  // gradient rather than the activation boundary. The boundary has its own
  // file; mixing them here would produce a test that fails for the right
  // reason and teaches nothing.
  const RigidBodyParams body{/*mass=*/1.0, /*inertia=*/1.0 / 6.0};
  const PenaltyParams penalty{/*stiffness=*/1.0e4, /*damping=*/50.0};
  const HalfPlane ground;
  const double dt = 1.0e-3;

  // Started at the static equilibrium for the applied load: with 100 N of
  // downward wrench on top of gravity, 2*k*(-d) = mg + 100 gives d = -5.5 mm.
  // Perched shallower than that and the spring launches it; perched deeper
  // and it does the same, harder. Sitting at equilibrium with a little
  // velocity keeps it oscillating inside contact for the whole horizon.
  std::vector<RigidBodyState> initial(1);
  initial[0].q = Eigen::Vector3d(0.0, 0.5 - 5.5e-3, 0.0);
  initial[0].v = Eigen::Vector3d(0.2, -0.1, 0.05);
  const std::vector<RigidBodyParams> params(1, body);
  const std::vector<BodyShape> shapes(1, UnitSquare());

  std::vector<std::vector<Eigen::Vector3d>> controls;
  for (int t = 0; t < 15; ++t) {
    controls.push_back({Eigen::Vector3d(0.5, -100.0, 0.3)});
  }

  // Confirm the premise: every step keeps at least one corner well clear of
  // d = 0 on the penetrating side, by far more than the finite-difference
  // step could move it.
  const std::vector<std::vector<RigidBodyState>> trajectory = rollout_system(initial, params, shapes, ground, penalty, controls, dt);
  for (std::size_t t = 0; t < trajectory.size(); ++t) {
    double deepest = 0.0;
    for (const Contact& contact : detect_contacts_body(trajectory[t][0], shapes[0], ground)) {
      deepest = std::min(deepest, contact.signed_distance);
    }
    ASSERT_LT(deepest, -1.0e-4) << "step " << t << " left contact; the premise of this test is gone";
  }

  ExpectGradientsMatchFiniteDifference(initial, params, shapes, ground, penalty, controls, dt, 1.0e-6);
}

TEST(RolloutGradients, FreeFlightStateGradientMatchesClosedForm) {
  // With gravity alone the step Jacobian is a constant shear, and shears
  // compose by adding their off-diagonal blocks:
  //
  //   (dz_dz)^H = [[Id, H*dt*Id], [0, Id]]
  //
  // so a full-horizon sweep can be checked against exact algebra rather than
  // against finite differences. Physically: perturbing the initial velocity
  // displaces the final position by exactly the elapsed time.
  const std::size_t horizon = 25;
  const double dt = 0.01;
  const std::vector<RigidBodyState> initial(1);
  const std::vector<RigidBodyParams> params(1, RigidBodyParams{2.3, 0.6});
  const std::vector<BodyShape> shapes(1);
  const std::vector<std::vector<Eigen::Vector3d>> controls(horizon, std::vector<Eigen::Vector3d>(1, Eigen::Vector3d(0.4, -0.2, 0.1)));

  const std::vector<std::vector<RigidBodyState>> trajectory = rollout_system(initial, params, shapes, HalfPlane{}, PenaltyParams{}, controls, dt);

  // Row i of dz_H/dz_0 is the initial-state gradient of the scalar (z_H)_i.
  Eigen::Matrix<double, 6, 6> full;
  for (int i = 0; i < 6; ++i) {
    std::vector<SystemStateVector> dl_dZ(horizon + 1, SystemStateVector::Zero(6));
    dl_dZ[horizon] = SystemStateVector::Unit(6, i);
    const std::vector<SystemControlVector> dl_dU(horizon, SystemControlVector::Zero(3));

    full.row(i) = adjoint_system(trajectory, params, shapes, HalfPlane{}, PenaltyParams{}, dl_dZ, dl_dU, dt).dJ_dZ0.transpose();
  }

  Eigen::Matrix<double, 6, 6> expected = Eigen::Matrix<double, 6, 6>::Identity();
  const Eigen::Matrix3d coupling = static_cast<double>(horizon) * dt * Eigen::Matrix3d::Identity();
  expected.block<3, 3>(0, 3) = coupling;

  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      EXPECT_NEAR(full(i, j), expected(i, j), 1.0e-13) << "at (" << i << ", " << j << ")";
    }
  }
}

TEST(RolloutGradients, FreeFlightControlGradientMatchesClosedForm) {
  // u_t perturbs v_{t+1} onward, so it moves the final position through
  // H - t remaining position updates:
  //
  //   d(q_H)/d(u_t) = dt^2 * (H - t) * M^-1
  //
  // which collapses to the single-step dz_df at t = H-1. An adjoint that
  // used adjoint_t instead of adjoint_{t+1} would be off by exactly one
  // factor here, which is why this is worth checking across the horizon
  // rather than at one step.
  const std::size_t horizon = 20;
  const double dt = 0.01;
  const RigidBodyParams body{/*mass=*/2.5, /*inertia=*/0.4};
  const std::vector<RigidBodyState> initial(1);
  const std::vector<RigidBodyParams> params(1, body);
  const std::vector<BodyShape> shapes(1);
  const std::vector<std::vector<Eigen::Vector3d>> controls(horizon, std::vector<Eigen::Vector3d>(1, Eigen::Vector3d::Zero()));

  const std::vector<std::vector<RigidBodyState>> trajectory = rollout_system(initial, params, shapes, HalfPlane{}, PenaltyParams{}, controls, dt);

  // Objective reads off the final y position.
  std::vector<SystemStateVector> dl_dZ(horizon + 1, SystemStateVector::Zero(6));
  dl_dZ[horizon] = SystemStateVector::Unit(6, 1);
  const std::vector<SystemControlVector> dl_dU(horizon, SystemControlVector::Zero(3));

  const RolloutGradients gradients = adjoint_system(trajectory, params, shapes, HalfPlane{}, PenaltyParams{}, dl_dZ, dl_dU, dt);

  for (std::size_t t = 0; t < horizon; ++t) {
    const double remaining = static_cast<double>(horizon - t);
    EXPECT_NEAR(gradients.dJ_dU[t](1), dt * dt * remaining / body.mass, 1.0e-14) << "step " << t;
    EXPECT_NEAR(gradients.dJ_dU[t](0), 0.0, 1.0e-15) << "step " << t;
    EXPECT_NEAR(gradients.dJ_dU[t](2), 0.0, 1.0e-15) << "step " << t;
  }

  // The earliest control has H times the influence of the last one: early
  // pushes have longer to act.
  EXPECT_NEAR(gradients.dJ_dU[0](1) / gradients.dJ_dU[horizon - 1](1), static_cast<double>(horizon), 1.0e-10);
}

TEST(RolloutGradients, AdjointDecouplesAcrossBodies) {
  // Bodies only touch static scenery, so dZ_dZ is block-diagonal and the
  // backward sweep never mixes them. Seeding one body's cost must leave the
  // other's gradient at an identical zero -- not small, zero.
  //
  // Body-body contact is what breaks this, and it breaks it here first.
  const std::size_t horizon = 10;
  const double dt = 1.0e-3;

  std::vector<RigidBodyState> initial(2);
  initial[0].q = Eigen::Vector3d(0.0, 0.45, 0.1);
  initial[1].q = Eigen::Vector3d(3.0, 0.45, -0.2);
  const std::vector<RigidBodyParams> params(2, RigidBodyParams{1.0, 1.0 / 6.0});
  const std::vector<BodyShape> shapes(2, UnitSquare());
  const HalfPlane ground;
  const PenaltyParams penalty{/*stiffness=*/1.0e4, /*damping=*/50.0};
  const std::vector<std::vector<Eigen::Vector3d>> controls(horizon, std::vector<Eigen::Vector3d>(2, Eigen::Vector3d::Zero()));

  const std::vector<std::vector<RigidBodyState>> trajectory = rollout_system(initial, params, shapes, ground, penalty, controls, dt);

  // Seed body 0 only.
  std::vector<SystemStateVector> dl_dZ(horizon + 1, SystemStateVector::Zero(12));
  dl_dZ[horizon].head<6>().setConstant(1.0);
  const std::vector<SystemControlVector> dl_dU(horizon, SystemControlVector::Zero(6));

  const RolloutGradients gradients = adjoint_system(trajectory, params, shapes, ground, penalty, dl_dZ, dl_dU, dt);

  EXPECT_FALSE(gradients.dJ_dZ0.head<6>().isZero(0.0)) << "body 0 should have picked up a gradient";
  EXPECT_TRUE(gradients.dJ_dZ0.tail<6>().isZero(0.0)) << "body 1 is uncoupled and must stay at zero";
  for (std::size_t t = 0; t < horizon; ++t) {
    EXPECT_TRUE(gradients.dJ_dU[t].tail<3>().isZero(0.0)) << "body 1 control gradient at step " << t;
  }
}

TEST(RolloutGradients, TerminalSeedAloneReproducesTheJacobianProduct) {
  // The adjoint is reverse-mode accumulation of the same per-step Jacobians
  // the integrator already reports, so a terminal-only sweep must agree with
  // forming the product by hand. Cross-checks the sweep against the thing it
  // is an optimisation of.
  const std::size_t horizon = 12;
  const double dt = 1.0e-3;

  std::vector<RigidBodyState> initial(1);
  initial[0].q = Eigen::Vector3d(0.0, 0.48, 0.12);
  initial[0].v = Eigen::Vector3d(0.3, -0.4, 0.2);
  const std::vector<RigidBodyParams> params(1, RigidBodyParams{1.0, 1.0 / 6.0});
  const std::vector<BodyShape> shapes(1, UnitSquare());
  const HalfPlane ground;
  const PenaltyParams penalty{/*stiffness=*/1.0e4, /*damping=*/50.0};
  const std::vector<std::vector<Eigen::Vector3d>> controls(horizon, std::vector<Eigen::Vector3d>(1, Eigen::Vector3d(0.0, -20.0, 0.0)));

  const std::vector<std::vector<RigidBodyState>> trajectory = rollout_system(initial, params, shapes, ground, penalty, controls, dt);

  Eigen::MatrixXd product = Eigen::MatrixXd::Identity(6, 6);
  for (std::size_t t = 0; t < horizon; ++t) {
    product = step_system_jacobian(trajectory[t], params, shapes, ground, penalty, dt).dZ_dZ * product;
  }

  for (int i = 0; i < 6; ++i) {
    std::vector<SystemStateVector> dl_dZ(horizon + 1, SystemStateVector::Zero(6));
    dl_dZ[horizon] = SystemStateVector::Unit(6, i);
    const std::vector<SystemControlVector> dl_dU(horizon, SystemControlVector::Zero(3));

    const RolloutGradients gradients = adjoint_system(trajectory, params, shapes, ground, penalty, dl_dZ, dl_dU, dt);

    for (int j = 0; j < 6; ++j) {
      EXPECT_NEAR(gradients.dJ_dZ0(j), product(i, j), 1.0e-12) << "row " << i << ", column " << j;
    }
  }
}

}  // namespace
}  // namespace grip
