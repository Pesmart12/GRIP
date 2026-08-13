#include <cmath>
#include <cstddef>
#include <functional>
#include <numbers>
#include <vector>

#include <Eigen/LU>
#include <gtest/gtest.h>

#include "contact/detection.hpp"
#include "contact/half_plane.hpp"
#include "contact/penalty.hpp"
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

  const std::vector<BodyShape> shapes(2);  // no vertices: free flight
  const std::vector<Eigen::Vector3d> u = {Eigen::Vector3d(0.5, -0.3, 0.2), Eigen::Vector3d(-0.4, 0.1, 0.6)};
  const double dt = 0.05;
  const double gravity = kDefaultGravity;
  const std::size_t num_bodies = states.size();

  const SystemStepJacobians analytic = step_system_jacobian(states, params, shapes, HalfPlane{}, PenaltyParams{}, dt, gravity);

  const std::function<Eigen::VectorXd(const Eigen::VectorXd&)> step_from_state = [&](const Eigen::VectorXd& x) {
    return PackSystem(step_system(UnpackSystem(x, num_bodies), params, shapes, HalfPlane{}, PenaltyParams{}, u, dt, gravity));
  };
  const Eigen::MatrixXd fd_dZ_dZ = testutil::CentralDifferenceJacobianXd(step_from_state, PackSystem(states));

  const std::function<Eigen::VectorXd(const Eigen::VectorXd&)> step_from_control = [&](const Eigen::VectorXd& flat_u) {
    return PackSystem(step_system(states, params, shapes, HalfPlane{}, PenaltyParams{}, UnflattenControls(flat_u, num_bodies), dt, gravity));
  };
  const Eigen::MatrixXd fd_dZ_dU = testutil::CentralDifferenceJacobianXd(step_from_control, FlattenControls(u));

  constexpr double kTol = 1e-6;
  ASSERT_EQ(analytic.dZ_dZ.rows(), fd_dZ_dZ.rows());
  ASSERT_EQ(analytic.dZ_dZ.cols(), fd_dZ_dZ.cols());
  ASSERT_EQ(analytic.dZ_dF.rows(), fd_dZ_dU.rows());
  ASSERT_EQ(analytic.dZ_dF.cols(), fd_dZ_dU.cols());

  for (Eigen::Index i = 0; i < analytic.dZ_dZ.rows(); ++i) {
    for (Eigen::Index j = 0; j < analytic.dZ_dZ.cols(); ++j) {
      EXPECT_NEAR(analytic.dZ_dZ(i, j), fd_dZ_dZ(i, j), kTol)
          << "dZ_dZ mismatch at (" << i << ", " << j << ")";
    }
    // dZ_dF against a finite difference in U: the two coincide because each
    // body's control enters its own force additively, so dF/dU = Id.
    for (Eigen::Index j = 0; j < analytic.dZ_dF.cols(); ++j) {
      EXPECT_NEAR(analytic.dZ_dF(i, j), fd_dZ_dU(i, j), kTol)
          << "dZ_dF mismatch at (" << i << ", " << j << ")";
    }
  }
}

TEST(SystemJacobians, BodiesOutOfReachDoNotCouple) {
  // Bodies that touch only static scenery still produce an exactly
  // block-diagonal system Jacobian. This used to be a claim about the
  // assembly, which never wrote off-diagonal entries; now the assembly
  // is fully general and the zeros come from dF/dQ having none, which is
  // a statement about the physics rather than the code.
  std::vector<RigidBodyState> states(2);
  states[0].q = Eigen::Vector3d(1.0, 2.0, 3.0);
  states[1].q = Eigen::Vector3d(-1.0, 0.5, -0.2);
  const std::vector<RigidBodyParams> params(2, RigidBodyParams{1.2, 0.8});
  const std::vector<BodyShape> shapes(2);  // no vertices: free flight
  const std::vector<Eigen::Vector3d> u(2, Eigen::Vector3d::Zero());
  const double dt = 0.02;

  const SystemStepJacobians jac = step_system_jacobian(states, params, shapes, HalfPlane{}, PenaltyParams{}, dt, kDefaultGravity);

  const Eigen::MatrixXd off_diag_01 = jac.dZ_dZ.block<6, 6>(0, 6);
  const Eigen::MatrixXd off_diag_10 = jac.dZ_dZ.block<6, 6>(6, 0);
  const Eigen::MatrixXd off_diag_control_01 = jac.dZ_dF.block<6, 3>(0, 3);
  const Eigen::MatrixXd off_diag_control_10 = jac.dZ_dF.block<6, 3>(6, 0);

  EXPECT_TRUE(off_diag_01.isZero(0.0));
  EXPECT_TRUE(off_diag_10.isZero(0.0));
  EXPECT_TRUE(off_diag_control_01.isZero(0.0));
  EXPECT_TRUE(off_diag_control_10.isZero(0.0));
}

TEST(SystemJacobians, ConservativeStepPreservesPhaseSpaceVolume) {
  // Block-diagonal, so the system determinant is the product of the
  // per-body determinants -- each exactly 1 for a conservative force,
  // hence 1 overall regardless of body count. The system-level
  // counterpart of the same structural check in
  // test_integrator_jacobians.cpp.
  std::vector<RigidBodyState> states(3);
  states[0].q = Eigen::Vector3d(0.4, -1.2, 0.9);
  states[1].q = Eigen::Vector3d(-0.3, 0.7, -1.5);
  states[2].v = Eigen::Vector3d(1.1, -0.6, 0.2);

  std::vector<RigidBodyParams> params(3);
  params[0] = RigidBodyParams{1.0, 1.0};
  params[1] = RigidBodyParams{3.7, 0.2};
  params[2] = RigidBodyParams{0.4, 2.9};

  const std::vector<BodyShape> shapes(3);  // no vertices: free flight
  const std::vector<Eigen::Vector3d> u(3, Eigen::Vector3d(0.5, -0.2, 0.8));

  const SystemStepJacobians jac = step_system_jacobian(states, params, shapes, HalfPlane{}, PenaltyParams{}, 0.03, kDefaultGravity);

  EXPECT_NEAR(jac.dZ_dZ.determinant(), 1.0, 1e-14);
}

// A wide base with a narrower box resting on it, tilted a little off
// square so the pair sits clear of the reference-face tie that exact
// parallel lands on -- see
// PairDetection.ParallelFacesSitExactlyOnTheReferenceFlip.
BodyShape Platform() {
  return BodyShape{{{-1.0, -0.25}, {1.0, -0.25}, {1.0, 0.25}, {-1.0, 0.25}}};
}

BodyShape UnitSquare() {
  return BodyShape{{{-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}}};
}

std::vector<RigidBodyState> StackedPair() {
  std::vector<RigidBodyState> states(2);
  states[0].q = Eigen::Vector3d(0.0, 0.0, 0.02);
  states[0].v = Eigen::Vector3d(0.3, -0.2, 0.15);
  states[1].q = Eigen::Vector3d(0.0, 0.65, 0.0);
  states[1].v = Eigen::Vector3d(-0.1, -0.4, 0.25);
  return states;
}

TEST(SystemJacobians, ContactingBodiesCoupleOffDiagonally) {
  // The first configuration in the project where one body's motion
  // changes another's. multi_body_system.md has predicted this since
  // step 3 and every test until now has asserted its absence.
  const std::vector<RigidBodyState> states = StackedPair();
  const std::vector<RigidBodyParams> params(2, RigidBodyParams{1.0, 1.0 / 6.0});
  const std::vector<BodyShape> shapes = {Platform(), UnitSquare()};
  const PenaltyParams penalty{/*stiffness=*/500.0, /*damping=*/20.0};

  // The premise: they are actually in contact.
  ASSERT_FALSE(detect_contacts_pair(states[0], shapes[0], states[1], shapes[1]).empty());

  const SystemStepJacobians jac = step_system_jacobian(states, params, shapes, HalfPlane{}, penalty, 1.0e-3, kDefaultGravity);

  const Eigen::MatrixXd off_diagonal = jac.dZ_dZ.block<6, 6>(0, 6);
  EXPECT_FALSE(off_diagonal.isZero(0.0)) << "contacting bodies must couple";

  // Force still only moves its own body, whatever produced it.
  const Eigen::MatrixXd off_diagonal_force = jac.dZ_dF.block<6, 3>(0, 3);
  EXPECT_TRUE(off_diagonal_force.isZero(0.0)) << "dZ_dF stays block-diagonal";
}

TEST(SystemJacobians, MatchCentralFiniteDifferenceWithBodyBodyContact) {
  // The end-to-end check on everything step 9 added: the pair Jacobian,
  // the gap Hessian, the system force assembly, and the coupled chain
  // rule. Any one of them wrong shows up here.
  const std::vector<RigidBodyState> states = StackedPair();
  std::vector<RigidBodyParams> params(2);
  params[0] = RigidBodyParams{2.0, 0.5};
  params[1] = RigidBodyParams{1.0, 1.0 / 6.0};
  const std::vector<BodyShape> shapes = {Platform(), UnitSquare()};
  const HalfPlane ground;
  const PenaltyParams penalty{/*stiffness=*/500.0, /*damping=*/20.0};
  const std::vector<Eigen::Vector3d> u = {Eigen::Vector3d(0.4, -0.2, 0.1), Eigen::Vector3d(-0.3, 0.5, -0.2)};
  const double dt = 1.0e-3;
  const std::size_t num_bodies = states.size();

  const SystemStepJacobians analytic = step_system_jacobian(states, params, shapes, ground, penalty, dt, kDefaultGravity);

  const std::function<Eigen::VectorXd(const Eigen::VectorXd&)> step_from_state = [&](const Eigen::VectorXd& x) {
    return PackSystem(step_system(UnpackSystem(x, num_bodies), params, shapes, ground, penalty, u, dt, kDefaultGravity));
  };
  const Eigen::MatrixXd fd = testutil::CentralDifferenceJacobianXd(step_from_state, PackSystem(states));

  for (Eigen::Index i = 0; i < analytic.dZ_dZ.rows(); ++i) {
    for (Eigen::Index j = 0; j < analytic.dZ_dZ.cols(); ++j) {
      EXPECT_NEAR(analytic.dZ_dZ(i, j), fd(i, j), 1e-6) << "dZ_dZ mismatch at (" << i << ", " << j << ")";
    }
  }
}

TEST(SystemJacobians, MatchCentralFiniteDifferenceWithBodyBodyFriction) {
  // The same end-to-end check with the cone switched on, so the slip
  // Jacobian and its gradient are in the loop as well. Sliding rather
  // than sticking, to exercise the branch where beta's whole state
  // dependence runs through the normal force.
  std::vector<RigidBodyState> states = StackedPair();
  states[1].v = Eigen::Vector3d(3.0, -0.4, 0.25);  // sliding hard across the base
  std::vector<RigidBodyParams> params(2);
  params[0] = RigidBodyParams{2.0, 0.5};
  params[1] = RigidBodyParams{1.0, 1.0 / 6.0};
  const std::vector<BodyShape> shapes = {Platform(), UnitSquare()};
  const HalfPlane ground;
  const PenaltyParams penalty{/*stiffness=*/500.0, /*damping=*/20.0, /*slip_damping=*/40.0, /*friction=*/0.5};
  const std::vector<Eigen::Vector3d> u = {Eigen::Vector3d(0.4, -0.2, 0.1), Eigen::Vector3d(-0.3, 0.5, -0.2)};
  const double dt = 1.0e-3;
  const std::size_t num_bodies = states.size();

  const SystemStepJacobians analytic = step_system_jacobian(states, params, shapes, ground, penalty, dt, kDefaultGravity);

  const std::function<Eigen::VectorXd(const Eigen::VectorXd&)> step_from_state = [&](const Eigen::VectorXd& x) {
    return PackSystem(step_system(UnpackSystem(x, num_bodies), params, shapes, ground, penalty, u, dt, kDefaultGravity));
  };
  const Eigen::MatrixXd fd = testutil::CentralDifferenceJacobianXd(step_from_state, PackSystem(states));

  for (Eigen::Index i = 0; i < analytic.dZ_dZ.rows(); ++i) {
    for (Eigen::Index j = 0; j < analytic.dZ_dZ.cols(); ++j) {
      EXPECT_NEAR(analytic.dZ_dZ(i, j), fd(i, j), 1e-6) << "dZ_dZ mismatch at (" << i << ", " << j << ")";
    }
  }
}

TEST(SystemContactForces, PairFrictionHoldsBelowTheFrictionAngleAndReleasesAbove) {
  // The pair analogue of the plane's slope test, and a stronger one:
  // nothing here is axis-aligned, so the normal, the slip direction, the
  // cone and the coupling all have to be right together.
  //
  // The base is given an enormous mass and its weight cancelled by a
  // control wrench, so it stands in for immovable scenery while still
  // being an ordinary body with an ordinary contact.
  const double base_mass = 1.0e6;
  const std::vector<RigidBodyParams> params = {RigidBodyParams{base_mass, base_mass}, RigidBodyParams{1.0, 1.0 / 6.0}};
  const std::vector<BodyShape> shapes = {Platform(), UnitSquare()};
  const PenaltyParams penalty{/*stiffness=*/1.0e4, /*damping=*/50.0, /*slip_damping=*/1000.0, /*friction=*/0.5};
  const double gravity = 9.81;
  const double dt = 1.0e-4;
  const int steps = 5000;

  HalfPlane far_away;
  far_away.offset = -100.0;

  const auto slide_speed_after = [&](double degrees) {
    const double theta = degrees * std::numbers::pi / 180.0;
    // Square resting square-on against the tilted top face, a hair into it.
    const Eigen::Vector2d normal(-std::sin(theta), std::cos(theta));
    const Eigen::Vector2d seat = 0.749 * normal;

    std::vector<RigidBodyState> states(2);
    states[0].q = Eigen::Vector3d(0.0, 0.0, theta);
    states[1].q = Eigen::Vector3d(seat.x(), seat.y(), theta);

    const std::vector<Eigen::Vector3d> u = {Eigen::Vector3d(0.0, base_mass * gravity, 0.0), Eigen::Vector3d::Zero()};
    for (int i = 0; i < steps; ++i) {
      states = step_system(states, params, shapes, far_away, penalty, u, dt, gravity);
    }
    return states[1].v.head<2>().norm();
  };

  // mu = 0.5 gives a friction angle of atan(0.5) = 26.6 degrees.
  const double holding = slide_speed_after(20.0);
  const double running = slide_speed_after(35.0);

  EXPECT_LT(holding, 2.0e-2) << "below the friction angle the box should only creep";
  EXPECT_GT(running, 0.4) << "above it the box should be sliding down the face";
  EXPECT_GT(running / holding, 20.0) << "the two regimes should be unmistakable";
}

TEST(SystemContactForces, ObeyNewtonsThirdLaw) {
  // A pair contact applies equal and opposite forces at a single point,
  // so it can change neither the system's linear momentum nor its
  // angular momentum about any fixed origin.
  //
  // The per-body torques do NOT cancel on their own -- each is taken
  // about its own centre of mass -- so the angular check has to carry
  // the c x f terms that convert them to a common origin.
  const std::vector<RigidBodyState> states = StackedPair();
  const std::vector<BodyShape> shapes = {Platform(), UnitSquare()};
  const PenaltyParams penalty{/*stiffness=*/500.0, /*damping=*/20.0};

  // No plane, so the only forces are the pair contact's.
  HalfPlane far_away;
  far_away.offset = -100.0;
  const std::vector<Eigen::Vector3d> forces = penalty_forces_system(states, shapes, far_away, penalty);

  ASSERT_EQ(forces.size(), 2u);
  EXPECT_NEAR(forces[0].x() + forces[1].x(), 0.0, 1e-12);
  EXPECT_NEAR(forces[0].y() + forces[1].y(), 0.0, 1e-12);
  EXPECT_FALSE(forces[0].isZero(0.0)) << "the premise is that they are in contact";

  double angular = 0.0;
  for (std::size_t i = 0; i < forces.size(); ++i) {
    const Eigen::Vector2d centre = states[i].q.head<2>();
    angular += centre.x() * forces[i].y() - centre.y() * forces[i].x() + forces[i].z();
  }
  EXPECT_NEAR(angular, 0.0, 1e-12);
}

}  // namespace
}  // namespace grip
