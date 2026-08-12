#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <gtest/gtest.h>

#include "contact/detection.hpp"
#include "contact/half_plane.hpp"
#include "contact/penalty.hpp"
#include "core/rigid_body.hpp"
#include "dynamics/integrator.hpp"
#include "dynamics/mass.hpp"
#include "utils/finite_difference.hpp"

namespace grip {
namespace {

BodyShape UnitSquare() {
  return BodyShape{{{-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}}};
}

// Central-difference df_c/dq, by re-running the real force law at
// perturbed configurations.
Eigen::Matrix3d FiniteDifferenceForceJacobian(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty) {
  const std::function<Eigen::Vector3d(const Eigen::Vector3d&)> force_of_q = [&](const Eigen::Vector3d& q) {
    RigidBodyState perturbed = state;
    perturbed.q = q;
    return penalty_force_body(perturbed, shape, plane, penalty);
  };
  return testutil::CentralDifferenceJacobian<3, 3>(force_of_q, state.q);
}

// Central-difference df_c/dv. Only the damping term reads velocity, so
// this is identically zero until b != 0.
Eigen::Matrix3d FiniteDifferenceVelocityJacobian(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty) {
  const std::function<Eigen::Vector3d(const Eigen::Vector3d&)> force_of_v = [&](const Eigen::Vector3d& v) {
    RigidBodyState perturbed = state;
    perturbed.v = v;
    return penalty_force_body(perturbed, shape, plane, penalty);
  };
  return testutil::CentralDifferenceJacobian<3, 3>(force_of_v, state.v);
}

// Contacts must also be clear of the friction cone, for the same reason:
// straddling |b_slip*s| = mu*lambda measures the boundary, not the
// gradient. Returns the smallest relative margin so a test can assert
// which side of the cone it is exercising.
double SmallestConeMargin(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty) {
  const std::vector<Contact> contacts = detect_contacts_body(state, shape, plane);
  const std::vector<Eigen::RowVector3d> jacobians = detect_contacts_body_jacobian(state, shape, plane);
  const std::vector<Eigen::RowVector3d> perp = detect_contacts_body_perp_jacobian(state, shape, plane);

  double smallest = std::numeric_limits<double>::max();
  for (std::size_t i = 0; i < contacts.size(); ++i) {
    if (contacts[i].signed_distance >= 0.0) {
      continue;
    }
    const double normal_force = -penalty.stiffness * contacts[i].signed_distance - penalty.damping * jacobians[i].dot(state.v);
    if (normal_force <= 0.0) {
      continue;
    }
    const double demand = std::abs(penalty.slip_damping * perp[i].dot(state.v));
    smallest = std::min(smallest, penalty.friction * normal_force - demand);
  }
  return smallest;
}

// Every contact must be clear of d = 0 by far more than the FD step, or
// the comparison straddles the activation boundary and is meaningless.
// That boundary gets its own file.
void ExpectClearOfBoundary(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane, double margin) {
  const std::vector<Contact> contacts = detect_contacts_body(state, shape, plane);
  for (std::size_t i = 0; i < contacts.size(); ++i) {
    EXPECT_GT(std::abs(contacts[i].signed_distance), margin) << "vertex " << i << " sits too near the activation boundary";
  }
}

TEST(PenaltyJacobians, MatchCentralFiniteDifferenceOneActiveContact) {
  // Tilted so a single corner is through the floor and its moment arm is
  // not vertical, which is what makes the geometric (theta, theta) term
  // nonzero.
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.5, 0.3);
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;
  const PenaltyParams penalty{/*stiffness=*/100.0};

  ExpectClearOfBoundary(state, shape, ground, 1.0e-3);

  const ForceJacobian analytic = penalty_force_body_jacobian(state, shape, ground, penalty);
  const Eigen::Matrix3d fd = FiniteDifferenceForceJacobian(state, shape, ground, penalty);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(analytic.df_dq(i, j), fd(i, j), 1.0e-6) << "df_dq mismatch at (" << i << ", " << j << ")";
    }
  }

  // The geometric term is the whole point of this configuration.
  EXPECT_NE(analytic.df_dq(2, 2), 0.0);
}

TEST(PenaltyJacobians, MatchCentralFiniteDifferenceTwoActiveContacts) {
  // Driven deeper so two corners are active at different depths -- the
  // sum over contacts has to be right, not just a single term.
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.2, 0.3);
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;
  const PenaltyParams penalty{/*stiffness=*/100.0};

  ExpectClearOfBoundary(state, shape, ground, 1.0e-3);

  const ForceJacobian analytic = penalty_force_body_jacobian(state, shape, ground, penalty);
  const Eigen::Matrix3d fd = FiniteDifferenceForceJacobian(state, shape, ground, penalty);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(analytic.df_dq(i, j), fd(i, j), 1.0e-6) << "df_dq mismatch at (" << i << ", " << j << ")";
    }
  }
}

TEST(PenaltyJacobians, MatchCentralFiniteDifferenceOnTiltedPlane) {
  // Non-axis-aligned normal and a nonzero offset, so both translation
  // rows of df_c/dq are nontrivial rather than mostly zero.
  HalfPlane plane;
  plane.normal = Eigen::Vector2d(1.0, 2.0).normalized();
  plane.offset = -0.25;

  RigidBodyState state;
  state.q = Eigen::Vector3d(-0.2, -0.4, -1.1);
  const BodyShape shape = UnitSquare();
  const PenaltyParams penalty{/*stiffness=*/100.0};

  ExpectClearOfBoundary(state, shape, plane, 1.0e-3);

  const ForceJacobian analytic = penalty_force_body_jacobian(state, shape, plane, penalty);
  const Eigen::Matrix3d fd = FiniteDifferenceForceJacobian(state, shape, plane, penalty);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(analytic.df_dq(i, j), fd(i, j), 1.0e-6) << "df_dq mismatch at (" << i << ", " << j << ")";
    }
  }
}

TEST(PenaltyJacobians, UndampedVelocityBlockIsExactlyZero) {
  // With b = 0 the force reads position only, whatever the velocity is.
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.2, 0.3);
  state.v = Eigen::Vector3d(1.3, -2.1, 0.7);  // moving, and it must not matter

  const ForceJacobian analytic = penalty_force_body_jacobian(state, UnitSquare(), HalfPlane{}, PenaltyParams{100.0});

  EXPECT_TRUE(analytic.df_dv.isZero(0.0));
}

TEST(PenaltyJacobians, UndampedForceJacobianIsExactlySymmetric) {
  // The spring alone is conservative -- f_c = -grad U with
  // U = sum_i (k/2) min(0, d_i)^2 -- so df_c/dq is -Hessian(U) and must
  // be symmetric. Exactly, not approximately: the material term is a sum
  // of outer products J_i^T J_i, and the geometric term touches only the
  // (theta, theta) diagonal entry.
  //
  // This is a free correctness check only while b = 0. See
  // DampingBreaksJacobianSymmetry for the other side of it.
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.15, 0.2, 0.3);
  state.v = Eigen::Vector3d(0.4, -0.9, 1.1);

  const ForceJacobian analytic = penalty_force_body_jacobian(state, UnitSquare(), HalfPlane{}, PenaltyParams{100.0});
  const Eigen::Matrix3d df_dq = analytic.df_dq;

  EXPECT_DOUBLE_EQ(df_dq(0, 1), df_dq(1, 0));
  EXPECT_DOUBLE_EQ(df_dq(0, 2), df_dq(2, 0));
  EXPECT_DOUBLE_EQ(df_dq(1, 2), df_dq(2, 1));
}

TEST(PenaltyJacobians, MatchCentralFiniteDifferenceWithDamping) {
  // Damping makes the force depend on q twice over: directly through
  // d_i, and through the closing rate J_i(q).v. The second path is the
  // one a naive derivation drops.
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.2, 0.3);
  state.v = Eigen::Vector3d(0.5, -1.4, 0.8);  // approaching, and spinning
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;
  const PenaltyParams penalty{/*stiffness=*/100.0, /*damping=*/12.0};

  ExpectClearOfBoundary(state, shape, ground, 1.0e-3);

  const ForceJacobian analytic = penalty_force_body_jacobian(state, shape, ground, penalty);
  const Eigen::Matrix3d fd = FiniteDifferenceForceJacobian(state, shape, ground, penalty);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(analytic.df_dq(i, j), fd(i, j), 1.0e-6) << "df_dq mismatch at (" << i << ", " << j << ")";
    }
  }
}

TEST(PenaltyJacobians, VelocityBlockMatchesCentralFiniteDifference) {
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.2, 0.3);
  state.v = Eigen::Vector3d(0.5, -1.4, 0.8);
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;
  const PenaltyParams penalty{/*stiffness=*/100.0, /*damping=*/12.0};

  const ForceJacobian analytic = penalty_force_body_jacobian(state, shape, ground, penalty);
  const Eigen::Matrix3d fd = FiniteDifferenceVelocityJacobian(state, shape, ground, penalty);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(analytic.df_dv(i, j), fd(i, j), 1.0e-6) << "df_dv mismatch at (" << i << ", " << j << ")";
    }
  }
}

TEST(PenaltyJacobians, VelocityBlockIsSymmetricNegativeSemidefinite) {
  // df_c/dv = -b * sum_i J_i^T J_i. Symmetric exactly, and negative
  // semidefinite because w^T (df_c/dv) w = -b * sum_i (J_i.w)^2. That is
  // the statement that the damper can only take energy out: with w = v,
  // it is exactly the dissipation rate.
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.2, 0.3);
  state.v = Eigen::Vector3d(0.5, -1.4, 0.8);

  const ForceJacobian analytic = penalty_force_body_jacobian(state, UnitSquare(), HalfPlane{}, PenaltyParams{100.0, 12.0});
  const Eigen::Matrix3d df_dv = analytic.df_dv;

  EXPECT_DOUBLE_EQ(df_dv(0, 1), df_dv(1, 0));
  EXPECT_DOUBLE_EQ(df_dv(0, 2), df_dv(2, 0));
  EXPECT_DOUBLE_EQ(df_dv(1, 2), df_dv(2, 1));

  for (const Eigen::Vector3d& w : {Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(0.0, 1.0, 0.0), Eigen::Vector3d(0.0, 0.0, 1.0), Eigen::Vector3d(0.7, -1.3, 2.2), state.v}) {
    EXPECT_LE(w.transpose() * df_dv * w, 0.0) << "w = " << w.transpose();
  }
}

TEST(PenaltyJacobians, DampingBreaksJacobianSymmetry) {
  // The damper is non-conservative, so df_c/dq stops being a Hessian.
  // Concretely, d(J_i.v)/dq is nonzero only in its theta component, so
  // J_i^T times it is an outer product confined to the third column --
  // rank-1 and not symmetric.
  //
  // For the ground plane J_i = [0, 1, *], so the asymmetry shows up in
  // the (1,2) / (2,1) pair; the (0,*) entries stay zero because n_x = 0.
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.2, 0.3);
  state.v = Eigen::Vector3d(0.5, -1.4, 0.8);  // omega != 0 is required
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;

  const Eigen::Matrix3d damped = penalty_force_body_jacobian(state, shape, ground, PenaltyParams{100.0, 12.0}).df_dq;
  const Eigen::Matrix3d undamped = penalty_force_body_jacobian(state, shape, ground, PenaltyParams{100.0, 0.0}).df_dq;

  EXPECT_FALSE(damped.isApprox(damped.transpose(), 1.0e-12));
  EXPECT_NE(damped(1, 2), damped(2, 1));

  // Control: the same operating point with b = 0 is symmetric, so the
  // asymmetry is the damping and not the configuration.
  EXPECT_TRUE(undamped.isApprox(undamped.transpose(), 1.0e-12));

  // And the whole difference lives in the third column.
  const Eigen::Matrix3d difference = damped - undamped;
  EXPECT_TRUE(difference.leftCols<2>().isZero(0.0));
}

TEST(PenaltyJacobians, DampedStepContractsPhaseSpaceVolumeByTheDelassusOperator) {
  // The counterpart of the undamped det = 1 check, and the sharpest
  // statement 5b makes. For any force law
  //
  //   det(dz_dz) = det(Id + dt * M^-1 * df/dv)
  //
  // and with df_c/dv = -b * J_A^T J_A, Sylvester's identity turns that
  // into det(Id - dt*b*Delassus) over the active contacts, where
  // Delassus = J_A * M^-1 * J_A^T is the inverse effective mass seen at
  // the contacts -- the same operator step 6's solver is built around.
  //
  // Expected value is assembled here from detection output and the mass
  // matrix, independently of the integrator's assembly.
  const RigidBodyParams params{/*mass=*/1.0, /*inertia=*/1.0 / 6.0};
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;
  const PenaltyParams penalty{/*stiffness=*/100.0, /*damping=*/50.0};
  const double dt = 1.0e-3;

  // Both closing, so nothing is held at zero by the adhesion clamp.
  const std::vector<Eigen::Vector3d> configurations = {
      Eigen::Vector3d(0.0, 0.45, 0.0),   // two active contacts
      Eigen::Vector3d(0.0, 0.50, 0.3),   // one active contact
      Eigen::Vector3d(0.4, 0.30, -0.9),  // two, off-centre and rotated
  };

  for (const Eigen::Vector3d& q : configurations) {
    RigidBodyState state;
    state.q = q;
    state.v = Eigen::Vector3d(0.2, -1.0, 0.3);

    // Assemble J_A over the contacts that are actually carrying force.
    const std::vector<Contact> contacts = detect_contacts_body(state, shape, ground);
    const std::vector<Eigen::RowVector3d> jacobians = detect_contacts_body_jacobian(state, shape, ground);
    std::vector<Eigen::RowVector3d> active;
    for (std::size_t i = 0; i < contacts.size(); ++i) {
      const double closing_rate = jacobians[i].dot(state.v);
      if (contacts[i].signed_distance < 0.0 && -penalty.stiffness * contacts[i].signed_distance - penalty.damping * closing_rate > 0.0) {
        active.push_back(jacobians[i]);
      }
    }
    ASSERT_FALSE(active.empty()) << "configuration " << q.transpose() << " carries no contact force";

    Eigen::MatrixXd stacked(static_cast<Eigen::Index>(active.size()), 3);
    for (std::size_t i = 0; i < active.size(); ++i) {
      stacked.row(static_cast<Eigen::Index>(i)) = active[i];
    }
    const Eigen::Matrix3d m_inv = inverse_mass_diagonal(params).asDiagonal();
    const Eigen::MatrixXd delassus = stacked * m_inv * stacked.transpose();
    const Eigen::MatrixXd contraction = Eigen::MatrixXd::Identity(delassus.rows(), delassus.cols()) - dt * penalty.damping * delassus;
    const double expected = contraction.determinant();

    const StepJacobians jac = step_body_jacobian(state, params, shape, ground, penalty, dt, kDefaultGravity);

    EXPECT_NEAR(jac.dz_dz.determinant(), expected, 1.0e-14) << "configuration " << q.transpose();
    // Dissipation contracts phase-space volume -- strictly, not just
    // "no longer exactly 1".
    EXPECT_LT(jac.dz_dz.determinant(), 1.0) << "configuration " << q.transpose();
  }
}

TEST(PenaltyJacobians, ContactStepStillPreservesPhaseSpaceVolume) {
  // The reason 5a is its own increment. det(dz_dz) = 1 held trivially for
  // gravity because df/dq was identically zero; here it is emphatically
  // not, and the dt^2*M^-1*df/dq terms in the position row have to cancel
  // against the velocity row for the determinant to come back to 1.
  //
  // Generally, det(dz_dz) = det(Id + dt*M^-1*df/dv) for ANY force law --
  // the df/dq contribution cancels identically. A conservative force has
  // df/dv = 0 and therefore determinant 1. Damping (5b) will make this
  // det(Id - dt*b*Delassus) < 1 instead. See
  // docs/derivations/penalty_contact.md.
  const RigidBodyParams params{/*mass=*/1.0, /*inertia=*/1.0 / 6.0};
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;
  const PenaltyParams penalty{/*stiffness=*/100.0};
  const double dt = 1.0e-3;

  // Vary depth and orientation: one active contact, two active contacts,
  // and a configuration with a nonzero moment arm. The identity is
  // independent of the operating point.
  const std::vector<Eigen::Vector3d> configurations = {
      Eigen::Vector3d(0.0, 0.5, 0.3),
      Eigen::Vector3d(0.0, 0.2, 0.3),
      Eigen::Vector3d(0.4, 0.35, -0.9),
  };

  for (const Eigen::Vector3d& q : configurations) {
    RigidBodyState state;
    state.q = q;
    state.v = Eigen::Vector3d(0.7, -1.2, 0.4);

    const StepJacobians jac = step_body_jacobian(state, params, shape, ground, penalty, dt, kDefaultGravity);

    // Contact must actually be active, or this reduces to the free-flight
    // case the step 2 test already covers.
    const ForceJacobian force_jac = penalty_force_body_jacobian(state, shape, ground, penalty);
    ASSERT_FALSE(force_jac.df_dq.isZero(0.0)) << "configuration " << q.transpose() << " is not in contact";

    EXPECT_NEAR(jac.dz_dz.determinant(), 1.0, 1.0e-14) << "configuration " << q.transpose();
  }
}

TEST(PenaltyJacobians, MatchCentralFiniteDifferenceWithSlipDamping) {
  // Sliding and spinning while pressed in, so the tangential force
  // depends on q through two paths -- J_perp,i itself rotating, and the
  // slip rate J_perp,i(q).v changing with it -- exactly mirroring the
  // normal damper.
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.2, 0.3);
  state.v = Eigen::Vector3d(1.6, -1.4, 0.8);
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;
  const PenaltyParams penalty{/*stiffness=*/100.0, /*damping=*/12.0, /*slip_damping=*/8.0, /*friction=*/5.0};

  ExpectClearOfBoundary(state, shape, ground, 1.0e-3);
  ASSERT_GT(SmallestConeMargin(state, shape, ground, penalty), 1.0) << "this test exercises the sticking branch";

  const ForceJacobian analytic = penalty_force_body_jacobian(state, shape, ground, penalty);
  const Eigen::Matrix3d fd_dq = FiniteDifferenceForceJacobian(state, shape, ground, penalty);
  const Eigen::Matrix3d fd_dv = FiniteDifferenceVelocityJacobian(state, shape, ground, penalty);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(analytic.df_dq(i, j), fd_dq(i, j), 1.0e-6) << "df_dq mismatch at (" << i << ", " << j << ")";
      EXPECT_NEAR(analytic.df_dv(i, j), fd_dv(i, j), 1.0e-6) << "df_dv mismatch at (" << i << ", " << j << ")";
    }
  }
}

TEST(PenaltyJacobians, StickingFrictionKeepsTheVelocityBlockSymmetricNegativeSemidefinite) {
  // Unbounded tangential damping contributes -b_slip * sum J_perp^T
  // J_perp, which is the same shape as the normal damper's term. Stacked,
  // df_c/dv = -A^T diag(b, b_slip) A, symmetric and negative
  // semidefinite in both directions at once.
  //
  // This is the claim the Coulomb cone will destroy: once the tangential
  // force saturates at mu*lambda its sensitivity comes from the NORMAL
  // force instead, contributing an outer product of two different
  // vectors. Isolated here first, deliberately.
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.2, 0.3);
  state.v = Eigen::Vector3d(1.6, -1.4, 0.8);

  const PenaltyParams penalty{/*stiffness=*/100.0, /*damping=*/12.0, /*slip_damping=*/8.0, /*friction=*/5.0};
  ASSERT_GT(SmallestConeMargin(state, UnitSquare(), HalfPlane{}, penalty), 1.0) << "this test exercises the sticking branch";

  const ForceJacobian analytic = penalty_force_body_jacobian(state, UnitSquare(), HalfPlane{}, penalty);
  const Eigen::Matrix3d df_dv = analytic.df_dv;

  EXPECT_DOUBLE_EQ(df_dv(0, 1), df_dv(1, 0));
  EXPECT_DOUBLE_EQ(df_dv(0, 2), df_dv(2, 0));
  EXPECT_DOUBLE_EQ(df_dv(1, 2), df_dv(2, 1));

  for (const Eigen::Vector3d& w : {Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(0.0, 1.0, 0.0), Eigen::Vector3d(0.0, 0.0, 1.0), Eigen::Vector3d(0.7, -1.3, 2.2), state.v}) {
    EXPECT_LE(w.transpose() * df_dv * w, 0.0) << "w = " << w.transpose();
  }

  // The x row is nonzero only because of slip damping -- the normal
  // direction alone cannot resist horizontal motion on a flat floor.
  EXPECT_LT(df_dv(0, 0), 0.0);
}

TEST(PenaltyJacobians, MatchCentralFiniteDifferenceWhileSliding) {
  // The saturated branch, where beta is pinned to -mu*lambda and its
  // entire state dependence runs through the NORMAL force instead of
  // through slip. Structurally different derivatives, so it needs its own
  // finite-difference check rather than being covered by the sticking one.
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.2, 0.3);
  state.v = Eigen::Vector3d(10.0, -1.4, 0.8);
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;
  const PenaltyParams penalty{/*stiffness=*/100.0, /*damping=*/12.0, /*slip_damping=*/8.0, /*friction=*/0.5};

  ExpectClearOfBoundary(state, shape, ground, 1.0e-3);
  ASSERT_LT(SmallestConeMargin(state, shape, ground, penalty), -1.0) << "this test exercises the sliding branch";

  const ForceJacobian analytic = penalty_force_body_jacobian(state, shape, ground, penalty);
  const Eigen::Matrix3d fd_dq = FiniteDifferenceForceJacobian(state, shape, ground, penalty);
  const Eigen::Matrix3d fd_dv = FiniteDifferenceVelocityJacobian(state, shape, ground, penalty);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(analytic.df_dq(i, j), fd_dq(i, j), 1.0e-6) << "df_dq mismatch at (" << i << ", " << j << ")";
      EXPECT_NEAR(analytic.df_dv(i, j), fd_dv(i, j), 1.0e-6) << "df_dv mismatch at (" << i << ", " << j << ")";
    }
  }
}

TEST(PenaltyJacobians, SlidingFrictionBreaksVelocityBlockSymmetry) {
  // The claim 8a existed to isolate, now destroyed. While sticking,
  // df_c/dv = -A^T diag(b, b_slip) A and is symmetric negative
  // semidefinite. While sliding, beta = -sigma*mu*lambda contributes
  // J_perp^T (sigma*mu*b*J) -- an outer product of the perp row with the
  // NORMAL row, two vectors that are neither parallel nor even in the
  // same direction. Symmetry goes, and so does the -A^T D A form the
  // generalized Delassus determinant relied on.
  //
  // Dissipation survives: that is a property of the force, not of this
  // block, and test_penalty_force.cpp asserts it in both regimes.
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.2, 0.3);
  state.v = Eigen::Vector3d(10.0, -1.4, 0.8);
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;

  const PenaltyParams sliding{/*stiffness=*/100.0, /*damping=*/12.0, /*slip_damping=*/8.0, /*friction=*/0.5};
  const PenaltyParams sticking{/*stiffness=*/100.0, /*damping=*/12.0, /*slip_damping=*/8.0, /*friction=*/5.0};
  ASSERT_LT(SmallestConeMargin(state, shape, ground, sliding), -1.0);
  ASSERT_GT(SmallestConeMargin(state, shape, ground, sticking), 1.0);

  const Eigen::Matrix3d slid = penalty_force_body_jacobian(state, shape, ground, sliding).df_dv;
  const Eigen::Matrix3d stuck = penalty_force_body_jacobian(state, shape, ground, sticking).df_dv;

  EXPECT_FALSE(slid.isApprox(slid.transpose(), 1.0e-12)) << "sliding friction should break symmetry";
  EXPECT_TRUE(stuck.isApprox(stuck.transpose(), 1.0e-12)) << "the same operating point sticks symmetrically";

  // And it is no longer negative semidefinite either. Per contact the
  // quadratic form in (a, c) = (J.w, J_perp.w) is
  //
  //   -b*a^2 + sigma*mu*b*a*c
  //
  // whose matrix b*[[-1, mu/2], [mu/2, 0]] has determinant -b^2*mu^2/4,
  // negative for any mu > 0. Indefinite by construction, not by accident
  // of this operating point -- so the check is on the eigenvalues of the
  // symmetric part rather than on a handful of guessed directions.
  const Eigen::Matrix3d symmetric_part = 0.5 * (slid + slid.transpose());
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(symmetric_part);
  EXPECT_GT(solver.eigenvalues().maxCoeff(), 0.0) << "sliding should cost negative semidefiniteness, not just symmetry";

  // The sticking block, by contrast, is negative semidefinite outright.
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> stuck_solver(stuck);
  EXPECT_LE(stuck_solver.eigenvalues().maxCoeff(), 0.0);
}

TEST(PenaltyJacobians, DampedStepContractsByTheGeneralizedDelassus) {
  // The 5b identity, one direction wider. With
  //
  //   df_c/dv = -A^T diag(b, b_slip) A,     A = [J ; J_perp] stacked
  //
  // Sylvester turns det(dz_dz) = det(Id + dt*M^-1*df/dv) into
  //
  //   det(Id - dt * (A M^-1 A^T) * D)
  //
  // where A M^-1 A^T is the Delassus operator over BOTH directions
  // rather than the normal one alone. Assembled here from detection
  // output and the mass matrix, independently of the integrator.
  const RigidBodyParams params{/*mass=*/1.0, /*inertia=*/1.0 / 6.0};
  const BodyShape shape = UnitSquare();
  const HalfPlane ground;
  // mu deliberately absurd: this test is about the algebra of the
  // unsaturated branch, and a physical friction coefficient would put
  // some of these configurations on the cone.
  const PenaltyParams penalty{/*stiffness=*/100.0, /*damping=*/50.0, /*slip_damping=*/30.0, /*friction=*/50.0};
  const double dt = 1.0e-3;

  const std::vector<Eigen::Vector3d> configurations = {
      Eigen::Vector3d(0.0, 0.45, 0.0),
      Eigen::Vector3d(0.0, 0.50, 0.3),
      Eigen::Vector3d(0.4, 0.30, -0.9),
  };

  for (const Eigen::Vector3d& q : configurations) {
    RigidBodyState state;
    state.q = q;
    state.v = Eigen::Vector3d(0.9, -1.0, 0.3);
    ASSERT_GT(SmallestConeMargin(state, shape, ground, penalty), 1.0) << "configuration " << q.transpose() << " is on the cone";

    const std::vector<Contact> contacts = detect_contacts_body(state, shape, ground);
    const std::vector<Eigen::RowVector3d> jacobians = detect_contacts_body_jacobian(state, shape, ground);
    const std::vector<Eigen::RowVector3d> perp = detect_contacts_body_perp_jacobian(state, shape, ground);

    std::vector<Eigen::RowVector3d> rows;
    std::vector<double> coefficients;
    for (std::size_t i = 0; i < contacts.size(); ++i) {
      const double closing_rate = jacobians[i].dot(state.v);
      if (contacts[i].signed_distance < 0.0 && -penalty.stiffness * contacts[i].signed_distance - penalty.damping * closing_rate > 0.0) {
        rows.push_back(jacobians[i]);
        coefficients.push_back(penalty.damping);
        rows.push_back(perp[i]);
        coefficients.push_back(penalty.slip_damping);
      }
    }
    ASSERT_FALSE(rows.empty()) << "configuration " << q.transpose() << " carries no contact force";

    const auto size = static_cast<Eigen::Index>(rows.size());
    Eigen::MatrixXd stacked(size, 3);
    Eigen::VectorXd diagonal(size);
    for (std::size_t i = 0; i < rows.size(); ++i) {
      stacked.row(static_cast<Eigen::Index>(i)) = rows[i];
      diagonal(static_cast<Eigen::Index>(i)) = coefficients[i];
    }

    const Eigen::Matrix3d m_inv = inverse_mass_diagonal(params).asDiagonal();
    const Eigen::MatrixXd delassus = stacked * m_inv * stacked.transpose();
    const Eigen::MatrixXd contraction = Eigen::MatrixXd::Identity(size, size) - dt * delassus * diagonal.asDiagonal();

    const StepJacobians jac = step_body_jacobian(state, params, shape, ground, penalty, dt, kDefaultGravity);

    EXPECT_NEAR(jac.dz_dz.determinant(), contraction.determinant(), 1.0e-14) << "configuration " << q.transpose();
    EXPECT_LT(jac.dz_dz.determinant(), 1.0) << "configuration " << q.transpose();
  }
}

}  // namespace
}  // namespace grip
