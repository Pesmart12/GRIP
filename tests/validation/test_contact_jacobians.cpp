#include <cmath>
#include <cstddef>
#include <functional>
#include <numbers>
#include <vector>

#include <gtest/gtest.h>

#include "contact/detection.hpp"
#include "contact/half_plane.hpp"
#include "core/rigid_body.hpp"
#include "utils/finite_difference.hpp"

namespace grip {
namespace {

BodyShape UnitSquare() {
  return BodyShape{{{-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}}};
}

// Central-difference d(d_i)/dq for one contact, by re-running real
// detection at perturbed configurations.
Eigen::RowVector3d FiniteDifferenceDistanceJacobian(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane, std::size_t contact_index) {
  const std::function<Eigen::Matrix<double, 1, 1>(const Eigen::Vector3d&)> distance_of_q = [&](const Eigen::Vector3d& q) {
    RigidBodyState perturbed = state;
    perturbed.q = q;
    Eigen::Matrix<double, 1, 1> out;
    out(0) = detect_contacts_body(perturbed, shape, plane)[contact_index].signed_distance;
    return out;
  };
  return testutil::CentralDifferenceJacobian<1, 3>(distance_of_q, state.q);
}

TEST(ContactJacobians, MatchCentralFiniteDifference) {
  // Rotated to a generic angle so the rotation component is nonzero for
  // every vertex -- an axis-aligned box would leave some of them at
  // values that a sign error could still reproduce.
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.3, 0.45, 0.7);
  const BodyShape shape = UnitSquare();
  const HalfPlane plane;

  const std::vector<Eigen::RowVector3d> analytic = detect_contacts_body_jacobian(state, shape, plane);

  ASSERT_EQ(analytic.size(), shape.vertices.size());
  for (std::size_t i = 0; i < analytic.size(); ++i) {
    const Eigen::RowVector3d fd = FiniteDifferenceDistanceJacobian(state, shape, plane, i);
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(analytic[i](j), fd(j), 1e-6) << "contact " << i << ", component " << j;
    }
  }
}

TEST(ContactJacobians, MatchCentralFiniteDifferenceOnTiltedPlane) {
  // Non-axis-aligned normal, so both translation components of the
  // contact Jacobian are nontrivial rather than 0 and 1, and the plane
  // is offset from the origin as well.
  HalfPlane plane;
  plane.normal = Eigen::Vector2d(1.0, 2.0).normalized();
  plane.offset = -0.25;

  RigidBodyState state;
  state.q = Eigen::Vector3d(-0.2, 0.8, -1.1);
  const BodyShape shape = UnitSquare();

  const std::vector<Eigen::RowVector3d> analytic = detect_contacts_body_jacobian(state, shape, plane);

  for (std::size_t i = 0; i < analytic.size(); ++i) {
    const Eigen::RowVector3d fd = FiniteDifferenceDistanceJacobian(state, shape, plane, i);
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(analytic[i](j), fd(j), 1e-6) << "contact " << i << ", component " << j;
    }
  }
}

TEST(ContactJacobians, TranslationBlockIsThePlaneNormal) {
  // The signed distance is linear in the center position, so its x and
  // y derivatives are exactly the plane normal for every vertex, at any
  // orientation -- not approximately, exactly.
  RigidBodyState state;
  state.q = Eigen::Vector3d(1.3, -0.4, 2.2);
  const HalfPlane plane;

  const std::vector<Eigen::RowVector3d> analytic = detect_contacts_body_jacobian(state, UnitSquare(), plane);

  for (const Eigen::RowVector3d& jacobian : analytic) {
    EXPECT_DOUBLE_EQ(jacobian(0), plane.normal.x());
    EXPECT_DOUBLE_EQ(jacobian(1), plane.normal.y());
  }
}

TEST(ContactJacobians, RotationDerivativeVanishesAtExtremalCorner) {
  // Square balanced on a corner: that corner is at its lowest possible
  // height, so the signed distance is stationary in theta and its
  // rotation derivative must be zero. Checks the (p-c)^perp term
  // specifically -- a wrong sign or a missing rotation would still be
  // nonzero here.
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.6, std::numbers::pi / 4.0);

  const std::vector<Eigen::RowVector3d> analytic = detect_contacts_body_jacobian(state, UnitSquare(), HalfPlane{});

  EXPECT_NEAR(analytic[0](2), 0.0, 1e-15);
  // The opposite corner is at its highest point -- also stationary.
  EXPECT_NEAR(analytic[2](2), 0.0, 1e-15);
  // The two side corners are moving fastest in y, at +-half the diagonal.
  EXPECT_NEAR(std::abs(analytic[1](2)), std::sqrt(0.5), 1e-12);
  EXPECT_NEAR(std::abs(analytic[3](2)), std::sqrt(0.5), 1e-12);
}

TEST(PerpJacobians, TranslationBlockIsTheSurfaceDirection) {
  // n^perp = (-n_y, n_x). For the default ground plane n = (0, 1) that
  // points in -x, so a body sliding in +x has NEGATIVE slip. Correct and
  // self-consistent, but it reads backwards, which is why it is pinned
  // here rather than left to be rediscovered.
  RigidBodyState state;
  state.q = Eigen::Vector3d(1.3, -0.4, 2.2);
  const HalfPlane ground;

  const std::vector<Eigen::RowVector3d> perp = detect_contacts_body_perp_jacobian(state, UnitSquare(), ground);

  for (const Eigen::RowVector3d& row : perp) {
    EXPECT_DOUBLE_EQ(row(0), -ground.normal.y());
    EXPECT_DOUBLE_EQ(row(1), ground.normal.x());
  }

  // And the two translation blocks are orthogonal, since n . n^perp = 0.
  const std::vector<Eigen::RowVector3d> normal = detect_contacts_body_jacobian(state, UnitSquare(), ground);
  for (std::size_t i = 0; i < perp.size(); ++i) {
    EXPECT_DOUBLE_EQ(normal[i].head<2>().dot(perp[i].head<2>()), 0.0);
  }
}

TEST(PerpJacobians, MapVelocityToContactPointSlip) {
  // The defining property, checked against an independent construction:
  // s_i is the world velocity of contact point i, projected onto the
  // surface. The velocity of a point on a rigid body is the COM velocity
  // plus omega times the moment arm rotated ninety degrees.
  //
  // Tilted plane and a generic state so translation and rotation both
  // contribute, rather than one of them being zero by symmetry.
  HalfPlane plane;
  plane.normal = Eigen::Vector2d(1.0, 2.0).normalized();
  plane.offset = -0.25;

  RigidBodyState state;
  state.q = Eigen::Vector3d(-0.2, 0.8, -1.1);
  state.v = Eigen::Vector3d(0.7, -1.3, 2.1);
  const BodyShape shape = UnitSquare();

  const std::vector<Contact> contacts = detect_contacts_body(state, shape, plane);
  const std::vector<Eigen::RowVector3d> perp = detect_contacts_body_perp_jacobian(state, shape, plane);
  const Eigen::Vector2d tangent(-plane.normal.y(), plane.normal.x());

  for (std::size_t i = 0; i < contacts.size(); ++i) {
    const Eigen::Vector2d arm = contacts[i].point - state.q.head<2>();
    const Eigen::Vector2d arm_perp(-arm.y(), arm.x());
    const Eigen::Vector2d point_velocity = state.v.head<2>() + state.v.z() * arm_perp;

    EXPECT_NEAR(perp[i].dot(state.v), tangent.dot(point_velocity), 1e-14) << "contact " << i;
  }
}

TEST(PerpJacobians, RotationDerivativeVanishesWhereTheArmIsAlongTheSurface) {
  // The third component is n^perp . (p_i - c)^perp, which is zero exactly
  // when the moment arm is parallel to n^perp -- a vertex level with the
  // COM in the surface direction sweeps perpendicular to the surface, so
  // spinning does not slide it.
  //
  // Square at theta = 0 on the default ground: corners sit at (+-0.5,
  // +-0.5), so no arm is purely tangential and none of these vanish;
  // rotate by 45 degrees and two of them do.
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.6, std::numbers::pi / 4.0);

  const std::vector<Eigen::RowVector3d> perp = detect_contacts_body_perp_jacobian(state, UnitSquare(), HalfPlane{});

  // Corners 0 and 2 are directly below and above the COM, so their arms
  // are vertical -- along n, not along n^perp.
  EXPECT_NEAR(std::abs(perp[0](2)), std::sqrt(0.5), 1e-12);
  EXPECT_NEAR(std::abs(perp[2](2)), std::sqrt(0.5), 1e-12);
  // Corners 1 and 3 are level with the COM, arms purely horizontal.
  EXPECT_NEAR(perp[1](2), 0.0, 1e-15);
  EXPECT_NEAR(perp[3](2), 0.0, 1e-15);
}

}  // namespace
}  // namespace grip
