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

}  // namespace
}  // namespace grip
