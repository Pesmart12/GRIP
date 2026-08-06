#include <cmath>
#include <numbers>
#include <vector>

#include <gtest/gtest.h>

#include "contact/detection.hpp"
#include "contact/half_plane.hpp"
#include "core/rigid_body.hpp"

namespace grip {
namespace {

// Unit square centered on the COM, corners at (+-0.5, +-0.5). Vertex
// order is fixed and is also the contact order.
BodyShape UnitSquare() {
  return BodyShape{{{-0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}, {-0.5, 0.5}}};
}

TEST(ContactDetection, AxisAlignedBoxAboveGround) {
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.6, 0.0);

  const std::vector<Contact> contacts = detect_contacts_body(state, UnitSquare(), HalfPlane{});

  // Ground is y = 0 with normal (0,1), so the signed distance is just
  // the world y of each corner. Bottom corners sit at y = 0.1, top
  // corners at y = 1.1.
  ASSERT_EQ(contacts.size(), 4u);
  EXPECT_DOUBLE_EQ(contacts[0].signed_distance, 0.1);
  EXPECT_DOUBLE_EQ(contacts[1].signed_distance, 0.1);
  EXPECT_DOUBLE_EQ(contacts[2].signed_distance, 1.1);
  EXPECT_DOUBLE_EQ(contacts[3].signed_distance, 1.1);

  // Separated vertices are still reported, with their margins intact.
  for (const Contact& contact : contacts) {
    EXPECT_GT(contact.signed_distance, 0.0);
  }
}

TEST(ContactDetection, ContactOrderMatchesVertexOrder) {
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.6, 0.0);
  const BodyShape shape = UnitSquare();

  const std::vector<Contact> contacts = detect_contacts_body(state, shape, HalfPlane{});

  ASSERT_EQ(contacts.size(), shape.vertices.size());
  for (std::size_t i = 0; i < contacts.size(); ++i) {
    EXPECT_EQ(contacts[i].vertex_index, static_cast<int>(i));
    // point is the vertex's world position; at theta = 0 that's just
    // the body-frame vertex translated by the center.
    const Eigen::Vector2d expected = state.q.head<2>() + shape.vertices[i];
    EXPECT_TRUE(contacts[i].point.isApprox(expected));
  }
}

TEST(ContactDetection, RotationAlonePenetratesGround) {
  // Same center height as the axis-aligned case above -- only theta
  // changes. Balancing the square on a corner drops that corner from
  // y = 0.1 to y = 0.6 - sqrt(0.5), i.e. through the floor. This is the
  // case a disc could never produce: contact geometry that depends on
  // orientation. See docs/derivations/contact_detection.md.
  RigidBodyState state;
  state.q = Eigen::Vector3d(0.0, 0.6, std::numbers::pi / 4.0);

  const std::vector<Contact> contacts = detect_contacts_body(state, UnitSquare(), HalfPlane{});

  const double half_diagonal = std::sqrt(0.5);
  ASSERT_EQ(contacts.size(), 4u);
  EXPECT_NEAR(contacts[0].signed_distance, 0.6 - half_diagonal, 1e-12);
  EXPECT_LT(contacts[0].signed_distance, 0.0);  // penetrating

  // The two side corners swing level with the center.
  EXPECT_NEAR(contacts[1].signed_distance, 0.6, 1e-12);
  EXPECT_NEAR(contacts[3].signed_distance, 0.6, 1e-12);
  EXPECT_NEAR(contacts[2].signed_distance, 0.6 + half_diagonal, 1e-12);
}

TEST(ContactDetection, SignConventionAcrossTheBoundary) {
  const BodyShape shape = UnitSquare();
  RigidBodyState state;

  // Bottom corners exactly on the plane.
  state.q = Eigen::Vector3d(0.0, 0.5, 0.0);
  EXPECT_DOUBLE_EQ(detect_contacts_body(state, shape, HalfPlane{})[0].signed_distance, 0.0);

  // Lowered by 0.2: penetrating by exactly that much.
  state.q.y() = 0.3;
  EXPECT_DOUBLE_EQ(detect_contacts_body(state, shape, HalfPlane{})[0].signed_distance, -0.2);

  // Raised by 0.2: separated by exactly that much.
  state.q.y() = 0.7;
  EXPECT_DOUBLE_EQ(detect_contacts_body(state, shape, HalfPlane{})[0].signed_distance, 0.2);
}

TEST(ContactDetection, RespectsNonDefaultPlane) {
  // A wall: free space is x >= -1, so the escape direction is +x and
  // the signed distance measures horizontal clearance, not height.
  HalfPlane wall;
  wall.normal = Eigen::Vector2d(1.0, 0.0);
  wall.offset = -1.0;

  RigidBodyState state;
  state.q = Eigen::Vector3d(-0.8, 0.0, 0.0);

  const std::vector<Contact> contacts = detect_contacts_body(state, UnitSquare(), wall);

  // Left corners at x = -1.3 are 0.3 past the wall; right corners at
  // x = -0.3 are 0.7 clear of it.
  EXPECT_DOUBLE_EQ(contacts[0].signed_distance, -0.3);
  EXPECT_DOUBLE_EQ(contacts[3].signed_distance, -0.3);
  EXPECT_DOUBLE_EQ(contacts[1].signed_distance, 0.7);
  EXPECT_DOUBLE_EQ(contacts[2].signed_distance, 0.7);
}

}  // namespace
}  // namespace grip
