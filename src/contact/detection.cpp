#include "contact/detection.hpp"

#include <cstddef>

#include <Eigen/Geometry>

namespace grip {
namespace {

// Perp operator: a^perp = (-a_y, a_x), rotation by +90 degrees.
// dR/dtheta * r = (R(theta) r)^perp, so a vertex's velocity under unit
// angular rate is (p - c)^perp. See docs/derivations/notation.md.
Eigen::Vector2d Perp(const Eigen::Vector2d& a) {
  return Eigen::Vector2d(-a.y(), a.x());
}

}  // namespace

std::vector<Contact> detect_contacts_body(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane) {
  const Eigen::Vector2d center = state.q.head<2>();
  const Eigen::Matrix2d rotation = Eigen::Rotation2Dd(state.q.z()).toRotationMatrix();

  std::vector<Contact> contacts(shape.vertices.size());
  for (std::size_t i = 0; i < shape.vertices.size(); ++i) {
    const Eigen::Vector2d world_point = center + rotation * shape.vertices[i];
    contacts[i].vertex_index = static_cast<int>(i);
    contacts[i].signed_distance = plane.normal.dot(world_point) - plane.offset;
    contacts[i].normal = plane.normal;
    contacts[i].point = world_point;
  }
  return contacts;
}


std::vector<Eigen::RowVector3d> detect_contacts_body_jacobian(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane) {
  const Eigen::Matrix2d rotation = Eigen::Rotation2Dd(state.q.z()).toRotationMatrix();

  std::vector<Eigen::RowVector3d> jacobians(shape.vertices.size());
  for (std::size_t i = 0; i < shape.vertices.size(); ++i) {
    // arm = p_i - c = R(theta) * vertices[i]; no need to add the center
    // back only to subtract it again.
    const Eigen::Vector2d arm = rotation * shape.vertices[i];
    jacobians[i] << plane.normal.x(), plane.normal.y(), plane.normal.dot(Perp(arm));
  }
  return jacobians;
}


std::vector<Eigen::RowVector3d> detect_contacts_body_perp_jacobian(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane) {
  const Eigen::Matrix2d rotation = Eigen::Rotation2Dd(state.q.z()).toRotationMatrix();
  const Eigen::Vector2d tangent = Perp(plane.normal);

  std::vector<Eigen::RowVector3d> jacobians(shape.vertices.size());
  for (std::size_t i = 0; i < shape.vertices.size(); ++i) {
    // Identical construction to the normal row, with n^perp in place of
    // n -- the perp operator appears twice for different reasons: once
    // to turn the plane normal into the surface direction, once because
    // a vertex's velocity under unit angular rate is (p_i - c)^perp.
    const Eigen::Vector2d arm = rotation * shape.vertices[i];
    jacobians[i] << tangent.x(), tangent.y(), tangent.dot(Perp(arm));
  }
  return jacobians;
}

}  // namespace grip
