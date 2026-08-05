#include "contact/detection.hpp"

#include <cstddef>

#include <Eigen/Geometry>

namespace grip {
namespace {

// J: rotation by +90 degrees. dR/dtheta = J * R(theta), so a vertex's
// velocity under unit angular rate is J * (p - c).
const Eigen::Matrix2d& PerpRotation() {
  static const Eigen::Matrix2d kJ = (Eigen::Matrix2d() << 0.0, -1.0, 1.0, 0.0).finished();
  return kJ;
}

}  // namespace

std::vector<Contact> detect_contacts_body(const RigidBodyState& state, const BodyShape& shape, const HalfPlane& plane) {
  const Eigen::Vector2d center = state.q.head<2>();
  const Eigen::Matrix2d rotation = Eigen::Rotation2Dd(state.q.z()).toRotationMatrix();

  std::vector<Contact> contacts(shape.vertices.size());
  for (std::size_t i = 0; i < shape.vertices.size(); ++i) {
    const Eigen::Vector2d world_point = center + rotation * shape.vertices[i];
    contacts[i].vertex_index = static_cast<int>(i);
    contacts[i].phi = plane.normal.dot(world_point) - plane.offset;
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
    jacobians[i] << plane.normal.x(), plane.normal.y(), plane.normal.dot(PerpRotation() * arm);
  }
  return jacobians;
}

}  // namespace grip
