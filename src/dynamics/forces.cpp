#include "dynamics/forces.hpp"

namespace grip {

Eigen::Vector3d gravity_force(const RigidBodyParams& params, double gravity) {
  return Eigen::Vector3d(0.0, -params.mass * gravity, 0.0);
}

ForceJacobian gravity_force_jacobian(const Eigen::Vector3d&, const Eigen::Vector3d&, const RigidBodyParams&, double) {
  return ForceJacobian{};
}

}  // namespace grip
