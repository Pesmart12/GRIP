#pragma once

#include <Eigen/Core>

#include "core/rigid_body.hpp"

namespace grip {

// Standard gravitational acceleration, m/s^2. y-up convention: gravity
// acts in -y.
inline constexpr double kDefaultGravity = 9.81;

// Net gravitational wrench (fx, fy, tau) acting at the center of mass.
// Depends only on mass -- no torque, since it acts through the COM.
Eigen::Vector3d gravity_force(const RigidBodyParams& params, double gravity = kDefaultGravity);


// df/dq and df/dv for a force law f(q, v, u). Force laws that depend on
// state (a penalty spring's df/dq, a damper's df/dv) plug into the
// integrator's chain rule through this shape; df/du is not part of it
// because every force law is additive in u by construction (df/du = I
// always -- see docs/derivations/symplectic_euler.md).
struct ForceJacobian {
  Eigen::Matrix3d df_dq = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d df_dv = Eigen::Matrix3d::Zero();
};

// Gravity does not depend on q, v, mass, or g, so this is identically
// zero -- every parameter is accepted but unused, to keep the call
// signature uniform across force laws. Callers (the integrator's
// Jacobian assembly) don't need to know which force law is active.
ForceJacobian gravity_force_jacobian(const Eigen::Vector3d& /*q*/, const Eigen::Vector3d& /*v*/, const RigidBodyParams& /*params*/, double /*gravity*/ = kDefaultGravity);

}  // namespace grip
