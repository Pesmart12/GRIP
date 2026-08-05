#pragma once

#include <Eigen/Core>

namespace grip {

// Generalized position/velocity for a single planar rigid body.
// q = (x, y, theta), v = (vx, vy, omega). theta is left unbounded
// (no wrap to [-pi, pi)) so it stays differentiable across the update.
struct RigidBodyState {
  Eigen::Vector3d q = Eigen::Vector3d::Zero();
  Eigen::Vector3d v = Eigen::Vector3d::Zero();
};

// Fixed body parameters. Kept separate from RigidBodyState: dynamics
// Jacobians are taken with respect to (q, v) and control input, never
// with respect to mass/inertia.
struct RigidBodyParams {
  double mass = 1.0;
  double inertia = 1.0;
};

// Stacked state vector x = (q, v), q in the first three components.
// This ordering is the single source of truth for how state Jacobians
// are laid out -- Pack/Unpack are the only place it's encoded.
using StateVector = Eigen::Matrix<double, 6, 1>;

inline StateVector Pack(const RigidBodyState& state) {
  StateVector x;
  x << state.q, state.v;
  return x;
}

inline RigidBodyState Unpack(const StateVector& x) {
  return RigidBodyState{x.head<3>(), x.tail<3>()};
}

}  // namespace grip
