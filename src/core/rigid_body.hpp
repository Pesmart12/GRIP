#pragma once

#include <vector>

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

// Body geometry: the vertices of a convex polygon, expressed in the
// body's own frame relative to the center of mass. Fixed per body, like
// RigidBodyParams -- never differentiated with respect to.
//
// A world-frame vertex is p_i(q) = c + R(theta) * vertices[i], where
// c = q.head<2>(). Vertex order is the contact order: contact i is
// always vertex i, so detection is deterministic without sorting. See
// docs/derivations/contact_detection.md.
struct BodyShape {
  std::vector<Eigen::Vector2d> vertices;
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

// System-level stacking, for N independent bodies: body i occupies
// [6i, 6i+6) of the vector, each body's own (q, v) block laid out per
// StateVector's convention above. Size is runtime (N isn't known at
// compile time), unlike the fixed 6-dim single-body StateVector.
using SystemStateVector = Eigen::VectorXd;

inline SystemStateVector PackSystem(const std::vector<RigidBodyState>& states) {
  SystemStateVector x(6 * states.size());

  for (std::size_t i = 0; i < states.size(); ++i) {
    x.segment<6>(static_cast<Eigen::Index>(6 * i)) = Pack(states[i]);
  }
  
  return x;
}

inline std::vector<RigidBodyState> UnpackSystem(const SystemStateVector& x, std::size_t num_bodies) {
  std::vector<RigidBodyState> states(num_bodies);

  for (std::size_t i = 0; i < num_bodies; ++i) {
    states[i] = Unpack(x.segment<6>(static_cast<Eigen::Index>(6 * i)));
  }

  return states;
}

}  // namespace grip
