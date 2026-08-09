#pragma once

#include <vector>

#include <Eigen/Core>

#include "contact/half_plane.hpp"
#include "contact/penalty.hpp"
#include "core/rigid_body.hpp"
#include "dynamics/forces.hpp"

namespace grip {

// One symplectic (semi-implicit) Euler step for a single body:
//   f       = gravity_force(params, gravity) + penalty_force_body(...) + u
//   v_{t+1} = v_t + dt * M^-1 * f
//   q_{t+1} = q_t + dt * v_{t+1}
//
// Force is evaluated at the old state (q_t, v_t); position is updated
// using the new velocity v_{t+1}. That ordering, not the force law
// itself, is what makes this scheme symplectic. u is a generalized
// wrench (fx, fy, tau) applied at the center of mass; the contact force
// is a separate term rather than being folded into u, because u is
// state-independent (df/du = Id) and the contact force is not. See
// docs/derivations/symplectic_euler.md.
//
// A body with no vertices has no contacts, so BodyShape{} makes the
// contact term identically zero and recovers the steps 1-3 behaviour.
RigidBodyState step_body(const RigidBodyState& state, const RigidBodyParams& params, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty, const Eigen::Vector3d& u, double dt, double gravity = kDefaultGravity);


// z = (q, v) stacked per StateVector's ordering (core/rigid_body.hpp).
using StateJacobian = Eigen::Matrix<double, 6, 6>;
using ControlJacobian = Eigen::Matrix<double, 6, 3>;


struct StepJacobians {
  StateJacobian dz_dz;
  ControlJacobian dz_du;
};


// Analytic d(z_{t+1})/d(z_t) and d(z_{t+1})/du for step_body, via the
// chain rule through ForceJacobian. Force laws are additive and so are
// their Jacobians, so adding contact enriches df/dq and df/dv without
// changing the assembly below them. See
// docs/derivations/integrator_jacobians.md.
StepJacobians step_body_jacobian(const RigidBodyState& state, const RigidBodyParams& params, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty, const Eigen::Vector3d& u, double dt, double gravity = kDefaultGravity);


// Multiple bodies against the same static half-plane. They still don't
// couple to each other -- the plane is scenery, not a body, so each
// body's contact force reads only its own state. Steps each body with
// step_body above; no new integration math. shapes is indexed alongside
// params; plane and penalty are shared. See
// docs/derivations/multi_body_system.md.
std::vector<RigidBodyState> step_system(const std::vector<RigidBodyState>& states, const std::vector<RigidBodyParams>& params, const std::vector<BodyShape>& shapes, const HalfPlane& plane, const PenaltyParams& penalty, const std::vector<Eigen::Vector3d>& u, double dt, double gravity = kDefaultGravity);


// System state Z = concatenation of each body's (q, v), per
// SystemStateVector's convention (core/rigid_body.hpp): body i at
// [6i, 6i+6). Since bodies don't couple yet, dZ_dZ and dZ_dU are exactly
// block-diagonal -- each block is the already-validated per-body
// StepJacobians, placed at (6i, 6i) / (6i, 3i). Penalty contact against
// the static plane did not change that: it only made each diagonal
// block richer, since a body's contact force reads its own state alone.
// Block-diagonality stops holding only for actual body-body contact,
// which is handled by a structurally different (IFT) gradient path in
// steps 6-7, not by extending this assembly. See
// docs/derivations/multi_body_system.md.
using SystemStateJacobian = Eigen::MatrixXd;
using SystemControlJacobian = Eigen::MatrixXd;


struct SystemStepJacobians {
  SystemStateJacobian dZ_dZ;
  SystemControlJacobian dZ_dU;
};

SystemStepJacobians step_system_jacobian(const std::vector<RigidBodyState>& states, const std::vector<RigidBodyParams>& params, const std::vector<BodyShape>& shapes, const HalfPlane& plane, const PenaltyParams& penalty, const std::vector<Eigen::Vector3d>& u, double dt, double gravity = kDefaultGravity);

}  // namespace grip
