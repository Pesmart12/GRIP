#pragma once

#include <vector>

#include <Eigen/Core>

#include "core/rigid_body.hpp"
#include "dynamics/forces.hpp"

namespace grip {

// One symplectic (semi-implicit) Euler step for a single body:
//   v_{t+1} = v_t + dt * M^-1 * (gravity_force(params, gravity) + u)
//   q_{t+1} = q_t + dt * v_{t+1}
//
// Force is evaluated at the old state (q_t, v_t); position is updated
// using the new velocity v_{t+1}. That ordering, not the force law
// itself, is what makes this scheme symplectic. u is a generalized
// wrench (fx, fy, tau) applied at the center of mass. See
// docs/derivations/symplectic_euler.md.
RigidBodyState step_body(const RigidBodyState& state, const RigidBodyParams& params, const Eigen::Vector3d& u, double dt, double gravity = kDefaultGravity);


// z = (q, v) stacked per StateVector's ordering (core/rigid_body.hpp).
using StateJacobian = Eigen::Matrix<double, 6, 6>;
using ControlJacobian = Eigen::Matrix<double, 6, 3>;


struct StepJacobians {
  StateJacobian dz_dz;
  ControlJacobian dz_du;
};


// Analytic d(z_{t+1})/d(z_t) and d(z_{t+1})/du for step_body, via the
// chain rule through ForceJacobian. See
// docs/derivations/integrator_jacobians.md.
StepJacobians step_body_jacobian(const RigidBodyState& state, const RigidBodyParams& params, const Eigen::Vector3d& u, double dt, double gravity = kDefaultGravity);


// Multiple independent bodies -- still unconstrained, no coupling between
// them. Steps each body with step_body above; no new integration math.
// See docs/derivations/multi_body_system.md.
std::vector<RigidBodyState> step_system(const std::vector<RigidBodyState>& states, const std::vector<RigidBodyParams>& params, const std::vector<Eigen::Vector3d>& u, double dt, double gravity = kDefaultGravity);


// System state Z = concatenation of each body's (q, v), per
// SystemStateVector's convention (core/rigid_body.hpp): body i at
// [6i, 6i+6). Since bodies don't couple yet, dZ_dZ and dZ_dU are exactly
// block-diagonal -- each block is the already-validated per-body
// StepJacobians, placed at (6i, 6i) / (6i, 3i). That stays true once
// step 5 makes a per-body force q/v-dependent (each block just gets
// richer); it stops being true only for actual body-body contact, which
// is handled by a structurally different (IFT) gradient path in steps
// 6-7, not by extending this assembly. See
// docs/derivations/multi_body_system.md.
using SystemStateJacobian = Eigen::MatrixXd;
using SystemControlJacobian = Eigen::MatrixXd;


struct SystemStepJacobians {
  SystemStateJacobian dZ_dZ;
  SystemControlJacobian dZ_dU;
};

SystemStepJacobians step_system_jacobian(const std::vector<RigidBodyState>& states, const std::vector<RigidBodyParams>& params, const std::vector<Eigen::Vector3d>& u, double dt, double gravity = kDefaultGravity);

}  // namespace grip
