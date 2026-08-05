#pragma once

#include <Eigen/Core>

#include "core/rigid_body.hpp"
#include "dynamics/forces.hpp"

namespace grip {

// One symplectic (semi-implicit) Euler step:
//   v_{t+1} = v_t + h * M^-1 * (gravity_force(params, gravity) + u)
//   q_{t+1} = q_t + h * v_{t+1}
//
// Force is evaluated at the old state (q_t, v_t); position is updated
// using the new velocity v_{t+1}. That ordering, not the force law
// itself, is what makes this scheme symplectic. u is a generalized
// wrench (fx, fy, tau) applied at the center of mass. See
// docs/derivations/symplectic_euler.md.
RigidBodyState symplectic_euler_step(const RigidBodyState& state,
                                      const RigidBodyParams& params,
                                      const Eigen::Vector3d& u, double dt,
                                      double gravity = kDefaultGravity);

// x = (q, v) stacked per StateVector's ordering (core/rigid_body.hpp).
using StateJacobian = Eigen::Matrix<double, 6, 6>;
using ControlJacobian = Eigen::Matrix<double, 6, 3>;

struct StepJacobians {
  StateJacobian dx_dx;
  ControlJacobian dx_du;
};

// Analytic d(x_{t+1})/d(x_t) and d(x_{t+1})/du for symplectic_euler_step,
// via the chain rule through ForceJacobian. See
// docs/derivations/integrator_jacobians.md.
StepJacobians symplectic_euler_step_jacobian(const RigidBodyState& state,
                                              const RigidBodyParams& params,
                                              const Eigen::Vector3d& u,
                                              double dt,
                                              double gravity = kDefaultGravity);

}  // namespace grip
