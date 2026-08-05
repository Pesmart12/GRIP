#include "dynamics/integrator.hpp"

#include <cassert>
#include <cstddef>

namespace grip {

RigidBodyState step_body(const RigidBodyState& state, const RigidBodyParams& params, const Eigen::Vector3d& u, double dt, double gravity) {
  const Eigen::Vector3d f = gravity_force(params, gravity) + u;
  const Eigen::Vector3d inv_mass(1.0 / params.mass, 1.0 / params.mass, 1.0 / params.inertia);

  RigidBodyState next;
  next.v = state.v + dt * f.cwiseProduct(inv_mass);
  next.q = state.q + dt * next.v;
  return next;
}

StepJacobians step_body_jacobian(const RigidBodyState& state, const RigidBodyParams& params, const Eigen::Vector3d& /*u*/, double dt, double gravity) {
  // u is accepted (it's the linearization point, mirroring
  // step_body's signature) but unused: every force law is
  // additive in u, so df/du = I unconditionally and doesn't depend on
  // u's actual value. See docs/derivations/integrator_jacobians.md.
  const ForceJacobian force_jac = gravity_force_jacobian(state.q, state.v, params, gravity);
  const Eigen::Vector3d inv_mass(1.0 / params.mass, 1.0 / params.mass, 1.0 / params.inertia);
  const Eigen::Matrix3d m_inv = inv_mass.asDiagonal();
  const Eigen::Matrix3d identity3 = Eigen::Matrix3d::Identity();

  const Eigen::Matrix3d dv_dq = dt * m_inv * force_jac.df_dq;
  const Eigen::Matrix3d dv_dv = identity3 + dt * m_inv * force_jac.df_dv;
  const Eigen::Matrix3d dv_du = dt * m_inv;  // df/du = I: u enters additively

  const Eigen::Matrix3d dq_dq = identity3 + dt * dv_dq;
  const Eigen::Matrix3d dq_dv = dt * dv_dv;
  const Eigen::Matrix3d dq_du = dt * dv_du;

  StepJacobians jac;
  jac.dx_dx.block<3, 3>(0, 0) = dq_dq;
  jac.dx_dx.block<3, 3>(0, 3) = dq_dv;
  jac.dx_dx.block<3, 3>(3, 0) = dv_dq;
  jac.dx_dx.block<3, 3>(3, 3) = dv_dv;

  jac.dx_du.block<3, 3>(0, 0) = dq_du;
  jac.dx_du.block<3, 3>(3, 0) = dv_du;

  return jac;
}

std::vector<RigidBodyState> step_system(const std::vector<RigidBodyState>& states, const std::vector<RigidBodyParams>& params, const std::vector<Eigen::Vector3d>& u, double dt, double gravity) {
  assert(states.size() == params.size());
  assert(states.size() == u.size());

  std::vector<RigidBodyState> next(states.size());
  for (std::size_t i = 0; i < states.size(); ++i) {
    next[i] = step_body(states[i], params[i], u[i], dt, gravity);
  }
  return next;
}

SystemStepJacobians step_system_jacobian(const std::vector<RigidBodyState>& states, const std::vector<RigidBodyParams>& params, const std::vector<Eigen::Vector3d>& u, double dt, double gravity) {
  assert(states.size() == params.size());
  assert(states.size() == u.size());

  const auto n = static_cast<Eigen::Index>(states.size());
  SystemStepJacobians jac;
  jac.dX_dX = SystemStateJacobian::Zero(6 * n, 6 * n);
  jac.dX_dU = SystemControlJacobian::Zero(6 * n, 3 * n);

  for (Eigen::Index i = 0; i < n; ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    const StepJacobians body_jac = step_body_jacobian(states[idx], params[idx], u[idx], dt, gravity);
    jac.dX_dX.block<6, 6>(6 * i, 6 * i) = body_jac.dx_dx;
    jac.dX_dU.block<6, 3>(6 * i, 3 * i) = body_jac.dx_du;
  }
  return jac;
}

}  // namespace grip
