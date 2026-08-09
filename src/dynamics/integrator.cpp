#include "dynamics/integrator.hpp"

#include <cassert>
#include <cstddef>

#include "contact/penalty.hpp"
#include "dynamics/mass.hpp"

namespace grip {

RigidBodyState step_body(const RigidBodyState& state, const RigidBodyParams& params, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty, const Eigen::Vector3d& u, double dt, double gravity) {
  const Eigen::Vector3d f = gravity_force(params, gravity) + penalty_force_body(state, shape, plane, penalty) + u;
  const Eigen::Vector3d inv_mass = inverse_mass_diagonal(params);

  RigidBodyState next;
  next.v = state.v + dt * f.cwiseProduct(inv_mass);
  next.q = state.q + dt * next.v;
  return next;
}


StepJacobians step_body_jacobian(const RigidBodyState& state, const RigidBodyParams& params, const BodyShape& shape, const HalfPlane& plane, const PenaltyParams& penalty, const Eigen::Vector3d& /*u*/, double dt, double gravity) {
  // u is accepted (it's the linearization point, mirroring step_body's signature) but unused: every force law is
  // additive in u, so df/du = Id unconditionally and doesn't depend on
  // u's actual value. See docs/derivations/integrator_jacobians.md.
  //
  // Force laws sum, and so do their Jacobians -- that additivity is why
  // contact plugs in here without touching the chain rule below.
  ForceJacobian force_jac = gravity_force_jacobian(state.q, state.v, params, gravity);
  const ForceJacobian contact_jac = penalty_force_body_jacobian(state, shape, plane, penalty);
  force_jac.df_dq += contact_jac.df_dq;
  force_jac.df_dv += contact_jac.df_dv;

  const Eigen::Matrix3d m_inv = inverse_mass_diagonal(params).asDiagonal();
  const Eigen::Matrix3d identity3 = Eigen::Matrix3d::Identity();

  const Eigen::Matrix3d dv_dq = dt * m_inv * force_jac.df_dq;
  const Eigen::Matrix3d dv_dv = identity3 + dt * m_inv * force_jac.df_dv;
  const Eigen::Matrix3d dv_du = dt * m_inv;  // df/du = I: u enters additively

  const Eigen::Matrix3d dq_dq = identity3 + dt * dv_dq;
  const Eigen::Matrix3d dq_dv = dt * dv_dv;
  const Eigen::Matrix3d dq_du = dt * dv_du;

  StepJacobians jac;
  jac.dz_dz.block<3, 3>(0, 0) = dq_dq;
  jac.dz_dz.block<3, 3>(0, 3) = dq_dv;
  jac.dz_dz.block<3, 3>(3, 0) = dv_dq;
  jac.dz_dz.block<3, 3>(3, 3) = dv_dv;

  jac.dz_du.block<3, 3>(0, 0) = dq_du;
  jac.dz_du.block<3, 3>(3, 0) = dv_du;

  return jac;
}


std::vector<RigidBodyState> step_system(const std::vector<RigidBodyState>& states, const std::vector<RigidBodyParams>& params, const std::vector<BodyShape>& shapes, const HalfPlane& plane, const PenaltyParams& penalty, const std::vector<Eigen::Vector3d>& u, double dt, double gravity) {
  assert(states.size() == params.size());
  assert(states.size() == shapes.size());
  assert(states.size() == u.size());

  std::vector<RigidBodyState> next(states.size());
  for (std::size_t i = 0; i < states.size(); ++i) {
    next[i] = step_body(states[i], params[i], shapes[i], plane, penalty, u[i], dt, gravity);
  }
  return next;
}


SystemStepJacobians step_system_jacobian(const std::vector<RigidBodyState>& states, const std::vector<RigidBodyParams>& params, const std::vector<BodyShape>& shapes, const HalfPlane& plane, const PenaltyParams& penalty, const std::vector<Eigen::Vector3d>& u, double dt, double gravity) {
  assert(states.size() == params.size());
  assert(states.size() == shapes.size());
  assert(states.size() == u.size());

  const auto n = static_cast<Eigen::Index>(states.size());
  SystemStepJacobians jac;
  jac.dZ_dZ = SystemStateJacobian::Zero(6 * n, 6 * n);
  jac.dZ_dU = SystemControlJacobian::Zero(6 * n, 3 * n);

  for (Eigen::Index i = 0; i < n; ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    const StepJacobians body_jac = step_body_jacobian(states[idx], params[idx], shapes[idx], plane, penalty, u[idx], dt, gravity);
    jac.dZ_dZ.block<6, 6>(6 * i, 6 * i) = body_jac.dz_dz;
    jac.dZ_dU.block<6, 3>(6 * i, 3 * i) = body_jac.dz_du;
  }
  return jac;
}

}  // namespace grip
